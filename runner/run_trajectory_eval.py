"""跑轨迹级评测：多个 Agent 变体 × 低空评测集。

    python runner/run_trajectory_eval.py                    # 零模型，确定性
    python runner/run_trajectory_eval.py --variants guarded --limit 10
    python runner/run_trajectory_eval.py --llm              # 接本地 llama-server

输出
----
  results/trajectory_runs.json     每条 episode 的完整轨迹 + 三档得分
  results/trajectory_summary.json  按变体/分组的汇总
  results/trajectory_table.md      可直接贴进报告的表
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.graph import VARIANTS  # noqa: E402
from agent.llm_client import LLMClient  # noqa: E402
from trajectory.recorder import load_scenario_pool, run_episode  # noqa: E402
from trajectory.scorer import score_episode  # noqa: E402

EVALSET = ROOT / "data" / "evalset" / "lowalt_eval_set.json"
RESULTS = ROOT / "results"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间。n 小时正态近似不可靠，这里用 Wilson。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {}
    safety_pass = sum(r["safety_compliance"] for r in rows)
    lo, hi = wilson_ci(safety_pass, n)
    return {
        "n": n,
        "safety_pass_rate": round(safety_pass / n, 4),
        "safety_ci95": [round(lo, 4), round(hi, 4)],
        "tool_consistency_mean": round(statistics.fmean(r["tool_use_trajectory_consistency"] for r in rows), 4),
        "grounding_mean": round(statistics.fmean(r["rationale_grounding"] for r in rows), 4),
        "decision_match_rate": round(sum(r["decision_match"] for r in rows) / n, 4),
        "decision_acceptable_rate": round(sum(r.get("decision_acceptable", r["decision_match"]) for r in rows) / n, 4),
        "false_alarm_rate": round(sum(r["false_alarm"] for r in rows) / n, 4),
        "parse_ok_rate": round(sum(r["parse_ok"] for r in rows) / n, 4),
        "latency_mean_s": round(statistics.fmean(r["latency_s"] for r in rows), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", default=str(EVALSET))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm", action="store_true", help="接 LLM（默认零模型确定性）")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--tag", default="", help="结果文件后缀标记")
    args = ap.parse_args()

    items = json.loads(Path(args.evalset).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    scenario_ids = {it["metadata"]["scenario_id"] for it in items}
    pool = load_scenario_pool(scenario_ids)
    print(f"评测集 {len(items)} 条，场景命中 {len(pool)} / {len(scenario_ids)}")

    llm = None
    if args.llm:
        llm = LLMClient()
        h = llm.health()
        print("LLM 健康检查:", h)
        if not h.get("ok"):
            print("LLM 不可用，回退到零模型模式", file=sys.stderr)
            llm = None

    RESULTS.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    traces_out: list[dict] = []

    for variant in args.variants:
        rows = []
        for i, item in enumerate(items, 1):
            sid = item["metadata"]["scenario_id"]
            scenario = pool.get(sid)
            if scenario is None:
                continue
            trace = run_episode(item, scenario, variant=variant, llm=llm, max_tokens=args.max_tokens)
            if trace.error:
                print(f"  [{variant}] {item['metadata']['item_id']} 运行失败: {trace.error}", file=sys.stderr)
                continue
            scored = score_episode(trace, item, scenario)
            rows.append(scored)
            traces_out.append(trace.to_dict())
            if i % 10 == 0:
                print(f"  [{variant}] {i}/{len(items)}", flush=True)
        all_rows.extend(rows)
        s = summarize(rows)
        print(f"[{variant}] safety={s['safety_pass_rate']:.2f} ttc={s['tool_consistency_mean']:.2f} "
              f"grounding={s['grounding_mean']:.2f} match={s['decision_match_rate']:.2f}")

    # ---- 汇总 ----
    summary: dict[str, Any] = {"by_variant": {}, "by_variant_group": {}, "config": vars(args)}
    for v in args.variants:
        rows = [r for r in all_rows if r["variant"] == v]
        summary["by_variant"][v] = summarize(rows)
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r.get("group") or "?"].append(r)
        summary["by_variant_group"][v] = {g: summarize(rs) for g, rs in sorted(groups.items())}

    suffix = f"_{args.tag}" if args.tag else ""
    (RESULTS / f"trajectory_runs{suffix}.json").write_text(
        json.dumps({"results": all_rows, "traces": traces_out}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (RESULTS / f"trajectory_summary{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # ---- Markdown 表 ----
    lines = [
        "| 变体 | 安全通过率 | 95% CI | 工具轨迹一致性 | 结论可复核性 | 决策匹配率 | 误报率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for v, s in summary["by_variant"].items():
        if not s:
            continue
        lines.append(
            f"| {v} | {s['safety_pass_rate']:.2%} | "
            f"[{s['safety_ci95'][0]:.2f}, {s['safety_ci95'][1]:.2f}] | "
            f"{s['tool_consistency_mean']:.3f} | {s['grounding_mean']:.3f} | "
            f"{s['decision_match_rate']:.2%} | {s['false_alarm_rate']:.2%} |"
        )
    (RESULTS / f"trajectory_table{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
