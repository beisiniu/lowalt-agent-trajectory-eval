"""离线重放：用修正后的解析逻辑重新评分，不重新调用模型。

为什么可以这样做
----------------
`trajectory_runs_*.json` 里保存了每个 episode 的**原始模型输出**（`answer_raw`）、
完整节点序列、工具调用与证据。评分只是对这些记录的纯函数计算，
所以解析逻辑修好后可以离线重算，不需要再花 28 分钟重跑 160 次推理。

这本身是 LangGraph 轨迹化带来的好处：**过程被完整记录，就能事后追责与重放**。

用法
----
    python runner/rescore.py --runs results/trajectory_runs_llm.json --tag llm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.graph import parse_decision  # noqa: E402
from runner.run_trajectory_eval import summarize  # noqa: E402
from trajectory.recorder import Trace, load_scenario_pool  # noqa: E402
from trajectory.scorer import score_episode  # noqa: E402

ROOT_RESULTS = ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT_RESULTS / "trajectory_runs_llm.json"))
    ap.add_argument("--evalset", default=str(ROOT / "data" / "evalset" / "lowalt_eval_set.json"))
    ap.add_argument("--tag", default="llm")
    args = ap.parse_args()

    runs_path = Path(args.runs)
    payload = json.loads(runs_path.read_text(encoding="utf-8"))
    items = json.loads(Path(args.evalset).read_text(encoding="utf-8"))
    by_id = {it["metadata"]["item_id"]: it for it in items}

    traces = payload["traces"]
    pool = load_scenario_pool({it["metadata"]["scenario_id"] for it in items})

    n_changed = 0
    rows = []
    new_traces = []
    for t in traces:
        item = by_id.get(t["item_id"])
        if item is None:
            continue
        ans = parse_decision(t.get("answer_raw") or "")
        if ans["decision"] != t.get("decision"):
            n_changed += 1
        tr = Trace(
            item_id=t["item_id"],
            variant=t["variant"],
            node_path=t.get("node_path") or [],
            tool_calls=t.get("tool_calls") or [],
            context=t.get("context") or [],
            answer_raw=t.get("answer_raw") or "",
            decision=ans["decision"],
            waypoints=ans.get("waypoints") or [],
            rationale=ans.get("rationale", ""),
            parse_ok=bool(ans.get("parse_ok")),
            latency_s=t.get("latency_s", 0.0),
            n_context=len(t.get("context") or []),
        )
        sid = item["metadata"]["scenario_id"]
        scenario = pool.get(sid)
        if scenario is None:
            continue
        rows.append(score_episode(tr, item, scenario))
        new_traces.append(tr.to_dict())

    print(f"重放 {len(new_traces)} 条，决策被修正 {n_changed} 条")

    variants = list(dict.fromkeys(r["variant"] for r in rows))
    summary = {"by_variant": {}, "by_variant_group": {}}
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        summary["by_variant"][v] = summarize(rs)
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            groups[r.get("group") or "?"].append(r)
        summary["by_variant_group"][v] = {g: summarize(x) for g, x in sorted(groups.items())}

    (ROOT_RESULTS / f"trajectory_runs_{args.tag}.json").write_text(
        json.dumps({"results": rows, "traces": new_traces}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (ROOT_RESULTS / f"trajectory_summary_{args.tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    lines = [
        "| 变体 | 安全通过率 | 95% CI | 工具轨迹一致性 | 结论可复核性 | 决策匹配率 | 可接受决策率 | 误报率 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v, s in summary["by_variant"].items():
        lines.append(
            f"| {v} | {s['safety_pass_rate']:.2%} | "
            f"[{s['safety_ci95'][0]:.2f}, {s['safety_ci95'][1]:.2f}] | "
            f"{s['tool_consistency_mean']:.3f} | {s['grounding_mean']:.3f} | "
            f"{s['decision_match_rate']:.2%} | {s.get('decision_acceptable_rate', 0):.2%} | "
            f"{s['false_alarm_rate']:.2%} |"
        )
    (ROOT_RESULTS / f"trajectory_table_{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
