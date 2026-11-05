"""分析「空改航」：模型声明 reroute，但航点没改。

为什么这个分析重要
------------------
项目里四个 Agent 变体的缺陷是刻意设计的，容易被质疑「bug 是你自己埋的」。
这个脚本给出反证：**在正确实现的 `guarded` 变体上，模型自发产生了大量失效**。

判据（全部可几何复算）：
  1. decision == "reroute"
  2. 模型输出的 waypoints 与场景原航点逐点比较
     - 完全一致        -> 「空改航」（no-op reroute）
     - 有改动但落在禁飞区 -> 「假改航」
     - 有改动且全程避开   -> 真正改航成功

为什么这不可能是变体代码造成的
------------------------------
`guarded` 的 `generate_response` 节点只在 `decision == "proceed"` 时回填原航点，
其余情况 `waypoints` **完全来自 LLM 输出**，代码里没有任何生成坐标的逻辑。
且 `repeat_tool` 变体的工具层与 `guarded` 不同，却出现几乎相同的比例——
说明该行为属于模型，不属于变体。

用法
----
    python runner/analyze_noop_reroute.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import geo  # noqa: E402
from trajectory.recorder import load_scenario_pool  # noqa: E402

RESULTS = ROOT / "results"


def norm(wps) -> list[list[float]]:
    out = []
    for p in wps or []:
        if isinstance(p, (list, tuple)) and len(p) == 3:
            out.append([round(float(v), 2) for v in p])
    return out


def classify(new: list[list[float]], orig: list[list[float]], scenario: dict) -> str:
    if not new:
        return "no_waypoints"          # 声明改航但没给航线
    if new == orig:
        return "noop"                  # 原样返回
    zones = (scenario.get("airspace") or {}).get("no_fly_zones") or []
    dzones = (scenario.get("airspace") or {}).get("no_fly_zones_dynamic") or []
    for p in new:
        if geo.check_zones(p[0], p[1], p[2], zones, prefix="NFZ")["in_no_fly_zone"]:
            return "reroute_into_nfz"
        if geo.check_zones(p[0], p[1], p[2], dzones, prefix="DNFZ")["in_no_fly_zone"]:
            return "reroute_into_nfz"
    return "reroute_ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RESULTS / "trajectory_runs_llm.json"))
    ap.add_argument("--evalset", default=str(ROOT / "data" / "evalset" / "lowalt_eval_set.json"))
    ap.add_argument("--out", default=str(RESULTS / "noop_reroute_analysis.json"))
    args = ap.parse_args()

    payload = json.loads(Path(args.runs).read_text(encoding="utf-8"))
    items = {x["metadata"]["item_id"]: x for x in json.loads(Path(args.evalset).read_text(encoding="utf-8"))}
    pool = load_scenario_pool({x["metadata"]["scenario_id"] for x in items.values()})

    variants = list(dict.fromkeys(r["variant"] for r in payload["results"]))
    report: dict = {"variants": {}, "examples": []}

    for v in variants:
        traces = {t["item_id"]: t for t in payload["traces"] if t["variant"] == v}
        rows = [r for r in payload["results"] if r["variant"] == v]
        reroute = [r for r in rows if r["predicted_decision"] == "reroute"]
        bucket: Counter[str] = Counter()
        by_group: dict[str, Counter[str]] = defaultdict(Counter)

        for r in reroute:
            tid = r["item_id"]
            t = traces.get(tid)
            item = items.get(tid)
            if t is None or item is None:
                continue
            sc = pool[item["metadata"]["scenario_id"]]
            orig = norm((sc.get("mission") or {}).get("waypoints"))
            new = norm(t.get("waypoints"))
            kind = classify(new, orig, sc)
            bucket[kind] += 1
            by_group[r["group"]][kind] += 1

            if kind in ("noop", "reroute_into_nfz"):
                report["examples"].append(
                    {
                        "variant": v,
                        "item_id": tid,
                        "group": r["group"],
                        "kind": kind,
                        "orig_waypoints": orig,
                        "model_waypoints": new,
                    }
                )

        report["variants"][v] = {
            "n_episodes": len(rows),
            "n_reroute": len(reroute),
            "breakdown": dict(bucket),
            "by_group": {g: dict(c) for g, c in by_group.items()},
            "noop_rate_among_reroute": round(bucket["noop"] / len(reroute), 4) if reroute else None,
            "true_success_rate_among_reroute": (
                round(bucket["reroute_ok"] / len(reroute), 4) if reroute else None
            ),
        }
        print(f"[{v}] reroute={len(reroute)}  {dict(bucket)}")
        if reroute:
            print(f"        空改航占比 {bucket['noop']/len(reroute):.1%}，"
                  f"真正改航成功 {bucket['reroute_ok']}/{len(reroute)}")

    report["note"] = (
        "空改航（noop）= decision 为 reroute 但 waypoints 与场景原航点逐点一致；"
        "reroute_into_nfz = 有新航点但落在禁飞区内。"
        "guarded 变体的 generate_response 仅在 decision==proceed 时回填原航点，"
        "其余情况下 waypoints 完全来自 LLM 输出，代码中不存在生成坐标的逻辑。"
    )
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
