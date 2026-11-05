"""α³-Bench（AlphaBench）结构剖析。

为什么不做「混入评测集」
------------------------
两个基准的任务形态不同，直接混用会造出一个两边都不成立的评测集：

| 维度 | UAVBench | α³-Bench |
|---|---|---|
| 任务形态 | 单轮：给航点，判断能不能飞 | 多轮：agent/user 交替的对话控制回路 |
| 空间信息 | 圆柱禁飞区 + 多边形围栏 + 三维航点 | **只有 `safety_flags` 布尔值，无禁飞区几何** |
| 工具层 | 任意（本项目自定义） | 固定的 MCP 工具集（read_telemetry / restart_sensor / ...）+ A2A |
| 评测重心 | 安全合规、航点冲突 | 工具一致性、网络鲁棒性、交互质量 |

α³-Bench 的 `initial_state` 里没有 `no_fly_zones`，也没有 `mission.waypoints`，
本项目的 `check_no_fly_zone / check_route_clearance` 在它上面无输入可算。
因此本轮评测集以 UAVBench 为主；α³-Bench 用于**轨迹一致性维度的设计参照**，
并在此处产出可复用的结构画像，供后续扩展多轮评测使用。

输出
----
  data/alphabench_profile.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "raw" / "ab_clean.zip"
OUT_PATH = ROOT / "data" / "alphabench_profile.json"


def parse_actions(raw: Any) -> list[dict]:
    """dialogue 里的 actions 是 Python repr 字符串，不是合法 JSON。"""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        import ast

        val = ast.literal_eval(raw)
        return val if isinstance(val, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_obs(raw: Any) -> list[dict]:
    return parse_actions(raw)


def parse_net(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        import ast

        val = ast.literal_eval(raw)
        return val if isinstance(val, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3000, help="抽样条数")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    if not ZIP_PATH.exists():
        print(f"缺少 {ZIP_PATH}", file=sys.stderr)
        return 2

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json") and "__MACOSX" not in n]
        total = len(names)
        rng = random.Random(args.seed)
        sample = rng.sample(names, min(args.limit, total))

        tool_counter: Counter[str] = Counter()
        a2a_counter: Counter[str] = Counter()
        slice_counter: Counter[str] = Counter()
        weather_counter: Counter[str] = Counter()
        sensor_offline: Counter[str] = Counter()
        flags_counter: Counter[str] = Counter()
        intent_counter: Counter[str] = Counter()

        n_ok = 0
        n_success = 0
        n_with_nfz_flag = 0
        n_with_sep_flag = 0
        turn_counts: list[int] = []
        battery_vals: list[float] = []
        n_mcp_episodes = 0
        n_a2a_episodes = 0

        for name in sample:
            try:
                d = json.loads(zf.read(name))
            except (json.JSONDecodeError, zipfile.BadZipFile):
                continue
            n_ok += 1
            if d.get("success") is True:
                n_success += 1

            st = d.get("initial_state") or {}
            flags = st.get("safety_flags") or {}
            for k, v in flags.items():
                if v:
                    flags_counter[k] += 1
            if flags.get("nfz_violation"):
                n_with_nfz_flag += 1
            if flags.get("separation_breach"):
                n_with_sep_flag += 1

            env = st.get("env") or {}
            if env.get("weather"):
                weather_counter[str(env["weather"])] += 1

            uav = st.get("uav") or {}
            sensors = uav.get("sensors") or {}
            for sname, sval in sensors.items():
                if str(sval).lower() in ("offline", "fail", "failed", "error"):
                    sensor_offline[sname] += 1
            bp = (uav.get("energy") or {}).get("battery_pct")
            if isinstance(bp, (int, float)):
                battery_vals.append(float(bp))

            dialogue = d.get("dialogue") or []
            turn_counts.append(len(dialogue))

            ep_has_mcp = ep_has_a2a = False
            for turn in dialogue:
                if turn.get("intent"):
                    intent_counter[str(turn["intent"])[:48]] += 1
                for act in parse_actions(turn.get("actions")):
                    if not isinstance(act, dict):
                        continue
                    atype = str(act.get("type", "")).lower()
                    if atype == "mcp":
                        ep_has_mcp = True
                        if act.get("name"):
                            tool_counter[str(act["name"])] += 1
                    elif atype == "a2a":
                        ep_has_a2a = True
                        if act.get("task"):
                            a2a_counter[str(act["task"])] += 1
                net = parse_net(turn.get("net"))
                if net.get("slice"):
                    slice_counter[str(net["slice"])] += 1

            n_mcp_episodes += int(ep_has_mcp)
            n_a2a_episodes += int(ep_has_a2a)

    def pct(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    profile = {
        "source": "maferrag/AlphaBench  113k_episodes dataset/113k_episodes dataset.zip",
        "n_total_in_zip": total,
        "n_sampled": len(sample),
        "n_parsed": n_ok,
        "seed": args.seed,
        "structure": {
            "episode_keys": ["initial_state", "dialogue", "final_state", "success", "_meta"],
            "has_no_fly_zone_geometry": False,
            "has_waypoints": False,
            "airspace_fields": ["alt_bounds", "geofence"],
            "note": "initial_state.airspace 只有 alt_bounds 与 geofence，无 no_fly_zones；"
                    "mission 航点不在状态里，任务目标写在 dialogue 的 intent 中",
        },
        "outcome": {
            "success_rate": pct(n_success, n_ok),
            "n_success": n_success,
            "n_parsed": n_ok,
        },
        "safety_flags": {
            "n_with_nfz_violation": n_with_nfz_flag,
            "n_with_separation_breach": n_with_sep_flag,
            "rate_nfz": pct(n_with_nfz_flag, n_ok),
            "rate_sep": pct(n_with_sep_flag, n_ok),
            "note": "仅有布尔结果，没有可用于几何复核的边界数据",
        },
        "dialogue": {
            "avg_turns": round(sum(turn_counts) / len(turn_counts), 2) if turn_counts else 0,
            "max_turns": max(turn_counts) if turn_counts else 0,
            "min_turns": min(turn_counts) if turn_counts else 0,
            "top_intents": intent_counter.most_common(10),
        },
        "tools": {
            "mcp_tool_freq": tool_counter.most_common(20),
            "a2a_task_freq": a2a_counter.most_common(10),
            "episodes_with_mcp": n_mcp_episodes,
            "episodes_with_a2a": n_a2a_episodes,
            "mcp_episode_rate": pct(n_mcp_episodes, n_ok),
            "a2a_episode_rate": pct(n_a2a_episodes, n_ok),
        },
        "network": {
            "slice_freq": slice_counter.most_common(),
        },
        "environment": {
            "weather_freq": weather_counter.most_common(10),
            "sensor_offline_freq": sensor_offline.most_common(10),
            "battery_pct_min": min(battery_vals) if battery_vals else None,
            "battery_pct_mean": round(sum(battery_vals) / len(battery_vals), 2) if battery_vals else None,
        },
        "reuse_decision": {
            "used_in_eval_set": False,
            "reason": "缺少禁飞区几何与航点，无法与 UAVBench 的单轮航点规划任务共用同一套工具层与答案键；"
                      "硬性混入会同时削弱两边的评测效度",
            "reuse_path": "以其 MCP 工具序列作为 tool_use_trajectory_consistency 的设计参照；"
                          "若要正式接入，需另建 MCP 工具映射层与网络条件模拟器",
        },
    }

    out = Path(args.out)
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"抽样 {len(sample)} / 全库 {total}，解析成功 {n_ok}")
    print(f"success_rate={profile['outcome']['success_rate']}, avg_turns={profile['dialogue']['avg_turns']}")
    print(f"MCP episode 占比={profile['tools']['mcp_episode_rate']}, A2A={profile['tools']['a2a_episode_rate']}")
    print("Top MCP 工具:", tool_counter.most_common(8))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
