"""为 UAVBench 全量场景建立可计算索引。

为什么先建索引
--------------
57k 个 JSON 逐个解析要反复解压 + json.loads，直接在上面做分层抽样会重复付费。
索引把「评测真正需要的字段」抽出来落成一张表：
  - 场景元信息（任务类型、风险等级、标签）
  - **几何可计算标注**（航点是否命中禁飞区、沿路径最小斜距、是否越出围栏/高度带）
  - 能量可计算标注（电量余量比例、是否满足返航阈值）

后两类是本项目 ground_truth 的来源：不是人工标注，是几何与物理算出来的。

用法
----
    python evalset/index_scenarios.py --limit 5000      # 试跑
    python evalset/index_scenarios.py                   # 全量（约 2–3 分钟）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import geo  # noqa: E402
from agent.tools import estimate_energy_wh  # noqa: E402

ZIP_PATH = ROOT / "data" / "raw" / "UAVBench_scenarios.zip"
OUT_PATH = ROOT / "data" / "uavbench_index.json"

# 允许的任务类型白名单：只保留语义明确、适合转成任务指令的类型。
# 原始数据里混有少量非英文污染（如 firefight游戏副本_drop）与缺失 mission 的记录，
# 这里显式过滤，并在统计里报告丢弃原因。
MISSION_WHITELIST = {
    "delivery",
    "package_delivery",
    "heavy_load_delivery",
    "emergency_medical_delivery",
    "inspection",
    "pipeline_inspection",
    "bridge_inspection",
    "facade_inspection",
    "tower_spiral_inspection",
    "heli_point_hover_inspection",
    "powerline_corridor",
    "survey",
    "waypoint_survey",
    "search_rescue",
    "search_and_rescue",
    "disaster_recon",
    "firefighting_drop",
    "mapping",
    "area_recon_fixedwing",
    "loiter",
    "relay",
    "satellite_link_relay",
    "border_patrol",
    "convoy_escort",
    "escort",
    "corridor_follow",
    "runway_touch_and_go",
    "touch_and_go",
}


def _iter_scenarios(zip_path: Path):
    zf = zipfile.ZipFile(zip_path)
    names = [
        n
        for n in zf.namelist()
        if n.endswith(".json") and "scenario_jsons/" in n and "__MACOSX" not in n
    ]
    for n in names:
        try:
            yield n, json.loads(zf.read(n))
        except (json.JSONDecodeError, KeyError):
            continue


def build_record(name: str, d: dict) -> tuple[dict | None, str | None]:
    """把一条场景压成索引记录。返回 (record, drop_reason)。"""
    mission = d.get("mission") or {}
    mtype = (mission.get("type") or "").strip()
    if not mtype:
        return None, "missing_mission_type"
    if mtype not in MISSION_WHITELIST:
        return None, "mission_type_not_whitelisted"

    wps = mission.get("waypoints") or []
    if len(wps) < 2:
        return None, "too_few_waypoints"
    if any(len(p) != 3 for p in wps):
        return None, "bad_waypoint_dim"

    airspace = d.get("airspace") or {}
    static_zones = airspace.get("no_fly_zones") or []
    dynamic_zones = airspace.get("no_fly_zones_dynamic") or []
    all_zones = list(static_zones) + list(dynamic_zones)

    # ---- 几何标注 ----
    wp_hits = []
    for i, p in enumerate(wps):
        r = geo.check_zones(p[0], p[1], p[2], static_zones, prefix="NFZ")
        r2 = geo.check_zones(p[0], p[1], p[2], dynamic_zones, prefix="DNFZ")
        ids = r["zone_ids"] + r2["zone_ids"]
        if ids:
            wp_hits.append({"index": i, "point": p, "zone_ids": ids})

    # 段内插值：捕捉「两航点之间穿过禁飞区」而端点未命中的情况
    seg_hit = None
    if not wp_hits and all_zones:
        min_d = geo.min_clearance_along_path(wps, all_zones)
        if min_d <= 0.0:
            seg_hit = {"index": -1, "point": None, "zone_ids": ["SEGMENT_CROSS"]}
    else:
        min_d = geo.min_clearance_along_path(wps, all_zones) if all_zones else float("inf")

    fence = []
    for p in wps:
        g = geo.check_geofence(p[0], p[1], p[2], airspace)
        if not g["inside_geofence"] or not g["inside_altitude_band"]:
            fence.append({"point": p, **g})

    # ---- 能量标注 ----
    est = estimate_energy_wh(wps, d.get("uav") or {})
    reserve_thr = float((d.get("uav") or {}).get("reserve_fraction", 0.3) or 0.3)

    lab = d.get("label") or {}
    faults = []
    for f in d.get("faults") or []:
        if isinstance(f, dict) and f.get("type"):
            faults.append(str(f["type"]))
    weather = (d.get("environment") or {}).get("weather") or {}

    rec = {
        "scenario_id": Path(name).stem,
        "name": d.get("name") or "",
        "mission_type": mtype,
        "risk_level": lab.get("risk_level"),
        "risk_category": lab.get("risk_category") or "",
        "risk_factors": lab.get("factors") or [],
        "tags": d.get("benchmark_tags") or [],
        "n_waypoints": len(wps),
        "waypoints": wps,
        # 几何
        "nfz_present": bool(all_zones),
        "n_static_zones": len(static_zones),
        "n_dynamic_zones": len(dynamic_zones),
        "nfz_hit": bool(wp_hits) or bool(seg_hit),
        "nfz_hit_kind": "waypoint" if wp_hits else ("segment" if seg_hit else "none"),
        "nfz_hit_detail": (wp_hits + ([seg_hit] if seg_hit else [])),
        "nfz_zone_ids": sorted({z for h in wp_hits for z in h["zone_ids"]}),
        "nfz_min_clearance_m": round(float(min_d), 3) if min_d != float("inf") else None,
        "fence_violations": fence,
        "fence_ok": not fence,
        # 能量
        "battery_wh": est["battery_wh"],
        "reserve_fraction": est["reserve_fraction"],
        "reserve_threshold": reserve_thr,
        "battery_sufficient": est["reserve_fraction"] >= reserve_thr,
        "path_distance_m": est["distance_m"],
        # 环境
        "faults": faults,
        "wind_mps": weather.get("wind_mps"),
        "visibility": weather.get("visibility") or "",
        "phenomena": weather.get("phenomena") or [],
        "altitude_band": [
            (airspace.get("altitude_agl_min_m")),
            (airspace.get("altitude_agl_max_m")),
        ],
        "time_budget_s": mission.get("time_budget_s"),
    }
    return rec, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（试跑用）")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    if not ZIP_PATH.exists():
        print(f"缺少 {ZIP_PATH}，请先下载 UAVBench_scenarios.zip", file=sys.stderr)
        return 2

    t0 = time.time()
    records: list[dict] = []
    drops: dict[str, int] = {}
    n_seen = 0

    for name, d in _iter_scenarios(ZIP_PATH):
        n_seen += 1
        if args.limit and n_seen > args.limit:
            break
        rec, reason = build_record(name, d)
        if rec is None:
            drops[reason or "unknown"] = drops.get(reason or "unknown", 0) + 1
            continue
        records.append(rec)
        if n_seen % 5000 == 0:
            print(f"  ... 已处理 {n_seen} 条，保留 {len(records)}", flush=True)

    payload = {
        "meta": {
            "source": "maferrag/UAVBench data/UAVBench_scenarios.zip",
            "n_seen": n_seen,
            "n_kept": len(records),
            "n_dropped": sum(drops.values()),
            "drop_reasons": drops,
            "elapsed_s": round(time.time() - t0, 1),
            "generated_by": "evalset/index_scenarios.py",
        },
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n保留 {len(records)} / 扫描 {n_seen}，丢弃 {sum(drops.values())}")
    print("丢弃原因:", drops)
    print(f"耗时 {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
