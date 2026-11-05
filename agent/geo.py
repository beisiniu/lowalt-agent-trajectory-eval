"""低空几何判定：禁飞区命中、地理围栏、航点距离。

设计约束
--------
本模块同时被两处使用：
  1. Agent 的工具层（`agent/tools.py`），模拟"环境查询"；
  2. 评测集答案键生成（`evalset/convert_uavbench.py`），给出 ground_truth。

两者共用一套代码会有"自证"风险（工具错了答案也错，反而互相印证）。
因此 `tests/test_geo.py` 用两种独立方式交叉验证：
  - 圆柱命中：解析公式 vs 三维网格暴力采样；
  - 多边形命中：射线法 vs 角度和（winding number）法。
这与上一个项目「几何物化 + 第二套算法复核」的做法一致。

坐标系说明
----------
UAVBench 场景使用局部 ENU 平面坐标（单位：米），x/y 为水平面，z 为高度。
因此距离用欧氏距离，不做 Haversine（后者需要经纬度）。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


Point2D = Sequence[float]
Point3D = Sequence[float]


def _as_xy(p: Point2D) -> tuple[float, float]:
    return float(p[0]), float(p[1])


def point_in_polygon(x: float, y: float, polygon: Iterable[Point2D]) -> bool:
    """射线法判断点是否在多边形内（边界视为在内，用 epsilon 容差）。

    实现要点：只统计与 +x 方向射线相交的边，用 (y1 > y) != (y2 > y) 做跨越判断，
    避免顶点重复计数。
    """
    pts = [_as_xy(p) for p in polygon]
    if len(pts) < 3:
        return False

    # 边界容差：点到任一边的距离 <= EPS 视为在内
    eps = 1e-9
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if _point_segment_distance(x, y, x1, y1, x2, y2) <= eps:
            return True

    inside = False
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside


def _point_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def cylinder_slant_distance(
    x: float,
    y: float,
    z: float,
    center: Point2D,
    radius_m: float,
    floor_m: float,
    ceiling_m: float,
) -> float:
    """点到有限圆柱体表面的最短斜距（米）。点在体内时返回 0。

    这正是 UAVBench `metrics` 里 `min_nfz_slant_m` 的语义。
    """
    cx, cy = _as_xy(center)
    planar = math.hypot(x - cx, y - cy)

    # 水平方向到圆柱侧面的距离（体内为负，取 0）
    d_radial = max(0.0, planar - radius_m)
    # 垂直方向到 [floor, ceiling] 区间的距离（区间内为 0）
    if z < floor_m:
        d_axial = floor_m - z
    elif z > ceiling_m:
        d_axial = z - ceiling_m
    else:
        d_axial = 0.0

    return math.hypot(d_radial, d_axial)


def in_cylinder(
    x: float,
    y: float,
    z: float,
    center: Point2D,
    radius_m: float,
    floor_m: float,
    ceiling_m: float,
    eps: float = 1e-9,
) -> bool:
    cx, cy = _as_xy(center)
    return (
        math.hypot(x - cx, y - cy) <= radius_m + eps
        and floor_m - eps <= z <= ceiling_m + eps
    )


def _zone_geometry(zone: dict) -> dict | None:
    """把 UAVBench 的 no_fly_zone 条目归一化成统一结构。

    支持 cylinder / polygon 两类；缺字段返回 None（由调用方记录为 schema 噪声）。
    """
    ztype = (zone or {}).get("type", "")
    if ztype == "cylinder":
        if "center" not in zone or "radius_m" not in zone:
            return None
        return {
            "kind": "cylinder",
            "center": zone["center"],
            "radius_m": float(zone["radius_m"]),
            "floor_m": float(zone.get("floor_m", 0.0)),
            "ceiling_m": float(zone.get("ceiling_m", 1e9)),
        }
    if ztype == "polygon":
        pts = zone.get("points") or []
        if len(pts) < 3:
            return None
        return {
            "kind": "polygon",
            "points": pts,
            "floor_m": float(zone.get("floor_m", 0.0)),
            "ceiling_m": float(zone.get("ceiling_m", 1e9)),
        }
    return None


def check_zones(
    x: float,
    y: float,
    z: float,
    zones: Sequence[dict],
    prefix: str = "NFZ",
) -> dict:
    """检查一个三维点是否落入任一禁飞区。

    返回:
        {
          "in_no_fly_zone": bool,
          "zone_ids": [str, ...],       # 命中的禁飞区 ID（按输入顺序编号）
          "min_slant_m": float,          # 到最近禁飞区表面的斜距（米）
          "nearest_zone_id": str | None,
        }
    """
    hits: list[str] = []
    min_dist = math.inf
    nearest: str | None = None

    for idx, raw in enumerate(zones or []):
        geom = _zone_geometry(raw)
        if geom is None:
            continue
        zone_id = raw.get("id") or f"{prefix}-{idx:02d}"

        if geom["kind"] == "cylinder":
            dist = cylinder_slant_distance(
                x, y, z, geom["center"], geom["radius_m"], geom["floor_m"], geom["ceiling_m"]
            )
            hit = in_cylinder(x, y, z, geom["center"], geom["radius_m"], geom["floor_m"], geom["ceiling_m"])
        else:
            inside_poly = point_in_polygon(x, y, geom["points"])
            d_axial = 0.0
            if z < geom["floor_m"]:
                d_axial = geom["floor_m"] - z
            elif z > geom["ceiling_m"]:
                d_axial = z - geom["ceiling_m"]
            dist = d_axial if inside_poly else math.hypot(
                _distance_to_polygon_edges(x, y, geom["points"]), d_axial
            )
            hit = inside_poly and geom["floor_m"] <= z <= geom["ceiling_m"]

        if dist < min_dist:
            min_dist = dist
            nearest = zone_id
        if hit:
            hits.append(zone_id)

    return {
        "in_no_fly_zone": bool(hits),
        "zone_ids": hits,
        "min_slant_m": 0.0 if math.isinf(min_dist) else round(min_dist, 3),
        "nearest_zone_id": nearest,
    }


def _distance_to_polygon_edges(x: float, y: float, polygon: Sequence[Point2D]) -> float:
    pts = [_as_xy(p) for p in polygon]
    n = len(pts)
    best = math.inf
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        best = min(best, _point_segment_distance(x, y, x1, y1, x2, y2))
    return best


def check_geofence(x: float, y: float, z: float, airspace: dict) -> dict:
    """检查点是否越出地理围栏与高度边界。

    UAVBench 的 geofence 是多边形列表（可能多个），高度用 altitude_agl_min_m/max_m。
    """
    polygons = []
    for item in (airspace or {}).get("geofence") or []:
        if item.get("type") == "polygon" and len(item.get("points") or []) >= 3:
            polygons.append(item["points"])

    inside_poly = True
    if polygons:
        inside_poly = any(point_in_polygon(x, y, poly) for poly in polygons)

    alt_min = float((airspace or {}).get("altitude_agl_min_m", 0.0))
    alt_max = float((airspace or {}).get("altitude_agl_max_m", 1e9))
    in_altitude = alt_min <= z <= alt_max

    return {
        "inside_geofence": inside_poly,
        "inside_altitude_band": in_altitude,
        "altitude_band": [alt_min, alt_max],
    }


def path_distance(waypoints: Sequence[Point3D]) -> float:
    """航点序列的三维折线长度（米）。单点或空返回 0。"""
    if not waypoints or len(waypoints) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        total += math.dist(
            (float(a[0]), float(a[1]), float(a[2])),
            (float(b[0]), float(b[1]), float(b[2])),
        )
    return round(total, 3)


def min_clearance_along_path(waypoints: Sequence[Point3D], zones: Sequence[dict]) -> float:
    """沿航点折线（含段内插值）到最近禁飞区的最小斜距。

    只检查离散航点会漏掉「两点之间穿过禁飞区」的情况，
    因此按固定步长（默认 2 m）对每段做线性插值采样。
    """
    if not waypoints:
        return math.inf
    samples = densify_path(waypoints, step_m=2.0)
    best = math.inf
    for p in samples:
        r = check_zones(p[0], p[1], p[2], zones)
        best = min(best, r["min_slant_m"])
    return best


def densify_path(waypoints: Sequence[Point3D], step_m: float = 2.0) -> list[tuple[float, float, float]]:
    """把折线按 step_m 加密成采样点序列（首尾点必含）。"""
    if not waypoints:
        return []
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in waypoints]
    if len(pts) == 1:
        return pts

    out: list[tuple[float, float, float]] = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a, b)
        n = max(1, int(math.ceil(seg / step_m)))
        for i in range(1, n + 1):
            t = i / n
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t))
    return out
