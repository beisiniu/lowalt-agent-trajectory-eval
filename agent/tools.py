"""低空任务规划 Agent 的工具集。

四个工具都通过 `build_tools(scenario)` 绑定到具体场景：
工具是「环境」的一部分，Agent 只能通过工具查询环境，不能直接读 scenario 字典。
这一点很重要——否则 Agent 等于开卷考试，轨迹评测也就失去意义。

工具契约（返回 dict，字段固定）
------------------------------
check_no_fly_zone(x, y, z)      -> in_no_fly_zone / zone_ids / min_slant_m / nearest_zone_id
retrieve_low_altitude_policy(q) -> snippets[{policy_id, title, text, score}]
calculate_waypoint_distance(w)  -> distance_m / legs
check_battery_reserve(w)        -> energy_wh / reserve_wh / reserve_fraction / sufficient

能耗模型是**简化估算**，仅用于「电量是否够」的定性判断，不是真实能耗模型。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from . import geo

POLICY_PATH = Path(__file__).parent / "policies" / "lowalt_policy.md"


# --------------------------------------------------------------------------
# 政策语料：切分 + 字符 bigram 检索（零依赖，对中文有效）
# --------------------------------------------------------------------------

def load_policy_snippets(path: Path = POLICY_PATH) -> list[dict[str, str]]:
    """把政策文件按 `## Pxx 标题` 切成片段。"""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    chunks = re.split(r"^##\s+", text, flags=re.MULTILINE)[1:]
    out: list[dict[str, str]] = []
    for chunk in chunks:
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        m = re.match(r"(P\d+)\s*(.*)", header)
        policy_id = m.group(1) if m else header[:8]
        title = (m.group(2) if m else header).strip()
        body = "\n".join(lines[1:]).strip()
        # 去掉引用块标记，保留正文
        body = "\n".join(ln.lstrip("> ").rstrip() for ln in body.splitlines()).strip()
        if body:
            out.append({"policy_id": policy_id, "title": title, "text": body})
    return out


def _bigrams(text: str) -> dict[str, int]:
    """字符 bigram 词频。中文无需分词，bigram 已足够区分政策片段。"""
    clean = re.sub(r"\s+", "", text)
    if len(clean) < 2:
        return {clean: 1} if clean else {}
    grams: dict[str, int] = {}
    for i in range(len(clean) - 1):
        g = clean[i : i + 2]
        grams[g] = grams.get(g, 0) + 1
    return grams


class PolicyIndex:
    """极简 TF-IDF + 余弦相似度检索器。"""

    def __init__(self, snippets: list[dict[str, str]]):
        self.snippets = snippets
        self._docs = [_bigrams(s["title"] + s["text"]) for s in snippets]
        df: dict[str, int] = {}
        for doc in self._docs:
            for g in doc:
                df[g] = df.get(g, 0) + 1
        n = max(1, len(self._docs))
        self._idf = {g: math.log((n + 1) / (c + 1)) + 1.0 for g, c in df.items()}

    def _vec(self, grams: dict[str, int]) -> dict[str, float]:
        return {g: (1 + math.log(c)) * self._idf.get(g, 1.0) for g, c in grams.items()}

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        qv = self._vec(_bigrams(query))
        if not qv:
            return []
        qnorm = math.sqrt(sum(v * v for v in qv.values())) or 1.0

        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            dv = self._vec(doc)
            dnorm = math.sqrt(sum(v * v for v in dv.values())) or 1.0
            dot = sum(qv[g] * dv.get(g, 0.0) for g in qv)
            scored.append((dot / (qnorm * dnorm), i))

        scored.sort(reverse=True)
        out = []
        for score, i in scored[:top_k]:
            s = self.snippets[i]
            out.append(
                {
                    "policy_id": s["policy_id"],
                    "title": s["title"],
                    "text": s["text"],
                    "score": round(float(score), 4),
                }
            )
        return out


_INDEX_CACHE: PolicyIndex | None = None


def get_policy_index() -> PolicyIndex:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = PolicyIndex(load_policy_snippets())
    return _INDEX_CACHE


# --------------------------------------------------------------------------
# 能耗估算
# --------------------------------------------------------------------------

def estimate_energy_wh(waypoints: Sequence[Sequence[float]], uav: dict) -> dict[str, float]:
    """简化能耗估算：巡航段按悬停功率的 1.15 倍计（前飞诱导功率的经验系数）。"""
    dist = geo.path_distance(waypoints)
    v_max = float(uav.get("max_speed_mps", 10.0) or 10.0)
    v_cruise = max(1.0, v_max * 0.6)
    hours = dist / v_cruise / 3600.0

    model = uav.get("battery_model") or {}
    hover_w = float(model.get("hover_power_W", 80.0) or 80.0)
    cruise_w = hover_w * 1.15

    battery_wh = float(uav.get("battery_Wh", 0.0) or 0.0)
    energy = cruise_w * hours
    reserve = battery_wh - energy
    frac = (reserve / battery_wh) if battery_wh > 0 else 0.0
    return {
        "distance_m": dist,
        "cruise_speed_mps": round(v_cruise, 2),
        "energy_wh": round(energy, 3),
        "battery_wh": round(battery_wh, 3),
        "reserve_wh": round(reserve, 3),
        "reserve_fraction": round(frac, 4),
    }


# --------------------------------------------------------------------------
# 工具工厂
# --------------------------------------------------------------------------

def build_tools(scenario: dict[str, Any]) -> dict[str, Callable[..., dict]]:
    """把工具绑定到给定场景，返回 name -> callable 的映射。"""
    airspace = scenario.get("airspace") or {}
    uav = scenario.get("uav") or {}
    static_zones = airspace.get("no_fly_zones") or []
    dynamic_zones = airspace.get("no_fly_zones_dynamic") or []

    def check_no_fly_zone(x: float, y: float, z: float = 0.0) -> dict:
        """查询坐标是否落入禁飞区（含静态与动态禁飞区）。"""
        x, y, z = float(x), float(y), float(z)
        static_res = geo.check_zones(x, y, z, static_zones, prefix="NFZ")
        dynamic_res = geo.check_zones(x, y, z, dynamic_zones, prefix="DNFZ")

        hits = static_res["zone_ids"] + dynamic_res["zone_ids"]
        min_slant = min(static_res["min_slant_m"], dynamic_res["min_slant_m"])
        nearest = (
            static_res["nearest_zone_id"]
            if static_res["min_slant_m"] <= dynamic_res["min_slant_m"]
            else dynamic_res["nearest_zone_id"]
        )
        return {
            "in_no_fly_zone": bool(hits),
            "zone_ids": hits,
            "min_slant_m": round(float(min_slant), 3),
            "nearest_zone_id": nearest,
        }

    def retrieve_low_altitude_policy(query: str, top_k: int = 3) -> dict:
        """检索低空政策片段。"""
        hits = get_policy_index().search(str(query), top_k=int(top_k))
        return {"query": str(query), "snippets": hits, "n_hits": len(hits)}

    def calculate_waypoint_distance(waypoints: list) -> dict:
        """计算航点序列的三维折线距离。"""
        pts = waypoints or []
        legs = []
        for a, b in zip(pts, pts[1:]):
            legs.append(round(math.dist([float(v) for v in a], [float(v) for v in b]), 3))
        return {"distance_m": geo.path_distance(pts), "n_waypoints": len(pts), "legs": legs}

    def check_battery_reserve(waypoints: list) -> dict:
        """估算执行给定航线后的电量余量，判断是否满足返航保留阈值。"""
        est = estimate_energy_wh(waypoints or [], uav)
        threshold = float(uav.get("reserve_fraction", 0.3) or 0.3)
        return {
            **est,
            "reserve_threshold": threshold,
            "sufficient": est["reserve_fraction"] >= threshold,
        }

    def check_route_clearance(waypoints: list, step_m: float = 2.0) -> dict:
        """沿整条航线做路径级禁飞区检查（含航点之间插值）。

        逐点查询只能发现「航点落在禁飞区」，发现不了「两个安全航点之间穿过去」。
        真实低空规划系统需要的是后者，所以单独提供路径级检查。
        """
        pts = waypoints or []
        if not pts:
            return {"route_clear": True, "n_samples": 0, "first_hit": None, "min_slant_m": None}

        samples = geo.densify_path(pts, step_m=float(step_m))
        first_hit = None
        min_slant = math.inf
        for i, p in enumerate(samples):
            r = geo.check_zones(p[0], p[1], p[2], static_zones, prefix="NFZ")
            r2 = geo.check_zones(p[0], p[1], p[2], dynamic_zones, prefix="DNFZ")
            ids = r["zone_ids"] + r2["zone_ids"]
            min_slant = min(min_slant, r["min_slant_m"], r2["min_slant_m"])
            if ids and first_hit is None:
                first_hit = {
                    "sample_index": i,
                    "point": [round(v, 2) for v in p],
                    "zone_ids": ids,
                }

        return {
            "route_clear": first_hit is None,
            "n_samples": len(samples),
            "step_m": float(step_m),
            "first_hit": first_hit,
            "min_slant_m": 0.0 if math.isinf(min_slant) else round(float(min_slant), 3),
        }

    return {
        "check_no_fly_zone": check_no_fly_zone,
        "check_route_clearance": check_route_clearance,
        "retrieve_low_altitude_policy": retrieve_low_altitude_policy,
        "calculate_waypoint_distance": calculate_waypoint_distance,
        "check_battery_reserve": check_battery_reserve,
    }


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "check_no_fly_zone": {
        "description": "查询给定坐标是否落入禁飞区，返回布尔值、命中禁飞区 ID 与最小斜距",
        "params": {"x": "float 东向坐标(m)", "y": "float 北向坐标(m)", "z": "float 高度(m)"},
    },
    "check_route_clearance": {
        "description": "沿整条航线做路径级禁飞区检查（含航点之间插值），返回航线上第一个冲突点",
        "params": {"waypoints": "list[[x,y,z], ...]", "step_m": "float 插值步长，默认2m"},
    },
    "retrieve_low_altitude_policy": {
        "description": "检索低空飞行政策片段",
        "params": {"query": "str 检索问题", "top_k": "int 返回条数，默认3"},
    },
    "calculate_waypoint_distance": {
        "description": "计算航点序列的折线距离",
        "params": {"waypoints": "list[[x,y,z], ...]"},
    },
    "check_battery_reserve": {
        "description": "估算航线能耗与剩余电量，判断是否满足返航保留阈值",
        "params": {"waypoints": "list[[x,y,z], ...]"},
    },
}
