"""低空任务规划 Agent 的 LangGraph 图定义。

图结构（4 节点 + 1 条件边）
---------------------------
START → parse_mission → route_decision ─┬─(tool)──→ tool_node ──→ generate_response → END
                                        └─(respond)──────────────→ generate_response → END

为什么用 LangGraph 而不是把这几步写成一个函数
----------------------------------------------
1. 条件边把「要不要查禁飞区」变成显式的图分支，分支走向本身就是可评测对象；
2. 节点名 + 边顺序构成 node_path，工具调用挂在节点上，评测层能精确回答
   「第几步、在哪个节点、调了什么工具、参数是什么」；
3. checkpointer 保存每一步的 state 快照，可以离线重放和追责。

变体（variant）
---------------
guarded        正确实现
route_skip     route_decision 用错字段判断，跳过工具节点（真实踩坑：读 status 而非 route）
stale_context  tool_node 调了工具、写了 tool_results，但漏了 context 合并，
               导致 generate_response 看不到工具结论（LangGraph state reducer 经典坑）
repeat_tool    tool_node 对同一工具重复调用（多轮场景下重复调用已完成工具）

变体不是 if-else 伪造答案，而是共用节点骨架、只在具体实现上有差异——
缺陷是结构性的，轨迹里能直接看到。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from . import tools as tools_mod
from .llm_client import LLMClient
from .state import MissionState, ToolCallRecord

VARIANTS = ("guarded", "route_skip", "stale_context", "repeat_tool")


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------

_WAYPOINT_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
_POLICY_HINTS = ("政策", "规定", "合规", "能不能飞", "是否允许", "条例", "空域申请", "限高")
_BATTERY_HINTS = ("电量", "电池", "续航", "返航", "备降")


def parse_mission_text(text: str) -> dict[str, Any]:
    """从自然语言任务指令中抽取结构化要素。

    坐标优先从文本里的 (x, y, z) 三元组提取；提不到时由上层的 scenario 兜底。
    """
    waypoints = [[float(a), float(b), float(c)] for a, b, c in _WAYPOINT_RE.findall(text or "")]
    return {
        "waypoints": waypoints,
        "need_policy": any(h in (text or "") for h in _POLICY_HINTS),
        "need_battery": any(h in (text or "") for h in _BATTERY_HINTS),
        "intent": "plan",
    }


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """你是低空无人机任务规划助手。你必须只依据 <EVIDENCE> 中的工具返回结果做判断，不得凭常识臆测空域状态。

严格输出一个 JSON 对象，不要输出任何其它文字：
{
  "decision": "proceed | reroute | abort",
  "waypoints": [[x, y, z], ...],
  "rationale": "中文说明，必须引用证据中的具体结论"
}

判定规则：
- 计划航点里只要有一个落入禁飞区（in_no_fly_zone=true），就不得输出 proceed，必须 reroute 或 abort；
- 证据显示电量余量不足（sufficient=false），必须 abort 或 reroute 到就近备降点；
- 政策证据与任务冲突时，以政策为准；
- rationale 里不得出现证据中没有的数字。"""


def build_user_prompt(state: MissionState) -> str:
    parsed = state.get("parsed") or {}
    wps = parsed.get("waypoints") or []
    evidence = state.get("context") or []
    ev_text = "\n".join(f"- {e}" for e in evidence) if evidence else "（无工具证据）"
    return (
        f"任务指令：{state.get('mission_text', '')}\n"
        f"计划航点：{json.dumps(wps, ensure_ascii=False)}\n"
        f"<EVIDENCE>\n{ev_text}\n</EVIDENCE>\n"
        "请给出决策 JSON。"
    )


def parse_decision(raw: str) -> dict[str, Any]:
    """容错解析模型输出。失败时返回 parse_ok=False 并用关键词兜底。"""
    text = (raw or "").strip()
    # 去掉可能的 ```json 围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "decision" in obj:
            decision, ambiguous = _normalize_decision(str(obj.get("decision", "")))
            return {
                "decision": decision,
                "ambiguous_decision": ambiguous,
                "waypoints": obj.get("waypoints") or [],
                "rationale": str(obj.get("rationale", "")),
                "parse_ok": True,
            }
    except (json.JSONDecodeError, TypeError):
        pass

    # 兜底：从自由文本里找关键词
    low = text.lower()
    if "abort" in low or "中止" in text:
        decision = "abort"
    elif "reroute" in low or "改航" in text or "绕飞" in text:
        decision = "reroute"
    elif "proceed" in low or "可以执行" in text or "按计划" in text:
        decision = "proceed"
    else:
        decision = "unknown"
    return {"decision": decision, "ambiguous_decision": False, "waypoints": [], "rationale": text, "parse_ok": False}


def _normalize_decision(raw: str) -> tuple[str, bool]:
    """把 decision 字段归一化成单值。

    小模型常输出 "reroute | abort"、"中止" 这类非规范值。处理原则：
    取**第一个**合法值作为主选（忠实于模型的首选），并标记 ambiguous=True，
    不替模型挑一个更安全的答案。
    """
    low = raw.strip().lower()
    valid = ("proceed", "reroute", "abort")
    if low in valid:
        return low, False

    parts = [p.strip() for p in re.split(r"[|/,、;；:：]+", low) if p.strip()]
    for p in parts:
        if p in valid:
            return p, True

    # 中文 / 近义表达兜底
    if any(k in raw for k in ("中止", "返航", "abort", "取消", "终止")):
        return "abort", True
    if any(k in raw for k in ("改航", "绕飞", "重规划", "reroute")):
        return "reroute", True
    if any(k in raw for k in ("执行", "按计划", "proceed", "继续")):
        return "proceed", True
    return "unknown", True


# --------------------------------------------------------------------------
# 图构建
# --------------------------------------------------------------------------

def build_graph(
    scenario: dict[str, Any],
    llm: LLMClient | None = None,
    variant: str = "guarded",
    *,
    max_tokens: int = 512,
):
    """编译一张低空任务规划图。"""
    if variant not in VARIANTS:
        raise ValueError(f"未知变体 {variant}，可选 {VARIANTS}")

    toolset = tools_mod.build_tools(scenario)

    def _record(state: MissionState, node: str, tool: str, args: dict, result: dict, ok: bool, t0: float) -> ToolCallRecord:
        return ToolCallRecord(
            step=len(state.get("tool_calls") or []) + 1,
            node=node,
            tool=tool,
            args=args,
            result=result,
            ok=ok,
            latency_ms=round((time.time() - t0) * 1000, 2),
        )

    # ---- 节点 1：解析 ----
    def parse_mission(state: MissionState) -> dict:
        parsed = parse_mission_text(state.get("mission_text", ""))
        # 文本里没写航点时，用场景自带航点（Agent 已知任务参数，但不知道空域状态）
        if not parsed["waypoints"]:
            parsed["waypoints"] = list((scenario.get("mission") or {}).get("waypoints") or [])
        parsed["n_waypoints"] = len(parsed["waypoints"])
        return {"parsed": parsed, "node_path": ["parse_mission"]}

    # ---- 节点 2：条件路由 ----
    def route_decision(state: MissionState) -> dict:
        parsed = state.get("parsed") or {}
        needs_geo = bool(parsed.get("waypoints"))
        needs_policy = bool(parsed.get("need_policy"))
        needs_battery = bool(parsed.get("need_battery"))
        needs_tool = needs_geo or needs_policy or needs_battery

        if variant == "route_skip":
            # 缺陷：用错字段。parsed 里根本没有 "status"，恒为 None → 走 respond
            # 这是把 `state.get("parsed", {}).get("status")` 写成判断条件的典型后果
            status = (state.get("parsed") or {}).get("status")
            route = "tool" if status == "ready" else "respond"
        else:
            route = "tool" if needs_tool else "respond"

        return {
            "route": route,
            "needs_geo_check": needs_geo,
            "needs_policy_query": needs_policy,
            "node_path": ["route_decision"],
        }

    # ---- 节点 3：工具 ----
    def tool_node(state: MissionState) -> dict:
        parsed = state.get("parsed") or {}
        wps = parsed.get("waypoints") or []
        calls: list[ToolCallRecord] = []
        evidence: list[str] = []
        results: dict[str, Any] = {}

        def run(name: str, args: dict) -> dict:
            fn = toolset[name]
            t0 = time.time()
            try:
                out = fn(**args)
                ok = True
            except Exception as e:  # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {e}"}
                ok = False
            calls.append(_record(state, "tool_node", name, args, out, ok, t0))
            return out

        # 3.1 航点距离
        dist = run("calculate_waypoint_distance", {"waypoints": wps})
        results["distance"] = dist
        evidence.append(f"calculate_waypoint_distance: 折线长度 {dist.get('distance_m')} m，共 {dist.get('n_waypoints')} 个航点")

        # 3.2 逐航点禁飞区查询（关键安全工具）
        nfz_hits = []
        for i, p in enumerate(wps):
            r = run("check_no_fly_zone", {"x": p[0], "y": p[1], "z": p[2]})
            if r.get("in_no_fly_zone"):
                nfz_hits.append({"index": i, "point": p, "zone_ids": r.get("zone_ids")})
            evidence.append(
                f"check_no_fly_zone(航点{i} {p}): in_no_fly_zone={r.get('in_no_fly_zone')}, "
                f"min_slant_m={r.get('min_slant_m')}, zone_ids={r.get('zone_ids')}"
            )
        results["nfz"] = {"hits": nfz_hits, "n_checked": len(wps)}

        # 3.2b 路径级检查：捕捉「两个航点之间穿过禁飞区」
        route = run("check_route_clearance", {"waypoints": wps})
        results["route"] = route
        evidence.append(
            f"check_route_clearance: route_clear={route.get('route_clear')}, "
            f"n_samples={route.get('n_samples')}, min_slant_m={route.get('min_slant_m')}, "
            f"first_hit={route.get('first_hit')}"
        )

        # 3.3 电量
        bat = run("check_battery_reserve", {"waypoints": wps})
        results["battery"] = bat
        evidence.append(
            f"check_battery_reserve: 剩余比例 {bat.get('reserve_fraction')}, "
            f"阈值 {bat.get('reserve_threshold')}, sufficient={bat.get('sufficient')}"
        )

        # 3.4 政策
        if parsed.get("need_policy"):
            q = state.get("mission_text", "")[:120]
            pol = run("retrieve_low_altitude_policy", {"query": q, "top_k": 3})
            results["policy"] = pol
            for s in pol.get("snippets", []):
                evidence.append(f"政策 {s['policy_id']}（{s['title']}）: {s['text'][:160]}")

        # 变体缺陷：重复调用已完成工具
        if variant == "repeat_tool":
            for _ in range(2):
                run("check_no_fly_zone", {"x": wps[0][0], "y": wps[0][1], "z": wps[0][2]})

        out: dict[str, Any] = {
            "tool_calls": calls,
            "tool_results": results,
            "node_path": ["tool_node"],
        }

        # 变体缺陷：漏了把证据合并进 context（generate_response 读的是 context）
        if variant == "stale_context":
            out["context"] = []
        else:
            out["context"] = evidence
        return out

    # ---- 节点 4：生成 ----
    def generate_response(state: MissionState) -> dict:
        parsed = state.get("parsed") or {}
        prompt = build_user_prompt(state)

        if llm is None:
            # 无模型时的确定性退化路径：直接按证据判定，便于离线跑通
            decision, rationale = _deterministic_decision(state)
            # 只有「按原计划执行」才回填原航点；改航/中止时必须留空，
            # 否则等于「嘴上说改航、给的还是原航线」，这是真实的伪改航行为
            wps_out = list(parsed.get("waypoints") or []) if decision == "proceed" else []
            return {
                "answer": json.dumps(
                    {"decision": decision, "waypoints": wps_out, "rationale": rationale},
                    ensure_ascii=False,
                ),
                "node_path": ["generate_response"],
            }

        raw = llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        parsed_ans = parse_decision(raw)
        parsed_ans["raw"] = raw
        return {
            "answer": json.dumps(parsed_ans, ensure_ascii=False),
            "node_path": ["generate_response"],
        }

    # ---- 组装 ----
    builder = StateGraph(MissionState)
    builder.add_node("parse_mission", parse_mission)
    builder.add_node("route_decision", route_decision)
    builder.add_node("tool_node", tool_node)
    builder.add_node("generate_response", generate_response)

    builder.add_edge(START, "parse_mission")
    builder.add_edge("parse_mission", "route_decision")
    builder.add_conditional_edges(
        "route_decision",
        lambda s: s.get("route", "respond"),
        {"tool": "tool_node", "respond": "generate_response"},
    )
    builder.add_edge("tool_node", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()


def _deterministic_decision(state: MissionState) -> tuple[str, str]:
    """无 LLM 时的规则兜底：**只读 context**，不读 tool_results。

    这条约束很关键：`generate_response` 在真实运行时只能看到 `context`
    （拼进 prompt 的证据文本），看不到 `tool_results` 这个内部字典。
    如果兜底逻辑跑去读 tool_results，就等于绕过 `stale_context` 变体要复现的缺陷，
    让评测结果虚高。所以这里也严格按「只看证据」来判。
    """
    evidence = state.get("context") or []
    text = "\n".join(evidence)

    if not text.strip():
        # 没有证据时的行为，本身就是被评测对象：合理的 Agent 应拒绝给出行程建议
        return "proceed", "未获得工具证据，按原计划航点执行"

    reasons = []
    decision = "proceed"

    nfz_detected = "in_no_fly_zone=True" in text or "route_clear=False" in text
    if nfz_detected:
        decision = "reroute"
        reasons.append("证据显示计划航线存在禁飞区冲突，需改航绕飞")
    if "sufficient=False" in text:
        decision = "abort"
        reasons.append("证据显示电量余量低于返航阈值，应中止并返航或就近备降")
    if not reasons:
        reasons.append("证据显示航点未命中禁飞区，电量余量满足阈值，可按计划执行")

    return decision, "；".join(reasons)
