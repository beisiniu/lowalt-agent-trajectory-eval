"""LangGraph 状态定义与轨迹记录结构。

为什么状态要显式记录 node_path / tool_calls
-------------------------------------------
通用 RAG 评测只看 (question, answer, contexts)，拿不到「Agent 走了哪些节点、
在哪个节点调了哪个工具、参数是什么」。把这两类信息作为 state 的一等字段，
评测层才能从 checkpointer 的 state_history 里还原完整轨迹。

注意 `context` 字段：它是 `generate_response` 唯一读取的证据来源。
工具节点必须显式把结果写进 `context`，否则模型看不到工具结论——
这正是 `stale_context` 变体要复现的真实失效。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ToolCallRecord(TypedDict, total=False):
    """一次工具调用的结构化记录（轨迹评分的最小单元）。"""

    step: int
    node: str
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]
    ok: bool
    latency_ms: float


class MissionState(TypedDict, total=False):
    # ---- 输入 ----
    mission_text: str
    scenario_id: str
    scenario: dict[str, Any]

    # ---- parse_mission 产出 ----
    parsed: dict[str, Any]
    # parsed 结构:
    #   waypoints: [[x,y,z], ...]
    #   target_xy: [x, y] | None
    #   need_policy: bool
    #   intent: str            # plan / check / query
    #   constraints: dict      # battery_ok, time_budget_s ...

    # ---- route_decision 产出 ----
    route: str
    needs_geo_check: bool
    needs_policy_query: bool

    # ---- tool_node 产出 ----
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    tool_results: dict[str, Any]
    # 供 generate_response 读取的证据；必须由 tool_node 显式追加
    context: Annotated[list[str], operator.add]

    # ---- generate_response 产出 ----
    answer: str

    # ---- 轨迹 ----
    node_path: Annotated[list[str], operator.add]
    # 运行期标记：用于触发受控缺陷（仅 defective 变体读取）
    _fault: str


def new_state(mission_text: str, scenario_id: str, scenario: dict[str, Any]) -> MissionState:
    """构造初始状态，保证 list 字段有初值（LangGraph 的 add reducer 需要）。"""
    return MissionState(
        mission_text=mission_text,
        scenario_id=scenario_id,
        scenario=scenario,
        parsed={},
        route="",
        needs_geo_check=False,
        needs_policy_query=False,
        tool_calls=[],
        tool_results={},
        context=[],
        answer="",
        node_path=[],
    )
