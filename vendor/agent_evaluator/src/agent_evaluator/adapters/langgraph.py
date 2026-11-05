"""LangGraph 平台适配器。

上游状态
--------
本文件在 CaptainJi/agent_evaluator 原仓库中只有一行注释 `# LangGraph适配器`，
README 宣称支持 LangGraph，但既没有实现，也没有示例配置，
`cli.create_adapter()` 也只接了 dify。本项目补齐了这三处。

实现要点
--------
适配器契约是 `invoke(input: str) -> AdapterResponse`，只给一段用户输入。
而 LangGraph 的一次运行还需要「场景」（禁飞区几何、电量参数）才能调工具，
因此适配器按 `api_config.dataset_path` 载入评测集，用 `user_input` 反查场景。

字段映射（这是「图轨迹」进入评测框架的通道）：
    answer      <- state["answer"]          模型输出的决策
    contexts    <- state["context"]         工具证据，供 Ragas faithfulness 使用
    tool_calls  <- state["tool_calls"]      结构化工具调用，含参数与返回值
    metadata    <- state["node_path"] 等    节点序列 + 完整轨迹

需要特别指出：上游 `EvalSample.to_ragas_single_turn()` 在 `contexts` 为空时
会用 `reference_contexts` 兜底。这意味着一个「工具调了但证据没回传」的 Agent
反而可能拿到更高的 faithfulness —— 这个 fallback 行为本身会被本项目用作
「通用指标不可信」的证据之一（见 docs/EVAL_REPORT.md）。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from agent_evaluator.adapters.base import AdapterResponse, PerformanceMetrics, PlatformAdapter

# vendor/agent_evaluator/src/agent_evaluator/adapters/langgraph.py
# -> parents[5] = 项目根 lowalt-agent-eval/
_PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _resolve(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


class LangGraphAdapter(PlatformAdapter):
    """把 LangGraph StateGraph 接入 agent_evaluator。"""

    def __init__(self, api_config: dict[str, Any]):
        super().__init__(api_config)

        dataset_path = _resolve(api_config.get("dataset_path"))
        if dataset_path is None or not dataset_path.exists():
            raise ValueError(f"dataset_path 无效: {api_config.get('dataset_path')}")

        self.items: list[dict] = json.loads(dataset_path.read_text(encoding="utf-8"))
        self.by_input: dict[str, dict] = {it["user_input"].strip(): it for it in self.items}

        self.variant = api_config.get("variant", "guarded")
        self.use_llm = bool(api_config.get("use_llm", False))
        self.max_tokens = int(api_config.get("max_tokens", 384))
        self.zip_path = _resolve(api_config.get("uavbench_zip"))

        self._llm = None
        if self.use_llm:
            from agent.llm_client import LLMClient

            self._llm = LLMClient(
                base_url=api_config.get("llm_base"),
                model=api_config.get("llm_model"),
            )

        self._pool: dict[str, dict] | None = None
        self._graphs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _ensure_pool(self) -> dict[str, dict]:
        if self._pool is None:
            from trajectory.recorder import load_scenario_pool

            ids = {it["metadata"]["scenario_id"] for it in self.items}
            self._pool = load_scenario_pool(ids, self.zip_path) if self.zip_path else {}
        return self._pool

    def _get_graph(self, scenario_id: str):
        if scenario_id not in self._graphs:
            from agent.graph import build_graph

            scenario = self._ensure_pool().get(scenario_id)
            if scenario is None:
                raise KeyError(f"场景未找到: {scenario_id}")
            self._graphs[scenario_id] = build_graph(
                scenario, llm=self._llm, variant=self.variant, max_tokens=self.max_tokens
            )
        return self._graphs[scenario_id]

    def _match_item(self, text: str) -> dict:
        item = self.by_input.get(text.strip())
        if item is not None:
            return item
        for k, v in self.by_input.items():
            if k.startswith(text.strip()[:40]):
                return v
        raise KeyError(f"评测集中找不到对应输入: {text[:60]}...")

    # ------------------------------------------------------------------
    async def invoke(self, input: str, stream: bool = False, **kwargs: Any) -> AdapterResponse:
        item = self._match_item(input)
        sid = item["metadata"]["scenario_id"]
        graph = self._get_graph(sid)

        init = {
            "mission_text": item["user_input"],
            "scenario_id": sid,
            "scenario": self._ensure_pool()[sid],
            "parsed": {},
            "route": "",
            "needs_geo_check": False,
            "needs_policy_query": False,
            "tool_calls": [],
            "tool_results": {},
            "context": [],
            "answer": "",
            "node_path": [],
        }

        t0 = time.time()
        # graph.invoke 同步阻塞，放线程里避免卡住事件循环
        state = await asyncio.to_thread(graph.invoke, init)
        elapsed = time.time() - t0

        raw_calls = list(state.get("tool_calls") or [])
        node_path = list(state.get("node_path") or [])
        tool_calls = [
            {
                "name": c.get("tool"),
                "args": c.get("args") or {},
                "result": c.get("result") or {},
                "node": c.get("node"),
                "step": c.get("step"),
                "ok": c.get("ok"),
            }
            for c in raw_calls
        ]

        return AdapterResponse(
            answer=state.get("answer", ""),
            contexts=list(state.get("context") or []),
            tool_calls=tool_calls,
            metadata={
                "platform": "langgraph",
                "variant": self.variant,
                "item_id": item["metadata"].get("item_id"),
                "scenario_id": sid,
                "node_path": node_path,
                "n_nodes": len(node_path),
                "n_tool_calls": len(tool_calls),
                "n_contexts": len(state.get("context") or []),
                "expected_decision": (item.get("expected") or {}).get("decision"),
                "state_dump": {
                    "parsed": state.get("parsed") or {},
                    "route": state.get("route"),
                    "tool_results": _safe_json(state.get("tool_results") or {}),
                },
            },
            performance=PerformanceMetrics(total_time=elapsed),
        )


def _safe_json(obj: Any, limit: int = 4000) -> Any:
    """压成可序列化、体积可控的形式，避免报告 JSON 爆掉。"""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return obj if len(s) <= limit else s[:limit]
    except (TypeError, ValueError):
        return str(obj)[:limit]
