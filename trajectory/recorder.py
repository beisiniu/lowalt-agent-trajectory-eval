"""轨迹记录器：跑一个 episode，把 LangGraph 的 state 转成可评分的 Trace。

为什么要单独抽 Trace
-------------------
Ragas 的样本是 (user_input, response, contexts) 三元组，丢掉了过程信息。
Trace 保留 node_path / tool_calls / context / decision，轨迹评分器只依赖它，
因此可以脱离 LLM judge 独立运行、可复现、可回归。
"""

from __future__ import annotations

import json
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.graph import build_graph, parse_decision
from agent.llm_client import LLMClient

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "raw" / "UAVBench_scenarios.zip"


@dataclass
class Trace:
    item_id: str
    variant: str
    node_path: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    answer_raw: str = ""
    decision: str = ""
    waypoints: list = field(default_factory=list)
    rationale: str = ""
    parse_ok: bool = False
    latency_s: float = 0.0
    n_context: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_scenario_pool(scenario_ids: set[str], zip_path: Path = ZIP_PATH) -> dict[str, dict]:
    """按需从 zip 里取出指定场景（不整包解压）。"""
    if not zip_path.exists():
        raise FileNotFoundError(f"缺少 {zip_path}")
    pool: dict[str, dict | None] = {sid: None for sid in scenario_ids}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json") or "__MACOSX" in name:
                continue
            stem = Path(name).stem
            if stem in pool and pool[stem] is None:
                try:
                    pool[stem] = json.loads(zf.read(name))
                except json.JSONDecodeError:
                    continue
    return {k: v for k, v in pool.items() if v is not None}


def run_episode(
    item: dict,
    scenario: dict,
    variant: str = "guarded",
    llm: LLMClient | None = None,
    max_tokens: int = 512,
) -> Trace:
    """执行一条评测样本，返回结构化轨迹。"""
    graph = build_graph(scenario, llm=llm, variant=variant, max_tokens=max_tokens)
    init = {
        "mission_text": item["user_input"],
        "scenario_id": item["metadata"]["scenario_id"],
        "scenario": scenario,
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
    try:
        state = graph.invoke(init)
    except Exception as e:  # noqa: BLE001
        return Trace(
            item_id=item["metadata"]["item_id"],
            variant=variant,
            error=f"{type(e).__name__}: {e}",
            latency_s=round(time.time() - t0, 2),
        )
    latency = time.time() - t0

    ans = parse_decision(state.get("answer", ""))
    return Trace(
        item_id=item["metadata"]["item_id"],
        variant=variant,
        node_path=list(state.get("node_path") or []),
        tool_calls=list(state.get("tool_calls") or []),
        context=list(state.get("context") or []),
        answer_raw=state.get("answer", ""),
        decision=ans.get("decision", ""),
        waypoints=ans.get("waypoints") or [],
        rationale=ans.get("rationale", ""),
        parse_ok=bool(ans.get("parse_ok")),
        latency_s=round(latency, 2),
        n_context=len(state.get("context") or []),
    )
