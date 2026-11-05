"""流式数据累积器"""

import time
from dataclasses import dataclass, field
from typing import Any

from agent_evaluator.adapters.base import AdapterResponse, PerformanceMetrics


@dataclass
class StreamingAccumulator:
    """流式响应累积器，用于累积SSE事件并构建完整响应"""

    answer: str = ""  # 累积的完整答案
    contexts: list[str] = field(default_factory=list)  # 检索到的上下文
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 工具调用
    metadata: dict[str, Any] = field(default_factory=dict)  # 完整元数据

    # 性能指标相关
    first_token_time: float | None = None  # 首token时间戳（相对于请求开始）
    last_token_time: float | None = None  # 最后token时间戳
    token_timestamps: list[float] = field(default_factory=list)  # 每个token的时间戳
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_price: float | None = None
    currency: str | None = None
    elapsed_time: float | None = None  # 从workflow_finished事件获取的总耗时

    def accumulate(self, event: dict[str, Any], current_time: float) -> None:
        """
        累积SSE事件

        Args:
            event: SSE事件字典
            current_time: 当前时间戳（相对于请求开始）
        """
        event_type = event.get("event")

        if event_type == "message":
            # Chat API的message事件：累积答案片段
            answer_chunk = event.get("answer", "")
            if answer_chunk:
                self.answer += answer_chunk
                # 记录首token时间
                if self.first_token_time is None:
                    self.first_token_time = current_time
                self.last_token_time = current_time
                self.token_timestamps.append(current_time)
        
        elif event_type == "agent_message":
            # Chat API的agent_message事件：累积Agent答案片段
            answer_chunk = event.get("answer", "")
            if answer_chunk:
                self.answer += answer_chunk
                if self.first_token_time is None:
                    self.first_token_time = current_time
                self.last_token_time = current_time
                self.token_timestamps.append(current_time)
        
        elif event_type == "message_end":
            # Chat API的message_end事件：提取元数据和性能指标
            metadata = event.get("metadata", {})
            self.metadata.update(metadata)
            
            # 提取usage信息
            usage = metadata.get("usage", {})
            if usage:
                self.total_tokens = usage.get("total_tokens", 0)
                self.input_tokens = usage.get("prompt_tokens", 0)
                self.output_tokens = usage.get("completion_tokens", 0)
        
        elif event_type == "workflow_started":
            # Workflow API的workflow_started事件：记录工作流开始信息
            data = event.get("data", {})
            self.metadata.setdefault("workflow", {}).update({
                "workflow_id": data.get("workflow_id"),
                "workflow_run_id": data.get("id"),
                "started_at": data.get("created_at"),
            })
        
        elif event_type == "node_started":
            # Workflow API的node_started事件：记录节点开始信息
            data = event.get("data", {})
            node_info = {
                "node_id": data.get("node_id"),
                "node_type": data.get("node_type"),
                "title": data.get("title"),
                "index": data.get("index"),
                "started_at": data.get("created_at"),
            }
            self.metadata.setdefault("nodes_started", []).append(node_info)
        
        elif event_type == "text_chunk":
            # Workflow API的text_chunk事件：累积文本片段
            data = event.get("data", {})
            text_chunk = data.get("text", "")
            if text_chunk:
                self.answer += text_chunk
                # 记录首token时间
                if self.first_token_time is None:
                    self.first_token_time = current_time
                self.last_token_time = current_time
                self.token_timestamps.append(current_time)

        elif event_type == "workflow_finished":
            # 提取完整元数据
            data = event.get("data", {})
            self.metadata.update(data)

            # 提取性能指标
            self.elapsed_time = data.get("elapsed_time")
            self.total_tokens = data.get("total_tokens", 0)

            # 提取执行元数据（如果存在）
            execution_metadata = data.get("execution_metadata", {})
            if execution_metadata:
                self.total_tokens = execution_metadata.get("total_tokens", self.total_tokens)
                self.total_price = execution_metadata.get("total_price")
                self.currency = execution_metadata.get("currency")

            # 从nodes中汇总token信息（如果有）
            nodes = data.get("nodes", [])
            total_input_tokens = 0
            total_output_tokens = 0
            for node in nodes:
                # 尝试从process_data.usage获取
                process_data = node.get("process_data", {})
                usage = process_data.get("usage", {})
                if usage:
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)
                else:
                    # 尝试从outputs.usage获取
                    outputs = node.get("outputs", {})
                    usage = outputs.get("usage", {})
                    if usage:
                        total_input_tokens += usage.get("prompt_tokens", 0)
                        total_output_tokens += usage.get("completion_tokens", 0)

            if total_input_tokens > 0:
                self.input_tokens = total_input_tokens
            if total_output_tokens > 0:
                self.output_tokens = total_output_tokens
            
            # 从workflow_finished的outputs中提取retrieved_contexts（如果存在）
            # 注意：workflow_finished的outputs可能包含最终输出，也可能包含retrieved_contexts
            outputs = data.get("outputs", {})
            if isinstance(outputs, dict):
                if "retrieved_contexts" in outputs:
                    contexts = outputs["retrieved_contexts"]
                    if isinstance(contexts, list):
                        # 合并到已有的contexts中（避免重复）
                        existing_contexts = set(self.contexts)
                        for ctx in contexts:
                            if ctx and ctx not in existing_contexts:
                                self.contexts.append(ctx)
                                existing_contexts.add(ctx)
                    elif isinstance(contexts, str) and contexts not in self.contexts:
                        self.contexts.append(contexts)
                
                # 也检查outputs中是否有其他可能的上下文字段
                # 某些Dify版本可能使用不同的字段名
                for key in ["contexts", "context", "retrieved_context"]:
                    if key in outputs and key != "retrieved_contexts":
                        value = outputs[key]
                        if isinstance(value, list):
                            for ctx in value:
                                if ctx and ctx not in self.contexts:
                                    self.contexts.append(ctx)
                        elif isinstance(value, str) and value not in self.contexts:
                            self.contexts.append(value)

        elif event_type == "node_finished":
            # 提取节点输出（可能包含contexts和tool_calls）
            data = event.get("data", {})
            outputs = data.get("outputs", {})

            # 提取contexts（如果存在）
            # 支持多种可能的字段名
            contexts_found = False
            for key in ["retrieved_contexts", "contexts", "context", "retrieved_context"]:
                if key in outputs:
                    contexts = outputs[key]
                    contexts_found = True
                    if isinstance(contexts, list):
                        # 避免重复添加
                        existing_contexts = set(self.contexts)
                        for ctx in contexts:
                            if ctx and ctx not in existing_contexts:
                                self.contexts.append(ctx)
                                existing_contexts.add(ctx)
                    elif isinstance(contexts, str) and contexts not in self.contexts:
                        self.contexts.append(contexts)
                    break  # 找到第一个匹配的字段就停止
            
            # 如果outputs是字典但没有找到contexts字段，检查是否有嵌套结构
            if not contexts_found and isinstance(outputs, dict):
                # 检查outputs中是否有其他可能包含contexts的字段
                for key, value in outputs.items():
                    if isinstance(value, dict):
                        # 嵌套字典中查找
                        if "retrieved_contexts" in value:
                            nested_contexts = value["retrieved_contexts"]
                            if isinstance(nested_contexts, list):
                                existing_contexts = set(self.contexts)
                                for ctx in nested_contexts:
                                    if ctx and ctx not in existing_contexts:
                                        self.contexts.append(ctx)
                                        existing_contexts.add(ctx)
                            elif isinstance(nested_contexts, str) and nested_contexts not in self.contexts:
                                self.contexts.append(nested_contexts)

            # 提取tool_calls（如果存在）
            if "tool_calls" in outputs:
                tool_calls = outputs["tool_calls"]
                if isinstance(tool_calls, list):
                    self.tool_calls.extend(tool_calls)

            # 更新元数据
            self.metadata.setdefault("nodes", []).append(data)

    def to_performance_metrics(self, start_time: float) -> PerformanceMetrics:
        """
        转换为性能指标对象

        Args:
            start_time: 请求开始时间戳

        Returns:
            PerformanceMetrics对象
        """
        # 计算总耗时
        if self.elapsed_time is not None:
            total_time = self.elapsed_time
        elif self.last_token_time is not None:
            total_time = self.last_token_time
        else:
            total_time = 0.0

        # 计算TTFT
        ttft = self.first_token_time if self.first_token_time is not None else None

        # 计算流式延迟（相邻token之间的时间差）
        streaming_latency: list[float] = []
        if len(self.token_timestamps) > 1:
            for i in range(1, len(self.token_timestamps)):
                latency = self.token_timestamps[i] - self.token_timestamps[i - 1]
                streaming_latency.append(latency)

        # 估算input/output tokens（如果metadata中没有明确提供）
        # 注意：这是粗略估算，实际应该从API响应中获取
        input_tokens = self.input_tokens if self.input_tokens > 0 else 0
        output_tokens = self.output_tokens if self.output_tokens > 0 else 0

        return PerformanceMetrics(
            total_time=total_time,
            time_to_first_token=ttft,
            streaming_latency=streaming_latency,
            total_tokens=self.total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_price=self.total_price,
            currency=self.currency,
        )

    def to_adapter_response(self, start_time: float) -> AdapterResponse:
        """
        转换为AdapterResponse对象

        Args:
            start_time: 请求开始时间戳

        Returns:
            AdapterResponse对象
        """
        return AdapterResponse(
            answer=self.answer,
            contexts=self.contexts,
            tool_calls=self.tool_calls,
            metadata=self.metadata,
            performance=self.to_performance_metrics(start_time),
        )

