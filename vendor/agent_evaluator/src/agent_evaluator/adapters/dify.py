"""Dify平台适配器"""

import asyncio
import json
import time
from typing import Any

from agent_evaluator.adapters.base import AdapterResponse, PerformanceMetrics, PlatformAdapter
from agent_evaluator.adapters.streaming import StreamingAccumulator
from agent_evaluator.utils.logger import get_logger

logger = get_logger(__name__)

# Chat API的上下文字段名（根据官方文档）
CHAT_API_CONTEXT_FIELD = "retriever_resources"

# Workflow API的上下文字段可能名称（按优先级排序）
WORKFLOW_API_CONTEXT_FIELDS = ["retrieved_contexts", "contexts", "context", "retrieved_context"]


class DifyAdapter(PlatformAdapter):
    """Dify平台适配器"""

    def __init__(self, api_config: dict[str, Any], show_streaming_content: bool = False):
        """
        初始化Dify适配器
        
        Args:
            api_config: API配置字典
            show_streaming_content: 是否显示流式输出的详细内容
        """
        super().__init__(api_config)
        self.show_streaming_content = show_streaming_content

    def _get_headers(self) -> dict[str, str]:
        """获取Dify API请求头"""
        headers = super()._get_headers()
        api_key = self.api_config.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(self, input: str, response_mode: str, **kwargs: Any) -> dict[str, Any]:
        """构建请求payload - 消除重复代码"""
        inputs = kwargs.get("inputs", {})
        query = inputs.get("query") if "query" in inputs else input
        # 如果query来自inputs，payload中的query用占位符
        payload_query = "-" if "query" in inputs else query
        
        return {
            "inputs": inputs,
            "query": payload_query,
            "response_mode": response_mode,
            "conversation_id": kwargs.get("conversation_id"),
            "user": kwargs.get("user", "agent-evaluator"),
        }

    async def invoke(
        self,
        input: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> AdapterResponse:
        """
        调用Dify API

        Args:
            input: 用户输入（如果inputs中包含query，则此参数可能被忽略）
            stream: 是否使用流式输出
            **kwargs: 其他参数，包括：
                - conversation_id: 对话ID（可选）
                - user: 用户ID（可选）
                - app_id: 应用ID（可选，某些API可能需要）
                - inputs: 工作流输入变量字典（可选），如果包含query字段，则使用inputs.query作为实际查询

        Returns:
            AdapterResponse对象
        """
        logger.debug(f"调用Dify API，流式模式: {stream}, 输入长度: {len(input)}")
        method = self._invoke_streaming if stream else self._invoke_non_streaming
        return await method(input, **kwargs)

    async def _invoke_non_streaming(
        self,
        input: str,
        **kwargs: Any,
    ) -> AdapterResponse:
        """非流式调用"""
        if not self._client:
            raise RuntimeError("Adapter must be used as async context manager")

        base_url = self.api_config.get("base_url", "https://api.dify.ai/v1")
        path = kwargs.get("path", "chat-messages")
        url = f"{base_url}/{path}"
        payload = self._build_payload(input, "blocking", **kwargs)

        logger.debug(f"发送非流式请求到: {url}")
        start_time = time.time()
        response = await self._client.post(url, json=payload)
        response.raise_for_status()

        response_data = response.json()
        total_time = time.time() - start_time
        logger.debug(f"API调用完成，耗时: {total_time:.3f}秒")
        
        is_workflow = self._is_workflow_api(path)
        if is_workflow:
            # Workflow API响应结构: {workflow_run_id, task_id, data: {...}}
            workflow_data = response_data.get("data", {})
            self._log_response_diagnostics(workflow_data, path)
            
            # Workflow API没有answer字段，答案在outputs中
            outputs = workflow_data.get("outputs", {})
            answer = self._extract_answer_from_workflow_outputs(outputs)
            contexts = self._extract_contexts_from_workflow_outputs(outputs)
            
            metadata = {
                "workflow_run_id": response_data.get("workflow_run_id"),
                "task_id": response_data.get("task_id"),
                "workflow_id": workflow_data.get("workflow_id"),
                "status": workflow_data.get("status"),
                "created_at": workflow_data.get("created_at"),
                "finished_at": workflow_data.get("finished_at"),
                **response_data,
            }
            
            performance = self._extract_performance_metrics_from_workflow(workflow_data, total_time)
        else:
            # Chat API响应结构: {answer, retriever_resources, ...}
            self._log_response_diagnostics(response_data, path)
            
            answer = response_data.get("answer", "")
            contexts = self._extract_contexts_from_chat_response(response_data)
            
            metadata = {
                "message_id": response_data.get("message_id"),
                "conversation_id": response_data.get("conversation_id"),
                "created_at": response_data.get("created_at"),
                **response_data,
            }
            
            performance = self._extract_performance_metrics_from_chat(response_data, total_time)
        
        return AdapterResponse(
            answer=answer,
            contexts=contexts,
            metadata=metadata,
            performance=performance,
        )

    def _is_workflow_api(self, path: str) -> bool:
        """判断是否为Workflow API路径"""
        return "workflow" in path.lower() or path.endswith("/workflows/run")

    def _extract_answer_from_workflow_outputs(self, outputs: dict[str, Any]) -> str:
        """从Workflow API的outputs中提取答案"""
        if not isinstance(outputs, dict):
            return ""
        
        # 尝试常见的答案字段
        for key in ["text", "answer", "output", "result", "content"]:
            if key in outputs:
                value = outputs[key]
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict):
                    # 如果是字典，尝试提取text或content
                    return str(value.get("text") or value.get("content") or value)
        
        # 如果没有找到，返回outputs的字符串表示
        return str(outputs) if outputs else ""

    def _extract_contexts_from_workflow_outputs(self, outputs: dict[str, Any]) -> list[str]:
        """从Workflow API的outputs中提取上下文"""
        if not isinstance(outputs, dict):
            return []
        
        contexts = []
        for field in WORKFLOW_API_CONTEXT_FIELDS:
            if field in outputs:
                value = outputs[field]
                if isinstance(value, list):
                    contexts.extend([str(ctx) for ctx in value if ctx])
                elif isinstance(value, str) and value:
                    contexts.append(value)
                break  # 找到第一个匹配的字段就停止
        
        return contexts

    def _extract_contexts_from_chat_response(self, data: dict[str, Any]) -> list[str]:
        """从Chat API响应中提取retriever_resources"""
        retriever_resources = data.get(CHAT_API_CONTEXT_FIELD, [])
        if not isinstance(retriever_resources, list):
            return []
        
        contexts = []
        for resource in retriever_resources:
            if isinstance(resource, dict):
                # 提取content字段（如果有）
                content = resource.get("content") or resource.get("chunk_content")
                if content:
                    contexts.append(str(content))
            elif resource:
                contexts.append(str(resource))
        return contexts

    def _log_response_diagnostics(self, data: dict[str, Any], path: str) -> None:
        """记录API响应诊断信息"""
        logger.debug(f"Dify API响应字段: {list(data.keys())}")
        
        if self._is_workflow_api(path):
            # Workflow API诊断：检查data.outputs中的上下文字段
            outputs = data.get("outputs", {})
            if isinstance(outputs, dict):
                for field in WORKFLOW_API_CONTEXT_FIELDS:
                    if field in outputs:
                        contexts_count = len(outputs.get(field, [])) if isinstance(outputs.get(field), list) else 1
                        logger.debug(f"API响应outputs中包含{field}，数量: {contexts_count}")
                        return
            logger.warning("API响应outputs中未找到retrieved_contexts相关字段，可能是应用未配置RAG/知识库检索")
        else:
            # Chat API诊断
            if CHAT_API_CONTEXT_FIELD in data:
                resources_count = len(data.get(CHAT_API_CONTEXT_FIELD, []))
                logger.debug(f"API响应中包含{CHAT_API_CONTEXT_FIELD}，数量: {resources_count}")
                return
            
            logger.warning(f"API响应中未找到{CHAT_API_CONTEXT_FIELD}字段，可能是应用未配置RAG/知识库检索")
            possible_fields = [k for k in data.keys() if "context" in k.lower() or "retriev" in k.lower()]
            if possible_fields:
                logger.debug(f"发现可能的上下文相关字段: {possible_fields}")

    def _extract_performance_metrics_from_workflow(self, workflow_data: dict[str, Any], total_time: float) -> PerformanceMetrics | None:
        """从Workflow API响应中提取性能指标"""
        # Workflow API的性能指标在data对象中
        elapsed_time = workflow_data.get("elapsed_time")
        if elapsed_time is not None:
            total_time = elapsed_time
        
        total_tokens = workflow_data.get("total_tokens", 0)
        
        return PerformanceMetrics(
            total_time=total_time,
            total_tokens=total_tokens,
            input_tokens=0,  # Workflow API不单独提供input/output tokens
            output_tokens=0,
        )

    def _extract_performance_metrics_from_chat(self, data: dict[str, Any], total_time: float) -> PerformanceMetrics | None:
        """从Chat API响应中提取性能指标"""
        if "metadata" not in data:
            return None
        
        meta = data["metadata"]
        usage = meta.get("usage", {})
        performance = PerformanceMetrics(
            total_time=total_time,
            total_tokens=usage.get("total_tokens", 0),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
        logger.debug(f"性能指标: tokens={performance.total_tokens}, 输入={performance.input_tokens}, 输出={performance.output_tokens}")
        return performance

    async def _invoke_streaming(
        self,
        input: str,
        **kwargs: Any,
    ) -> AdapterResponse:
        """流式调用"""
        if not self._client:
            raise RuntimeError("Adapter must be used as async context manager")

        base_url = self.api_config.get("base_url", "https://api.dify.ai/v1")
        path = kwargs.get("path", "chat-messages")
        url = f"{base_url}/{path}"
        payload = self._build_payload(input, "streaming", **kwargs)
        is_workflow = self._is_workflow_api(path)

        # 获取超时时间：流式响应需要更长的超时（基础超时的10倍，最少300秒）
        base_timeout = self.api_config.get("timeout", 30.0)
        streaming_timeout = max(base_timeout * 10, 300.0)

        logger.debug(f"发送流式请求到: {url} (API类型: {'Workflow' if is_workflow else 'Chat'}), 超时: {streaming_timeout}秒")
        start_time = time.time()
        accumulator = StreamingAccumulator()

        async def _read_stream():
            """内部函数：读取流式响应"""
            async with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue

                    event_data = line[6:]  # 移除 "data: " 前缀
                    if event_data == "[DONE]":
                        break

                    try:
                        event = json.loads(event_data)
                        current_time = time.time() - start_time
                        if self.show_streaming_content:
                            self._log_streaming_event(event, current_time, is_workflow)
                        accumulator.accumulate(event, current_time)
                        
                        # Chat API的message_end事件包含retriever_resources
                        if not is_workflow and event.get("event") == "message_end":
                            self._extract_contexts_from_message_end(event, accumulator)
                    except json.JSONDecodeError:
                        logger.debug(f"无法解析SSE行: {line[:50]}...")

        # 使用asyncio.wait_for包装，确保流式读取不会无限期等待
        try:
            await asyncio.wait_for(_read_stream(), timeout=streaming_timeout)
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"流式响应读取超时（{streaming_timeout}秒），已耗时: {elapsed:.2f}秒")
            # 即使超时，也返回已累积的数据
            if accumulator.answer:
                logger.info(f"返回已累积的部分响应（{len(accumulator.answer)}字符）")
            else:
                raise TimeoutError(f"流式响应读取超时，在{elapsed:.2f}秒内未收到任何数据")

        total_time = time.time() - start_time
        logger.debug(f"流式API调用完成，耗时: {total_time:.3f}秒, 收到tokens: {len(accumulator.token_timestamps)}")
        self._log_streaming_diagnostics(accumulator, path)
        return accumulator.to_adapter_response(start_time)

    def _extract_contexts_from_message_end(self, event: dict[str, Any], accumulator: StreamingAccumulator) -> None:
        """从Chat API的message_end事件中提取retriever_resources"""
        metadata = event.get("metadata", {})
        retriever_resources = metadata.get(CHAT_API_CONTEXT_FIELD, [])
        if isinstance(retriever_resources, list):
            for resource in retriever_resources:
                if isinstance(resource, dict):
                    content = resource.get("content") or resource.get("chunk_content")
                    if content and content not in accumulator.contexts:
                        accumulator.contexts.append(str(content))
                elif resource and resource not in accumulator.contexts:
                    accumulator.contexts.append(str(resource))

    def _log_streaming_event(self, event: dict[str, Any], current_time: float, is_workflow: bool) -> None:
        """记录流式事件详情 - 使用策略模式消除if-elif链"""
        event_type = event.get("event", "unknown")
        logger.debug(f"[流式事件] 类型: {event_type}, 时间: {current_time:.3f}s")
        
        if is_workflow:
            # Workflow API事件
            handlers = {
                "workflow_started": self._log_workflow_started_event,
                "node_started": self._log_node_started_event,
                "text_chunk": self._log_text_chunk_event,
                "node_finished": self._log_node_finished_event,
                "workflow_finished": self._log_workflow_finished_event,
            }
        else:
            # Chat API事件
            handlers = {
                "message": self._log_message_event,
                "agent_message": self._log_agent_message_event,
                "agent_thought": self._log_agent_thought_event,
                "message_end": self._log_message_end_event,
                "message_file": self._log_message_file_event,
                "error": self._log_error_event,
            }
        
        handler = handlers.get(event_type)
        if handler:
            handler(event)
        elif event_type not in ["ping"]:  # ping事件不需要记录
            logger.debug(f"[流式事件] 未处理的事件类型: {event_type}")

    def _log_message_event(self, event: dict[str, Any]) -> None:
        """记录Chat API的message事件"""
        answer_chunk = event.get("answer", "")
        if answer_chunk:
            preview = answer_chunk[:100] + ("..." if len(answer_chunk) > 100 else "")
            logger.debug(f"[流式内容] 答案片段: {preview}")

    def _log_agent_message_event(self, event: dict[str, Any]) -> None:
        """记录Chat API的agent_message事件"""
        answer_chunk = event.get("answer", "")
        if answer_chunk:
            preview = answer_chunk[:100] + ("..." if len(answer_chunk) > 100 else "")
            logger.debug(f"[流式内容] Agent答案片段: {preview}")

    def _log_agent_thought_event(self, event: dict[str, Any]) -> None:
        """记录Chat API的agent_thought事件"""
        thought = event.get("thought", "")
        tool = event.get("tool", "")
        if thought or tool:
            logger.debug(f"[流式思考] 工具: {tool}, 思考: {thought[:100] if thought else ''}")

    def _log_message_end_event(self, event: dict[str, Any]) -> None:
        """记录Chat API的message_end事件"""
        metadata = event.get("metadata", {})
        usage = metadata.get("usage", {})
        retriever_resources = metadata.get(CHAT_API_CONTEXT_FIELD, [])
        logger.debug(f"[流式完成] 总tokens: {usage.get('total_tokens', 0)}, retriever_resources数量: {len(retriever_resources) if isinstance(retriever_resources, list) else 0}")

    def _log_message_file_event(self, event: dict[str, Any]) -> None:
        """记录Chat API的message_file事件"""
        file_type = event.get("type", "unknown")
        file_id = event.get("id", "unknown")
        logger.debug(f"[流式文件] 类型: {file_type}, ID: {file_id}")

    def _log_error_event(self, event: dict[str, Any]) -> None:
        """记录Chat API的error事件"""
        error_code = event.get("code", "unknown")
        error_message = event.get("message", "")
        logger.warning(f"[流式错误] 错误码: {error_code}, 消息: {error_message}")

    def _log_workflow_started_event(self, event: dict[str, Any]) -> None:
        """记录Workflow API的workflow_started事件"""
        data = event.get("data", {})
        workflow_id = data.get("workflow_id", "unknown")
        workflow_run_id = data.get("id", "unknown")
        logger.debug(f"[流式工作流] 开始执行: workflow_id={workflow_id}, run_id={workflow_run_id}")

    def _log_node_started_event(self, event: dict[str, Any]) -> None:
        """记录Workflow API的node_started事件"""
        data = event.get("data", {})
        node_id = data.get("node_id", "unknown")
        node_type = data.get("node_type", "unknown")
        title = data.get("title", "unknown")
        index = data.get("index", 0)
        logger.debug(f"[流式节点] 开始执行: 节点{index} ({node_id}, 类型: {node_type}, 名称: {title})")

    def _log_text_chunk_event(self, event: dict[str, Any]) -> None:
        """记录text_chunk事件"""
        data = event.get("data", {})
        text_chunk = data.get("text", "")
        if text_chunk:
            preview = text_chunk[:100] + ("..." if len(text_chunk) > 100 else "")
            logger.debug(f"[流式内容] 文本片段: {preview}")

    def _log_node_finished_event(self, event: dict[str, Any]) -> None:
        """记录node_finished事件"""
        data = event.get("data", {})
        node_id = data.get("node_id", "unknown")
        node_type = data.get("node_type", "unknown")
        outputs = data.get("outputs", {})
        logger.debug(f"[流式节点] 节点ID: {node_id}, 节点类型: {node_type}, 输出字段: {list(outputs.keys())}")
        
        if outputs:
            preview = str(outputs)[:200] + ("..." if len(str(outputs)) > 200 else "")
            logger.debug(f"[流式节点] 输出内容预览: {preview}")
        
        self._log_contexts_from_outputs(outputs, f"节点 {node_id}")

    def _log_workflow_finished_event(self, event: dict[str, Any]) -> None:
        """记录workflow_finished事件"""
        data = event.get("data", {})
        outputs = data.get("outputs", {})
        logger.debug(f"[流式完成] 总tokens: {data.get('total_tokens', 0)}, 耗时: {data.get('elapsed_time', 0):.3f}s")
        logger.debug(f"[流式完成] 最终outputs字段: {list(outputs.keys()) if isinstance(outputs, dict) else 'N/A'}")
        
        if isinstance(outputs, dict):
            self._log_contexts_from_outputs(outputs, "workflow_finished")

    def _log_contexts_from_outputs(self, outputs: dict[str, Any], source: str) -> None:
        """从outputs中查找并记录contexts字段（Workflow API）"""
        if not isinstance(outputs, dict):
            return
        
        for key in WORKFLOW_API_CONTEXT_FIELDS:
            if key not in outputs:
                continue
            
            contexts = outputs[key]
            count = len(contexts) if isinstance(contexts, list) else 1
            logger.info(f"[流式上下文] ✅ 从{source}的 {key} 字段提取到 {count} 个上下文")
            
            if isinstance(contexts, list) and len(contexts) > 0:
                preview = str(contexts[0])[:100] + ("..." if len(str(contexts[0])) > 100 else "")
                logger.debug(f"[流式上下文] 第一个上下文预览: {preview}")
            break

    def _log_streaming_diagnostics(self, accumulator: StreamingAccumulator, path: str) -> None:
        """记录流式响应诊断信息"""
        is_workflow = self._is_workflow_api(path)
        
        if accumulator.contexts:
            field_name = "retrieved_contexts" if is_workflow else CHAT_API_CONTEXT_FIELD
            logger.info(f"✅ 流式模式成功提取到 {len(accumulator.contexts)} 个{field_name}")
            for i, ctx in enumerate(accumulator.contexts[:2], 1):
                preview = ctx[:100] + ("..." if len(ctx) > 100 else ctx)
                logger.debug(f"   上下文{i}预览: {preview}")
            return
        
        if is_workflow:
            logger.warning("⚠️ 流式模式retrieved_contexts为空，可能是节点输出中未包含retrieved_contexts字段")
            if "nodes" not in accumulator.metadata:
                logger.warning("   metadata中未找到nodes字段，无法进一步诊断")
                return
            self._diagnose_missing_contexts(accumulator.metadata)
        else:
            logger.warning(f"⚠️ 流式模式{CHAT_API_CONTEXT_FIELD}为空，可能是应用未配置RAG/知识库检索，或message_end事件中未包含该字段")

    def _diagnose_missing_contexts(self, metadata: dict[str, Any]) -> None:
        """诊断缺失的contexts字段"""
        nodes = metadata.get("nodes", [])
        logger.warning(f"   流式响应包含 {len(nodes)} 个节点，检查节点输出...")
        
        found_contexts = False
        for i, node in enumerate(nodes, 1):
            node_id = node.get("node_id", f"节点{i}")
            node_type = node.get("node_type", "unknown")
            outputs = node.get("outputs", {})
            logger.warning(f"   节点{i} ({node_id}, 类型: {node_type}) 输出字段: {list(outputs.keys()) if isinstance(outputs, dict) else 'N/A'}")
            
            if isinstance(outputs, dict):
                found_contexts = self._check_outputs_for_contexts(outputs, f"节点{i}") or found_contexts
        
        workflow_outputs = metadata.get("outputs", {})
        if isinstance(workflow_outputs, dict):
            logger.warning(f"   workflow_finished的outputs字段: {list(workflow_outputs.keys())}")
            found_contexts = self._check_outputs_for_contexts(workflow_outputs, "workflow_finished") or found_contexts
        
        if not found_contexts:
            logger.warning("   💡 所有节点输出和workflow_finished中均未找到retrieved_contexts相关字段")
            logger.warning("   💡 可能原因：")
            logger.warning("      1. 智能体未配置RAG/知识库检索功能")
            logger.warning("      2. 当前查询未触发知识库检索")
            logger.warning("      3. Dify API响应格式与预期不符（建议检查show_streaming_content日志）")

    def _check_outputs_for_contexts(self, outputs: dict[str, Any], source: str) -> bool:
        """检查outputs中是否有contexts字段（Workflow API），返回是否找到"""
        found = False
        for key in WORKFLOW_API_CONTEXT_FIELDS:
            if key not in outputs:
                continue
            
            found = True
            value = outputs[key]
            logger.warning(f"     ⚠️ {source}中发现字段 {key}，但值为: {type(value).__name__}")
            
            if isinstance(value, list):
                logger.warning(f"       列表长度: {len(value)}")
                if len(value) > 0:
                    logger.warning(f"       第一个元素类型: {type(value[0]).__name__}")
                    preview = str(value[0])[:100] + ("..." if len(str(value[0])) > 100 else "")
                    logger.warning(f"       第一个元素预览: {preview}")
            elif isinstance(value, str):
                logger.warning(f"       字符串长度: {len(value)}")
                preview = value[:100] + ("..." if len(value) > 100 else "")
                logger.warning(f"       字符串预览: {preview}")
            break
        
        # 检查嵌套字典
        for key, value in outputs.items():
            if isinstance(value, dict):
                logger.debug(f"     字段 {key} 是嵌套字典，包含: {list(value.keys())}")
                if "retrieved_contexts" in value:
                    logger.warning(f"     ⚠️ 在嵌套字段 {key}.retrieved_contexts 中发现上下文")
                
        return found