"""测试Dify适配器"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_evaluator.adapters.base import AdapterResponse, PerformanceMetrics
from agent_evaluator.adapters.dify import DifyAdapter


class TestDifyAdapter:
    """测试Dify适配器"""

    @pytest.fixture
    def api_config(self):
        """API配置fixture"""
        return {
            "api_key": "test-api-key",
            "base_url": "https://api.dify.ai/v1",
            "app_id": "test-app-id",
        }

    @pytest.fixture
    def adapter(self, api_config):
        """适配器fixture"""
        return DifyAdapter(api_config)

    def test_get_headers(self, adapter):
        """测试获取请求头"""
        headers = adapter._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_invoke_non_streaming(self, adapter, api_config):
        """测试非流式调用"""
        mock_response_data = {
            "answer": "Test answer",
            "retrieved_contexts": ["context1", "context2"],
            "message_id": "msg-123",
            "conversation_id": "conv-123",
            "metadata": {
                "usage": {
                    "total_tokens": 100,
                    "prompt_tokens": 50,
                    "completion_tokens": 50,
                },
            },
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        adapter._client = mock_client

        response = await adapter._invoke_non_streaming("test input")

        assert isinstance(response, AdapterResponse)
        assert response.answer == "Test answer"
        assert response.contexts == ["context1", "context2"]
        assert response.performance is not None
        assert response.performance.total_tokens == 100

    @pytest.mark.asyncio
    async def test_invoke_streaming(self, adapter, api_config):
        """测试流式调用"""
        # 模拟SSE流式响应
        sse_lines = [
            'data: {"event": "message", "answer": "Hello", "created_at": 1000}',
            'data: {"event": "message", "answer": " World", "created_at": 1001}',
            'data: {"event": "workflow_finished", "data": {"elapsed_time": 0.5, "total_tokens": 50}}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)
        mock_stream.aiter_lines = AsyncMock(return_value=mock_aiter_lines())
        mock_stream.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream)

        adapter._client = mock_client

        response = await adapter._invoke_streaming("test input")

        assert isinstance(response, AdapterResponse)
        assert response.answer == "Hello World"
        assert response.performance is not None
        assert response.performance.total_tokens == 50

    @pytest.mark.asyncio
    async def test_invoke_with_stream_flag(self, adapter):
        """测试invoke方法的stream参数"""
        with patch.object(adapter, "_invoke_streaming") as mock_streaming, patch.object(
            adapter, "_invoke_non_streaming"
        ) as mock_non_streaming:
            # 测试流式模式
            await adapter.invoke("test", stream=True)
            mock_streaming.assert_called_once()
            mock_non_streaming.assert_not_called()

            # 重置mock
            mock_streaming.reset_mock()
            mock_non_streaming.reset_mock()

            # 测试非流式模式
            await adapter.invoke("test", stream=False)
            mock_non_streaming.assert_called_once()
            mock_streaming.assert_not_called()

    @pytest.mark.asyncio
    async def test_invoke_requires_context_manager(self, adapter):
        """测试适配器必须作为上下文管理器使用"""
        adapter._client = None

        with pytest.raises(RuntimeError, match="Adapter must be used as async context manager"):
            await adapter.invoke("test")

    @pytest.mark.asyncio
    async def test_invoke_requires_app_id(self, adapter):
        """测试缺少app_id时抛出错误"""
        adapter.api_config.pop("app_id", None)
        adapter._client = AsyncMock()

        with pytest.raises(ValueError, match="app_id is required"):
            await adapter._invoke_non_streaming("test")

