"""基础Adapter接口和数据结构"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    total_time: float  # 总耗时（秒）
    time_to_first_token: float | None = None  # 首token时间（秒）
    streaming_latency: list[float] = field(default_factory=list)  # 每个chunk的延迟（秒）
    total_tokens: int = 0  # 总token数
    input_tokens: int = 0  # 输入token数
    output_tokens: int = 0  # 输出token数
    total_price: float | None = None  # 总成本（可选）
    currency: str | None = None  # 货币单位（可选）


@dataclass
class AdapterResponse:
    """统一的适配器响应格式"""
    answer: str  # 最终答案
    contexts: list[str] = field(default_factory=list)  # 检索到的上下文
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 工具调用
    metadata: dict[str, Any] = field(default_factory=dict)  # 平台特定的元数据
    performance: PerformanceMetrics | None = None  # 性能指标


class PlatformAdapter(ABC):
    """平台适配器基类"""

    def __init__(self, api_config: dict[str, Any]):
        """
        初始化适配器

        Args:
            api_config: API配置字典，包含api_key、base_url等
        """
        self.api_config = api_config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        # 获取基础超时时间
        base_timeout = self.api_config.get("timeout", 30.0)
        # 流式响应需要更长的超时时间（基础超时的10倍，最少300秒）
        # 因为流式响应可能持续很长时间
        streaming_timeout = max(base_timeout * 10, 300.0)
        
        # 设置超时：连接5秒，读取/写入使用streaming_timeout（流式响应需要更长）
        timeout = httpx.Timeout(
            connect=5.0,
            read=streaming_timeout,
            write=streaming_timeout,
            pool=5.0,
        )
        
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=self._get_headers(),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_headers(self) -> dict[str, str]:
        """
        获取请求头（可被子类覆盖）

        Returns:
            请求头字典
        """
        return {
            "Content-Type": "application/json",
        }

    @abstractmethod
    async def invoke(
        self,
        input: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> AdapterResponse:
        """
        调用平台API

        Args:
            input: 用户输入
            stream: 是否使用流式输出
            **kwargs: 其他平台特定参数

        Returns:
            AdapterResponse对象
        """
        pass
