"""YML配置的数据类定义"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class APIConfig(BaseModel):
    """平台API配置"""

    # 允许各平台追加自己的字段（如 LangGraph 的 dataset_path / variant / use_llm）。
    # 原实现会静默丢弃额外字段，导致非 dify 平台无法传参。
    model_config = ConfigDict(extra="allow")

    api_key: str = Field(..., description="API密钥")
    base_url: str = Field(..., description="API基础URL")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="超时时间（秒）")
    app_id: str | None = Field(default=None, description="应用ID（某些平台需要）")


class EvaluatorLLMConfig(BaseModel):
    """评估器LLM配置"""

    provider: Literal["openai", "anthropic", "azure_openai", "langgenius"] = Field(
        ..., description="LLM提供商"
    )
    model: str = Field(..., description="模型名称")
    api_key: str = Field(..., description="API密钥")
    base_url: str | None = Field(default=None, description="API基础URL（可选）")
    embeddings_model: str | None = Field(
        default=None, description="Embeddings模型名称（可选，如果不指定则使用默认值或自动检测）"
    )
    timeout: float | None = Field(
        default=None, description="LLM和Embeddings API调用的超时时间（秒），默认120秒（Ragas的prompt较长，建议至少120秒）"
    )
    max_retries: int | None = Field(
        default=None, description="最大重试次数（可选，默认3次，用于应对429限流错误）"
    )
    request_delay: float | None = Field(
        default=None, description="请求之间的延迟时间（秒，可选，用于减少429限流错误）"
    )


class LogConfig(BaseModel):
    """日志配置"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="日志级别"
    )
    format: Literal["detailed", "simple", "json"] = Field(
        default="detailed", description="日志格式：detailed(详细), simple(简单), json(JSON格式)"
    )
    show_streaming_content: bool = Field(
        default=False, description="是否显示流式输出的详细内容（DEBUG级别时有效）"
    )


class OutputConfig(BaseModel):
    """输出配置"""

    format: list[Literal["console", "json", "html", "csv"]] = Field(
        default=["console"], description="输出格式列表"
    )
    save_path: str = Field(default="./results/", description="保存路径")


class EvalConfig(BaseModel):
    """评估配置（主配置）"""

    platform: Literal["dify", "bisheng", "n8n", "coze", "langchain", "langgraph"] = Field(
        ..., description="目标平台"
    )
    api_config: APIConfig = Field(..., description="API配置")
    dataset: str = Field(..., description="数据集路径")
    metrics: list[str] = Field(..., min_length=1, description="评估指标列表")
    evaluator_llm: EvaluatorLLMConfig = Field(..., description="评估器LLM配置")
    output: OutputConfig = Field(default_factory=OutputConfig, description="输出配置")
    stream: bool = Field(default=False, description="是否使用流式输出")
    log: LogConfig = Field(default_factory=LogConfig, description="日志配置")
