"""测试样本数据结构"""

from dataclasses import dataclass
from typing import Any

from ragas import SingleTurnSample

from agent_evaluator.adapters.base import AdapterResponse


@dataclass
class TestSample:
    """测试样本（用户输入）"""

    user_input: str  # 用户输入
    reference: str | None = None  # 参考答案（可选）
    reference_contexts: list[str] | None = None  # 参考上下文（可选）
    metadata: dict[str, Any] | None = None  # 其他元数据


@dataclass
class EvalSample:
    """评估样本（执行后的数据）"""

    user_input: str
    response: str  # 智能体的响应
    contexts: list[str]  # 检索到的上下文
    reference: str | None = None
    reference_contexts: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_response(cls, test_sample: TestSample, response: AdapterResponse) -> "EvalSample":
        """
        从TestSample和AdapterResponse创建EvalSample

        Args:
            test_sample: 测试样本
            response: 适配器响应

        Returns:
            EvalSample对象
        """
        return cls(
            user_input=test_sample.user_input,
            response=response.answer,
            contexts=response.contexts,
            reference=test_sample.reference,
            reference_contexts=test_sample.reference_contexts,
            metadata={
                **(test_sample.metadata or {}),
                **(response.metadata or {}),
            },
        )

    def to_ragas_single_turn(self) -> SingleTurnSample:
        """
        转换为Ragas SingleTurnSample格式

        Returns:
            Ragas SingleTurnSample对象
            
        Note:
            Ragas的某些指标（如Faithfulness、ResponseRelevancy）要求retrieved_contexts非空。
            如果contexts为空，会使用reference_contexts作为fallback。
            如果都为空，会使用包含空字符串的列表作为占位符。
        """
        # 处理retrieved_contexts：优先使用contexts，如果为空则使用reference_contexts作为fallback
        # 过滤掉空字符串，只保留有效内容
        valid_contexts = [ctx for ctx in self.contexts if ctx and ctx.strip()]
        valid_reference_contexts = [ctx for ctx in (self.reference_contexts or []) if ctx and ctx.strip()]
        
        if valid_contexts:
            retrieved_contexts = valid_contexts
        elif valid_reference_contexts:
            retrieved_contexts = valid_reference_contexts
        else:
            # 如果都为空，提供一个占位符，避免ragas验证失败
            # 注意：这可能导致某些指标评分不准确，但至少不会崩溃
            # 使用一个通用的占位符文本，而不是空字符串
            retrieved_contexts = ["无可用上下文"]
        
        kwargs: dict[str, Any] = {
            "user_input": self.user_input,
            "response": self.response,
            "retrieved_contexts": retrieved_contexts,
        }

        if self.reference:
            kwargs["reference"] = self.reference

        if self.reference_contexts:
            kwargs["reference_contexts"] = self.reference_contexts

        return SingleTurnSample(**kwargs)
