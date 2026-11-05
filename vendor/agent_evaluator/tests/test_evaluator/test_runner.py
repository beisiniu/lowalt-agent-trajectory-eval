"""测试评估流程集成"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_evaluator.adapters.base import AdapterResponse, PerformanceMetrics
from agent_evaluator.core.result import EvalReport, SampleResult
from agent_evaluator.core.sample import TestSample
from agent_evaluator.evaluator.executor import EvaluatorExecutor
from agent_evaluator.runner import EvaluationRunner


class TestEvaluationRunner:
    """测试评估运行器"""

    @pytest.fixture
    def mock_adapter(self):
        """模拟适配器"""
        adapter = MagicMock()
        adapter.__aenter__ = AsyncMock(return_value=adapter)
        adapter.__aexit__ = AsyncMock(return_value=None)
        return adapter

    @pytest.fixture
    def mock_evaluator(self):
        """模拟评估器"""
        evaluator = MagicMock(spec=EvaluatorExecutor)
        return evaluator

    @pytest.fixture
    def runner(self, mock_adapter, mock_evaluator):
        """评估运行器fixture"""
        return EvaluationRunner(mock_adapter, mock_evaluator, stream=False)

    @pytest.mark.asyncio
    async def test_evaluate_sample(self, runner, mock_adapter, mock_evaluator):
        """测试评估单个样本"""
        # 准备测试数据
        test_sample = TestSample(
            user_input="测试问题",
            reference="参考答案",
        )

        # 模拟适配器响应
        mock_response = AdapterResponse(
            answer="智能体回答",
            contexts=["上下文1"],
            performance=PerformanceMetrics(
                total_time=1.5,
                time_to_first_token=0.3,
                total_tokens=100,
            ),
        )
        mock_adapter.invoke = AsyncMock(return_value=mock_response)

        # 模拟评估器响应
        mock_result = SampleResult(
            scores={"faithfulness": 0.9, "relevancy": 0.85},
        )
        mock_evaluator.evaluate = AsyncMock(return_value=mock_result)

        # 执行评估
        result = await runner.evaluate_sample(test_sample)

        # 验证结果
        assert result.is_success
        assert result.performance is not None
        assert result.performance.total_time == 1.5
        assert result.performance.time_to_first_token == 0.3
        assert result.performance.total_tokens == 100

        # 验证调用
        mock_adapter.invoke.assert_called_once_with("测试问题", stream=False)
        mock_evaluator.evaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, runner, mock_adapter, mock_evaluator):
        """测试批量评估"""
        # 准备测试数据
        test_samples = [
            TestSample(user_input="问题1", reference="答案1"),
            TestSample(user_input="问题2", reference="答案2"),
        ]

        # 模拟适配器响应
        mock_response = AdapterResponse(
            answer="回答",
            performance=PerformanceMetrics(total_time=1.0, total_tokens=50),
        )
        mock_adapter.invoke = AsyncMock(return_value=mock_response)

        # 模拟评估器响应
        mock_result = SampleResult(scores={"faithfulness": 0.9})
        mock_evaluator.evaluate = AsyncMock(return_value=mock_result)

        # 执行批量评估
        report = await runner.evaluate_batch(test_samples)

        # 验证结果
        assert isinstance(report, EvalReport)
        assert report.total_samples == 2
        assert report.start_time is not None
        assert report.end_time is not None

        # 验证每个结果都包含性能指标
        for result in report.results:
            assert result.performance is not None

    @pytest.mark.asyncio
    async def test_evaluate_sample_with_error(self, runner, mock_adapter):
        """测试评估样本时发生错误"""
        test_sample = TestSample(user_input="测试问题")

        # 模拟适配器抛出异常
        mock_adapter.invoke = AsyncMock(side_effect=Exception("API错误"))

        # 执行评估
        result = await runner.evaluate_sample(test_sample)

        # 验证错误被正确处理
        assert not result.is_success
        assert result.error == "API错误"

    @pytest.mark.asyncio
    async def test_streaming_mode(self, mock_adapter, mock_evaluator):
        """测试流式模式"""
        runner = EvaluationRunner(mock_adapter, mock_evaluator, stream=True)

        test_sample = TestSample(user_input="测试问题")
        mock_response = AdapterResponse(answer="回答")
        mock_adapter.invoke = AsyncMock(return_value=mock_response)
        mock_evaluator.evaluate = AsyncMock(return_value=SampleResult(scores={}))

        await runner.evaluate_sample(test_sample)

        # 验证使用了流式模式
        mock_adapter.invoke.assert_called_once_with("测试问题", stream=True)

