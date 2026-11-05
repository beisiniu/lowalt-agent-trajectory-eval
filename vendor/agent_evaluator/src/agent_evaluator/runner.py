"""主运行器（编排整个流程）"""

import time
from typing import Any

from agent_evaluator.adapters.base import PlatformAdapter
from agent_evaluator.core.result import EvalReport, SampleResult
from agent_evaluator.core.sample import EvalSample, TestSample
from agent_evaluator.evaluator.executor import EvaluatorExecutor
from agent_evaluator.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationRunner:
    """评估运行器，编排整个评估流程"""

    def __init__(
        self,
        adapter: PlatformAdapter,
        evaluator: EvaluatorExecutor,
        stream: bool = False,
    ):
        """
        初始化运行器

        Args:
            adapter: 平台适配器
            evaluator: 评估执行器
            stream: 是否使用流式输出
        """
        self.adapter = adapter
        self.evaluator = evaluator
        self.stream = stream

    async def evaluate_sample(self, test_sample: TestSample) -> SampleResult:
        """
        评估单个样本

        Args:
            test_sample: 测试样本

        Returns:
            SampleResult对象（包含性能指标）
        """
        sample_start_time = time.time()
        try:
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"开始评估样本: {test_sample.user_input[:80]}{'...' if len(test_sample.user_input) > 80 else ''}")
            
            # 1. 调用适配器获取响应（流式或非流式）
            adapter_start_time = time.time()
            logger.info(f"📡 调用适配器获取响应...")
            response = await self.adapter.invoke(
                test_sample.user_input,
                stream=self.stream,
            )
            adapter_duration = time.time() - adapter_start_time
            logger.info(f"✅ 适配器响应完成 (耗时: {adapter_duration:.2f}秒)")
            logger.info(f"   - 响应长度: {len(response.answer)} 字符")
            logger.info(f"   - 上下文数量: {len(response.contexts)}")

            # 2. 流式完成后，立即调用Ragas验证（统一时机）
            logger.info(f"📊 开始Ragas评估...")
            eval_sample = EvalSample.from_response(test_sample, response)
            result = await self.evaluator.evaluate(eval_sample)

            # 3. 将性能指标传递到结果中
            result.performance = response.performance

            sample_duration = time.time() - sample_start_time
            if result.is_success:
                logger.info(f"✅ 样本评估成功")
                logger.info(f"   - 平均分: {result.average_score:.4f}")
                logger.info(f"   - 得分详情: {result.scores}")
                logger.info(f"   - 总耗时: {sample_duration:.2f}秒")
            else:
                logger.warning(f"❌ 样本评估失败: {result.error}")
                logger.warning(f"   - 总耗时: {sample_duration:.2f}秒")
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            return result

        except Exception as e:
            sample_duration = time.time() - sample_start_time
            logger.error(f"评估样本时发生错误 (耗时: {sample_duration:.2f}秒): {e}")
            import traceback
            logger.debug(f"错误详情:\n{traceback.format_exc()}")
            return SampleResult(error=str(e))

    async def evaluate_batch(
        self,
        test_samples: list[TestSample],
    ) -> EvalReport:
        """
        批量评估样本

        Args:
            test_samples: 测试样本列表

        Returns:
            EvalReport对象
        """
        report = EvalReport()
        report.start_time = time.time()
        total_samples = len(test_samples)

        logger.info(f"╔══════════════════════════════════════════════════════════════╗")
        logger.info(f"║  开始批量评估                                                  ║")
        logger.info(f"║  总样本数: {total_samples:<45} ║")
        logger.info(f"╚══════════════════════════════════════════════════════════════╝")

        # 使用适配器作为异步上下文管理器
        async with self.adapter:
            for idx, test_sample in enumerate(test_samples, 1):
                logger.info(f"\n📝 样本进度: [{idx}/{total_samples}]")
                result = await self.evaluate_sample(test_sample)
                report.add_result(result)
                
                # 显示当前统计
                success_count = report.total_samples - report.failed_samples
                logger.info(f"📈 当前统计: 成功 {success_count}/{idx}, 失败 {report.failed_samples}/{idx}")

        report.finalize()
        logger.info(f"\n╔══════════════════════════════════════════════════════════════╗")
        logger.info(f"║  批量评估完成                                                  ║")
        logger.info(f"║  总样本数: {report.total_samples:<45} ║")
        logger.info(f"║  成功: {report.total_samples - report.failed_samples:<48} ║")
        logger.info(f"║  失败: {report.failed_samples:<48} ║")
        logger.info(f"║  总耗时: {report.duration:.2f}秒{'':<42} ║")
        logger.info(f"╚══════════════════════════════════════════════════════════════╝")
        return report
