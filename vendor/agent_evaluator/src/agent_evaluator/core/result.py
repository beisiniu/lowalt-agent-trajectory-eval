"""评估结果数据结构"""

from dataclasses import dataclass, field
from typing import Any

from agent_evaluator.adapters.base import PerformanceMetrics


@dataclass
class SampleResult:
    """单个样本的评估结果"""

    scores: dict[str, float] = field(default_factory=dict)  # 各指标的分数
    reasoning: dict[str, str] = field(default_factory=dict)  # 各指标的评分理由（可选）
    performance: PerformanceMetrics | None = None  # 性能指标
    error: str | None = None  # 错误信息（如果有）
    
    # 样本详细信息
    user_input: str | None = None  # 原问题
    response: str | None = None  # 智能体答案
    reference: str | None = None  # 标准答案（如果有）
    contexts: list[str] = field(default_factory=list)  # 召回的内容
    metadata: dict[str, Any] = field(default_factory=dict)  # 元数据（可能包含文档信息）

    @property
    def is_success(self) -> bool:
        """判断是否成功"""
        return self.error is None and len(self.scores) > 0

    @property
    def average_score(self) -> float:
        """计算平均分"""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


@dataclass
class EvalReport:
    """完整的评估报告"""

    results: list[SampleResult] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None

    @property
    def total_samples(self) -> int:
        """总样本数"""
        return len(self.results)

    @property
    def failed_samples(self) -> int:
        """失败样本数"""
        return sum(1 for r in self.results if not r.is_success)

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_samples == 0:
            return 0.0
        return (self.total_samples - self.failed_samples) / self.total_samples

    @property
    def overall_score(self) -> float:
        """总体平均分"""
        successful_results = [r for r in self.results if r.is_success]
        if not successful_results:
            return 0.0
        return sum(r.average_score for r in successful_results) / len(successful_results)

    @property
    def duration(self) -> float:
        """总耗时（秒）"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def average_performance(self) -> PerformanceMetrics | None:
        """平均性能指标"""
        performance_list = [r.performance for r in self.results if r.performance]
        if not performance_list:
            return None

        avg_total_time = sum(p.total_time for p in performance_list) / len(performance_list)
        avg_ttft = None
        ttft_list = [p.time_to_first_token for p in performance_list if p.time_to_first_token is not None]
        if ttft_list:
            avg_ttft = sum(ttft_list) / len(ttft_list)

        total_tokens = sum(p.total_tokens for p in performance_list)
        input_tokens = sum(p.input_tokens for p in performance_list)
        output_tokens = sum(p.output_tokens for p in performance_list)

        # 合并所有流式延迟
        all_latencies: list[float] = []
        for p in performance_list:
            all_latencies.extend(p.streaming_latency)

        return PerformanceMetrics(
            total_time=avg_total_time,
            time_to_first_token=avg_ttft,
            streaming_latency=all_latencies,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def add_result(self, result: SampleResult) -> None:
        """添加评估结果"""
        self.results.append(result)

    def finalize(self) -> None:
        """完成评估，记录结束时间"""
        import time

        if self.end_time is None:
            self.end_time = time.time()
