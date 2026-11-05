"""JSON格式输出"""

import json
from pathlib import Path

from agent_evaluator.core.result import EvalReport
from agent_evaluator.reporters.base import BaseReporter


class JSONReporter(BaseReporter):
    """JSON报告器"""

    def generate(self, report: EvalReport) -> str:
        """生成JSON报告"""
        data = {
            "summary": {
                "total_samples": report.total_samples,
                "failed_samples": report.failed_samples,
                "success_rate": report.success_rate,
                "overall_score": report.overall_score,
                "duration": report.duration,
            },
            "results": [],
        }

        # 添加性能指标摘要
        if report.average_performance:
            perf = report.average_performance
            data["summary"]["performance"] = {
                "average_total_time": perf.total_time,
                "average_time_to_first_token": perf.time_to_first_token,
                "total_tokens": perf.total_tokens,
                "input_tokens": perf.input_tokens,
                "output_tokens": perf.output_tokens,
                "average_streaming_latency": (
                    sum(perf.streaming_latency) / len(perf.streaming_latency)
                    if perf.streaming_latency
                    else None
                ),
            }

        # 添加详细结果
        for i, result in enumerate(report.results, 1):
            result_data = {
                "sample_index": i,
                "is_success": result.is_success,
                "average_score": result.average_score,
                "scores": result.scores,
                "reasoning": result.reasoning if result.reasoning else {},
                "user_input": result.user_input,
                "response": result.response,
                "reference": result.reference,
                "contexts": result.contexts,
                "metadata": result.metadata,
            }

            if result.performance:
                result_data["performance"] = {
                    "total_time": result.performance.total_time,
                    "time_to_first_token": result.performance.time_to_first_token,
                    "total_tokens": result.performance.total_tokens,
                    "input_tokens": result.performance.input_tokens,
                    "output_tokens": result.performance.output_tokens,
                    "streaming_latency": result.performance.streaming_latency,
                }

            if result.error:
                result_data["error"] = result.error

            data["results"].append(result_data)

        return json.dumps(data, indent=2, ensure_ascii=False)

    def save(self, report: EvalReport, path: str) -> None:
        """保存JSON报告到文件"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.generate(report)
        output_path.write_text(content, encoding="utf-8")
