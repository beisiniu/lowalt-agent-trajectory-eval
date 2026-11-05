"""CSV格式输出"""

import csv
from pathlib import Path

from agent_evaluator.core.result import EvalReport
from agent_evaluator.reporters.base import BaseReporter


class CSVReporter(BaseReporter):
    """CSV报告器"""

    def generate(self, report: EvalReport) -> str:
        """生成CSV报告"""
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # 写入摘要行
        writer.writerow(["指标", "值"])
        writer.writerow(["总样本数", report.total_samples])
        writer.writerow(["成功样本", report.total_samples - report.failed_samples])
        writer.writerow(["失败样本", report.failed_samples])
        writer.writerow(["成功率", f"{report.success_rate:.4f}"])
        writer.writerow(["总体平均分", f"{report.overall_score:.4f}"])
        writer.writerow(["总耗时(秒)", f"{report.duration:.4f}"])

        # 写入性能指标
        if report.average_performance:
            perf = report.average_performance
            writer.writerow([])  # 空行
            writer.writerow(["性能指标", ""])
            writer.writerow(["平均总耗时(秒)", f"{perf.total_time:.4f}"])
            if perf.time_to_first_token is not None:
                writer.writerow(["平均首Token时间(秒)", f"{perf.time_to_first_token:.4f}"])
            writer.writerow(["总Token数", perf.total_tokens])
            writer.writerow(["输入Token数", perf.input_tokens])
            writer.writerow(["输出Token数", perf.output_tokens])
            if perf.streaming_latency:
                avg_latency = sum(perf.streaming_latency) / len(perf.streaming_latency)
                writer.writerow(["平均流式延迟(秒)", f"{avg_latency:.4f}"])

        # 写入详细结果
        writer.writerow([])  # 空行
        writer.writerow(["详细结果", ""])

        # 获取所有指标名称
        all_metrics = set()
        for result in report.results:
            if result.scores:
                all_metrics.update(result.scores.keys())

        # 写入表头
        header = ["样本", "状态", "平均分"]
        header.extend(sorted(all_metrics))
        header.extend(["总耗时(秒)", "TTFT(秒)", "Token数"])
        writer.writerow(header)

        # 写入数据行
        for i, result in enumerate(report.results, 1):
            row = [
                f"样本{i}",
                "成功" if result.is_success else "失败",
                f"{result.average_score:.4f}",
            ]

            # 添加各指标分数
            for metric in sorted(all_metrics):
                score = result.scores.get(metric, 0.0)
                row.append(f"{score:.4f}")

            # 添加性能指标
            if result.performance:
                ttft = result.performance.time_to_first_token
                row.extend([
                    f"{result.performance.total_time:.4f}",
                    f"{ttft:.4f}" if ttft is not None else "N/A",
                    str(result.performance.total_tokens),
                ])
            else:
                row.extend(["N/A", "N/A", "N/A"])

            writer.writerow(row)
        
        # 添加评分理由部分
        if any(result.reasoning for result in report.results):
            writer.writerow([])  # 空行
            writer.writerow(["评分理由", ""])
            for i, result in enumerate(report.results, 1):
                if result.reasoning:
                    writer.writerow([f"样本{i}", ""])
                    for metric_name, reason in result.reasoning.items():
                        writer.writerow([f"  {metric_name}", reason])
                    writer.writerow([])  # 空行
        
        # 添加详细样本信息部分
        writer.writerow([])  # 空行
        writer.writerow(["详细样本信息", ""])
        for i, result in enumerate(report.results, 1):
            writer.writerow([f"样本{i}", ""])
            writer.writerow(["原问题", result.user_input or "(无)"])
            writer.writerow(["标准答案", result.reference or "(无)"])
            writer.writerow(["智能体答案", result.response or "(无)"])
            if result.metadata.get("response_full_length"):
                writer.writerow(["完整答案长度", f"{result.metadata.get('response_full_length')}字"])
            writer.writerow(["召回内容", ""])
            if result.contexts:
                for ctx in result.contexts:
                    writer.writerow(["", ctx])
            else:
                writer.writerow(["", "(无)"])
            
            # 召回的文档
            if result.metadata:
                doc_info = []
                if "documents" in result.metadata:
                    doc_info = result.metadata["documents"]
                elif "retrieved_documents" in result.metadata:
                    doc_info = result.metadata["retrieved_documents"]
                
                if doc_info:
                    writer.writerow(["召回文档", ""])
                    for doc in doc_info:
                        if isinstance(doc, dict):
                            doc_name = doc.get("name") or doc.get("title") or doc.get("id", "未知")
                            writer.writerow(["", doc_name])
                        else:
                            writer.writerow(["", str(doc)])
            
            # 评分理由
            if result.reasoning:
                writer.writerow(["评分理由", ""])
                for metric_name, reason in result.reasoning.items():
                    writer.writerow([f"  {metric_name}", reason])
            
            writer.writerow([])  # 空行

        return output.getvalue()

    def save(self, report: EvalReport, path: str) -> None:
        """保存CSV报告到文件"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.generate(report)
        output_path.write_text(content, encoding="utf-8")
