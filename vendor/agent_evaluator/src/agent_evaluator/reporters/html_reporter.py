"""HTML报告生成"""

import html as html_module
from pathlib import Path

from agent_evaluator.core.result import EvalReport
from agent_evaluator.reporters.base import BaseReporter


class HTMLReporter(BaseReporter):
    """HTML报告器"""

    def generate(self, report: EvalReport) -> str:
        """生成HTML报告"""
        html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>评估报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .summary-card {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .summary-card h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; }}
        .summary-card .value {{ font-size: 24px; font-weight: bold; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .success {{ color: #4CAF50; font-weight: bold; }}
        .error {{ color: #f44336; font-weight: bold; }}
        .performance-section {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .sample-detail {{ background: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #2196F3; }}
        .sample-detail h3 {{ margin-top: 0; color: #2196F3; }}
        .sample-detail .field {{ margin: 10px 0; }}
        .sample-detail .field-label {{ font-weight: bold; color: #555; }}
        .sample-detail .field-value {{ margin-top: 5px; padding: 8px; background: white; border-radius: 3px; }}
        .context-item {{ margin: 5px 0; padding: 5px; background: #fff; border-left: 3px solid #4CAF50; }}
        .reason-item {{ margin: 5px 0; padding: 5px; background: #fff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>评估报告</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>总样本数</h3>
                <div class="value">{total_samples}</div>
            </div>
            <div class="summary-card">
                <h3>成功样本</h3>
                <div class="value">{success_samples}</div>
            </div>
            <div class="summary-card">
                <h3>失败样本</h3>
                <div class="value">{failed_samples}</div>
            </div>
            <div class="summary-card">
                <h3>成功率</h3>
                <div class="value">{success_rate:.2%}</div>
            </div>
            <div class="summary-card">
                <h3>总体平均分</h3>
                <div class="value">{overall_score:.4f}</div>
            </div>
            <div class="summary-card">
                <h3>总耗时</h3>
                <div class="value">{duration:.2f}秒</div>
            </div>
        </div>
"""

        # 添加性能指标部分
        if report.average_performance:
            perf = report.average_performance
            avg_latency = (
                sum(perf.streaming_latency) / len(perf.streaming_latency)
                if perf.streaming_latency
                else None
            )
            html += f"""
        <div class="performance-section">
            <h2>性能指标</h2>
            <p><strong>平均总耗时:</strong> {perf.total_time:.3f}秒</p>
            <p><strong>平均首Token时间(TTFT):</strong> {perf.time_to_first_token:.3f}秒</p>
            <p><strong>总Token数:</strong> {perf.total_tokens}</p>
            <p><strong>输入Token数:</strong> {perf.input_tokens}</p>
            <p><strong>输出Token数:</strong> {perf.output_tokens}</p>
            {f'<p><strong>平均流式延迟:</strong> {avg_latency:.3f}秒</p>' if avg_latency else ''}
        </div>
"""

        # 添加详细结果表
        html += """
        <h2>详细结果</h2>
        <table>
            <thead>
                <tr>
                    <th>样本</th>
                    <th>状态</th>
                    <th>平均分</th>
"""

        # 获取所有指标名称
        all_metrics = set()
        for result in report.results:
            if result.scores:
                all_metrics.update(result.scores.keys())

        for metric in sorted(all_metrics):
            html += f'                    <th>{metric}</th>\n'

        html += """
                    <th>总耗时</th>
                    <th>TTFT</th>
                    <th>Token数</th>
                </tr>
            </thead>
            <tbody>
"""

        for i, result in enumerate(report.results, 1):
            status_class = "success" if result.is_success else "error"
            status_text = "✓" if result.is_success else "✗"
            html += f"""
                <tr>
                    <td>样本{i}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result.average_score:.4f}</td>
"""

            for metric in sorted(all_metrics):
                score = result.scores.get(metric, 0.0)
                html += f"                    <td>{score:.4f}</td>\n"

            if result.performance:
                ttft = result.performance.time_to_first_token
                html += f"""
                    <td>{result.performance.total_time:.3f}s</td>
                    <td>{ttft:.3f}s</td>
                    <td>{result.performance.total_tokens}</td>
"""
            else:
                html += """
                    <td>N/A</td>
                    <td>N/A</td>
                    <td>N/A</td>
"""

            html += "                </tr>\n"

        html += """
            </tbody>
        </table>
"""
        
        # 添加评分理由部分
        if any(result.reasoning for result in report.results):
            html += """
        <h2>评分理由</h2>
        <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
"""
            for i, result in enumerate(report.results, 1):
                if result.reasoning:
                    html += f"""
            <h3>样本{i}</h3>
            <ul>
"""
                    for metric_name, reason in result.reasoning.items():
                        html += f'                <li><strong>{metric_name}:</strong> {reason}</li>\n'
                    html += "            </ul>\n"
            
            html += "        </div>\n"
        
        # 添加详细样本信息部分
        html += """
        <h2>详细样本信息</h2>
"""
        for i, result in enumerate(report.results, 1):
            html += f"""
        <div class="sample-detail">
            <h3>样本{i}</h3>
"""
            # 原问题
            if result.user_input:
                user_input_escaped = html_module.escape(result.user_input)
                html += f"""
            <div class="field">
                <div class="field-label">原问题:</div>
                <div class="field-value">{user_input_escaped}</div>
            </div>
"""
            
            # 标准答案
            reference_escaped = html_module.escape(result.reference) if result.reference else "(无)"
            html += f"""
            <div class="field">
                <div class="field-label">标准答案:</div>
                <div class="field-value">{reference_escaped}</div>
            </div>
"""
            
            # 智能体答案
            if result.response:
                response_escaped = html_module.escape(result.response)
                html += f"""
            <div class="field">
                <div class="field-label">智能体答案:</div>
                <div class="field-value">{response_escaped}"""
                if result.metadata.get("response_full_length"):
                    full_length = result.metadata.get("response_full_length")
                    html += f'<br><small style="color: #666;">(显示前200字，完整答案共{full_length}字)</small>'
                html += """
                </div>
            </div>
"""
            else:
                html += """
            <div class="field">
                <div class="field-label">智能体答案:</div>
                <div class="field-value">(无)</div>
            </div>
"""
            
            # 召回内容
            if result.contexts:
                html += f"""
            <div class="field">
                <div class="field-label">召回内容 (共{len(result.contexts)}条):</div>
                <div class="field-value">
"""
                for ctx in result.contexts:
                    ctx_escaped = html_module.escape(ctx)
                    html += f'                    <div class="context-item">{ctx_escaped}</div>\n'
                html += """
                </div>
            </div>
"""
            else:
                html += """
            <div class="field">
                <div class="field-label">召回内容:</div>
                <div class="field-value">(无)</div>
            </div>
"""
            
            # 召回的文档
            if result.metadata:
                doc_info = []
                if "documents" in result.metadata:
                    doc_info = result.metadata["documents"]
                elif "retrieved_documents" in result.metadata:
                    doc_info = result.metadata["retrieved_documents"]
                
                if doc_info:
                    html += f"""
            <div class="field">
                <div class="field-label">召回文档 (共{len(doc_info)}个):</div>
                <div class="field-value">
                    <ul>
"""
                    for doc in doc_info[:5]:  # 最多显示5个文档
                        if isinstance(doc, dict):
                            doc_name = doc.get("name") or doc.get("title") or doc.get("id", "未知")
                            doc_name_escaped = html_module.escape(str(doc_name))
                            html += f'                        <li>{doc_name_escaped}</li>\n'
                        else:
                            doc_escaped = html_module.escape(str(doc))
                            html += f'                        <li>{doc_escaped}</li>\n'
                    if len(doc_info) > 5:
                        html += f'                        <li>... (还有{len(doc_info) - 5}个文档)</li>\n'
                    html += """
                    </ul>
                </div>
            </div>
"""
            
            # 评分理由
            if result.reasoning:
                html += """
            <div class="field">
                <div class="field-label">评分理由:</div>
                <div class="field-value">
"""
                for metric_name, reason in result.reasoning.items():
                    reason_escaped = html_module.escape(reason)
                    metric_name_escaped = html_module.escape(metric_name)
                    html += f'                    <div class="reason-item"><strong>{metric_name_escaped}:</strong> {reason_escaped}</div>\n'
                html += """
                </div>
            </div>
"""
            
            html += """
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""

        return html.format(
            total_samples=report.total_samples,
            success_samples=report.total_samples - report.failed_samples,
            failed_samples=report.failed_samples,
            success_rate=report.success_rate,
            overall_score=report.overall_score,
            duration=report.duration,
        )

    def save(self, report: EvalReport, path: str) -> None:
        """保存HTML报告到文件"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.generate(report)
        output_path.write_text(content, encoding="utf-8")
