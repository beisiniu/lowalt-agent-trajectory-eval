"""控制台输出"""

from rich.console import Console
from rich.table import Table

from agent_evaluator.core.result import EvalReport
from agent_evaluator.reporters.base import BaseReporter


class ConsoleReporter(BaseReporter):
    """控制台报告器"""

    def __init__(self):
        self.console = Console()

    def generate(self, report: EvalReport) -> str:
        """生成控制台报告"""
        output = []

        # 总体统计
        self.console.print("\n[bold blue]评估报告[/bold blue]")
        self.console.print(f"总样本数: {report.total_samples}")
        self.console.print(f"成功样本: {report.total_samples - report.failed_samples}")
        self.console.print(f"失败样本: {report.failed_samples}")
        self.console.print(f"成功率: {report.success_rate:.2%}")
        self.console.print(f"总体平均分: {report.overall_score:.4f}")
        self.console.print(f"总耗时: {report.duration:.2f}秒")

        # 指标说明（动态显示实际使用的指标）
        self.console.print("\n[bold yellow]指标说明[/bold yellow]")
        
        # 获取所有实际使用的指标
        all_metrics = set()
        for result in report.results:
            if result.scores:
                all_metrics.update(result.scores.keys())
        
        # 指标描述字典
        metric_descriptions = {
            "Faithfulness": "忠实度（0.0-1.0），衡量响应是否忠实于上下文，满分1.0",
            "ResponseRelevancy": "相关性（0.0-1.0），衡量响应与问题的相关性，满分1.0",
            "ContextPrecision": "上下文精确度（0.0-1.0），衡量检索到的上下文中与问题相关的比例，满分1.0",
            "ContextRecall": "上下文召回率（0.0-1.0），衡量检索到的上下文覆盖标准答案的程度，满分1.0",
            "ContextEntityRecall": "上下文实体召回率（0.0-1.0），衡量检索到的上下文中包含标准答案中实体的比例，满分1.0",
            "AnswerCorrectness": "答案正确性（0.0-1.0），衡量答案的正确程度，满分1.0",
            "AnswerAccuracy": "答案准确性（0.0-1.0），衡量答案的准确程度，满分1.0",
            "ContextRelevance": "上下文相关性（0.0-1.0），衡量检索到的上下文与问题的相关性，满分1.0",
            "ResponseGroundedness": "响应基础性（0.0-1.0），衡量响应基于上下文的程度，满分1.0",
            "SemanticSimilarity": "语义相似度（0.0-1.0），衡量响应与标准答案的语义相似度，满分1.0",
            "BleuScore": "BLEU分数（0.0-1.0），基于n-gram匹配的文本相似度，满分1.0",
            "RougeScore": "ROUGE分数（0.0-1.0），基于召回率的文本相似度，满分1.0",
            "ChrfScore": "CHRF分数（0.0-1.0），基于字符n-gram的文本相似度，满分1.0",
            "ExactMatch": "精确匹配（0.0-1.0），响应与标准答案是否完全匹配，满分1.0",
            "StringPresence": "字符串存在性（0.0-1.0），响应中是否包含特定字符串，满分1.0",
            "NoiseSensitivity": "噪声敏感性（0.0-1.0），衡量响应对上下文噪声的敏感性，满分1.0",
        }
        
        # 显示实际使用的指标
        for metric in sorted(all_metrics):
            description = metric_descriptions.get(metric, f"{metric}（0.0-1.0），满分1.0")
            self.console.print(f"  • [cyan]{metric}[/cyan]: {description}")
        
        # 性能指标说明
        self.console.print("  • [cyan]TTFT[/cyan]: Time To First Token，首Token时间，从发送请求到收到第一个Token的耗时")

        # 性能指标
        if report.average_performance:
            perf = report.average_performance
            self.console.print("\n[bold green]性能指标[/bold green]")
            self.console.print(f"平均总耗时: {perf.total_time:.3f}秒")
            if perf.time_to_first_token is not None:
                self.console.print(f"平均首Token时间(TTFT): {perf.time_to_first_token:.3f}秒")
            self.console.print(f"总Token数: {perf.total_tokens}")
            self.console.print(f"输入Token数: {perf.input_tokens}")
            self.console.print(f"输出Token数: {perf.output_tokens}")
            if perf.streaming_latency:
                avg_latency = sum(perf.streaming_latency) / len(perf.streaming_latency)
                self.console.print(f"平均流式延迟: {avg_latency:.3f}秒")

        # 指标分数表
        if report.results:
            table = Table(title="指标分数")
            table.add_column("样本", style="cyan")
            table.add_column("平均分", style="magenta")
            table.add_column("状态", style="green")

            # 获取所有指标名称
            all_metrics = set()
            for result in report.results:
                if result.scores:
                    all_metrics.update(result.scores.keys())

            # 添加指标列
            for metric in sorted(all_metrics):
                table.add_column(metric, style="yellow")

            # 添加性能列
            table.add_column("总耗时", style="blue")
            table.add_column("TTFT", style="blue")

            # 添加数据行
            for i, result in enumerate(report.results, 1):
                row = [
                    f"样本{i}",
                    f"{result.average_score:.4f}",
                    "✓" if result.is_success else "✗",
                ]

                # 添加各指标分数
                for metric in sorted(all_metrics):
                    score = result.scores.get(metric, 0.0)
                    row.append(f"{score:.4f}")

                # 添加性能指标
                if result.performance:
                    row.append(f"{result.performance.total_time:.3f}s")
                    ttft = result.performance.time_to_first_token
                    row.append(f"{ttft:.3f}s" if ttft is not None else "N/A")
                else:
                    row.extend(["N/A", "N/A"])

                table.add_row(*row)

            self.console.print("\n")
            self.console.print(table)
            
            # 显示评分理由（如果有）
            if any(result.reasoning for result in report.results):
                self.console.print("\n[bold yellow]评分理由[/bold yellow]")
                for i, result in enumerate(report.results, 1):
                    if result.reasoning:
                        self.console.print(f"\n[cyan]样本{i}[/cyan]:")
                        for metric_name, reason in result.reasoning.items():
                            self.console.print(f"  • {metric_name}: {reason}")
            
            # 显示详细样本信息
            self.console.print("\n[bold yellow]详细样本信息[/bold yellow]")
            for i, result in enumerate(report.results, 1):
                self.console.print(f"\n[bold cyan]样本{i}[/bold cyan]")
                self.console.print("─" * 80)
                
                # 原问题
                if result.user_input:
                    self.console.print(f"[bold]原问题:[/bold] {result.user_input}")
                
                # 标准答案
                if result.reference:
                    self.console.print(f"[bold]标准答案:[/bold] {result.reference}")
                else:
                    self.console.print("[bold]标准答案:[/bold] (无)")
                
                # 智能体答案
                if result.response:
                    self.console.print(f"[bold]智能体答案:[/bold] {result.response}")
                    if result.metadata.get("response_full_length"):
                        full_length = result.metadata.get("response_full_length")
                        self.console.print(f"[dim](显示前200字，完整答案共{full_length}字)[/dim]")
                else:
                    self.console.print("[bold]智能体答案:[/bold] (无)")
                
                # 召回内容
                if result.contexts:
                    self.console.print(f"[bold]召回内容:[/bold] (共{len(result.contexts)}条)")
                    for ctx in result.contexts:
                        self.console.print(f"  • {ctx}")
                else:
                    self.console.print("[bold]召回内容:[/bold] (无)")
                
                # 召回的文档（从metadata中获取）
                if result.metadata:
                    doc_info = []
                    if "documents" in result.metadata:
                        doc_info = result.metadata["documents"]
                    elif "retrieved_documents" in result.metadata:
                        doc_info = result.metadata["retrieved_documents"]
                    
                    if doc_info:
                        self.console.print(f"[bold]召回文档:[/bold] (共{len(doc_info)}个)")
                        for doc in doc_info[:5]:  # 最多显示5个文档
                            if isinstance(doc, dict):
                                doc_name = doc.get("name") or doc.get("title") or doc.get("id", "未知")
                                self.console.print(f"  • {doc_name}")
                            else:
                                self.console.print(f"  • {doc}")
                        if len(doc_info) > 5:
                            self.console.print(f"  ... (还有{len(doc_info) - 5}个文档)")
                
                # 评分理由
                if result.reasoning:
                    self.console.print(f"[bold]评分理由:[/bold]")
                    for metric_name, reason in result.reasoning.items():
                        self.console.print(f"  • {metric_name}: {reason}")
                
                self.console.print("─" * 80)

        return ""

    def save(self, report: EvalReport, path: str) -> None:
        """控制台报告器不需要保存文件"""
        pass
