"""评估执行器（调用ragas）"""

import asyncio
import time
from typing import Any

from agent_evaluator.core.result import SampleResult
from agent_evaluator.core.sample import EvalSample
from agent_evaluator.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluatorExecutor:
    """评估执行器，负责调用Ragas进行指标评估"""

    def __init__(self, metrics: list[Any], llm: Any, embeddings: Any | None = None, timeout: float = 120.0):
        """
        初始化评估执行器

        Args:
            metrics: Ragas指标列表
            llm: 评估用的LLM
            embeddings: 评估用的embeddings（可选）
            timeout: 单个指标评估的超时时间（秒），默认120秒
        """
        self.metrics = metrics
        self.llm = llm
        self.embeddings = embeddings
        self.timeout = timeout

    async def evaluate(self, eval_sample: EvalSample) -> SampleResult:
        """
        评估单个样本

        Args:
            eval_sample: 评估样本

        Returns:
            SampleResult对象
        """
        eval_start_time = time.time()
        try:
            # 数据验证：确保response不为空
            if not eval_sample.response or not eval_sample.response.strip():
                logger.warning("响应为空，跳过评估")
                return SampleResult(error="响应为空，无法进行评估")
            
            # 转换为Ragas格式
            logger.debug("正在转换评估样本为Ragas格式...")
            ragas_sample = eval_sample.to_ragas_single_turn()
            
            # 显示评估样本摘要
            logger.info(f"评估样本摘要:")
            logger.info(f"  - 用户输入: {eval_sample.user_input[:100]}{'...' if len(eval_sample.user_input) > 100 else ''}")
            logger.info(f"  - 响应长度: {len(eval_sample.response)} 字符")
            logger.info(f"  - 上下文数量: {len(ragas_sample.retrieved_contexts)}")
            if ragas_sample.retrieved_contexts:
                # 检查上下文是否为空字符串
                non_empty_contexts = [ctx for ctx in ragas_sample.retrieved_contexts if ctx and ctx.strip()]
                if len(non_empty_contexts) != len(ragas_sample.retrieved_contexts):
                    logger.warning(f"  ⚠️ 发现空上下文: 总数={len(ragas_sample.retrieved_contexts)}, 非空={len(non_empty_contexts)}")
                if non_empty_contexts:
                    contexts_preview = non_empty_contexts[0][:50] if non_empty_contexts[0] else ""
                    logger.info(f"  - 上下文预览: {contexts_preview}{'...' if len(contexts_preview) >= 50 else ''}")
                else:
                    logger.warning(f"  ⚠️ 所有上下文都为空字符串，这可能导致评估失败")
            
            # 验证ragas_sample的关键字段
            if not ragas_sample.retrieved_contexts or (len(ragas_sample.retrieved_contexts) == 1 and not ragas_sample.retrieved_contexts[0]):
                logger.warning("retrieved_contexts为空，某些指标可能无法正确评估")

            # 调用Ragas进行评估
            scores: dict[str, float] = {}
            reasoning: dict[str, str] = {}  # 存储评分理由
            errors: dict[str, str] = {}
            total_metrics = len(self.metrics)

            logger.info(f"开始评估指标，共 {total_metrics} 个指标")
            for idx, metric in enumerate(self.metrics, 1):
                metric_name = metric.__class__.__name__
                metric_start_time = time.time()
                
                # 在评估指标之间添加延迟，以减少429限流错误
                # 注意：这个延迟需要在创建EvaluatorExecutor时传入，暂时先不实现
                # 如果遇到429错误，建议在配置文件中设置request_delay
                
                try:
                    logger.info(f"[{idx}/{total_metrics}] 🔄 正在评估指标: {metric_name}...")
                    
                    # 创建一个后台任务来定期输出进度（每10秒输出一次）
                    async def progress_monitor():
                        while True:
                            await asyncio.sleep(10)  # 每10秒输出一次
                            elapsed = time.time() - metric_start_time
                            remaining = max(0, self.timeout - elapsed)
                            if remaining > 0:
                                logger.info(f"[{idx}/{total_metrics}] ⏳ {metric_name} 评估中... (已用: {elapsed:.1f}秒, 剩余: {remaining:.1f}秒)")
                    
                    progress_task = asyncio.create_task(progress_monitor())
                    
                    try:
                        # 检测是否为多轮对话指标
                        multi_turn_metrics = [
                            "TopicAdherenceScore",
                            "ToolCallAccuracy",
                            "AgentGoalAccuracyWithReference",
                            "AgentGoalAccuracyWithoutReference",
                        ]
                        
                        if metric_name in multi_turn_metrics:
                            # 多轮对话指标需要MultiTurnSample，当前数据格式不支持
                            error_msg = f"指标 {metric_name} 是多轮对话指标，需要MultiTurnSample数据格式，当前数据为单轮对话格式，无法评估"
                            logger.warning(f"[{idx}/{total_metrics}] ⚠️ {error_msg}")
                            errors[metric_name] = error_msg
                            continue
                        
                        logger.debug(f"[{idx}/{total_metrics}] 开始调用Ragas的single_turn_ascore方法...")
                        logger.debug(f"[{idx}/{total_metrics}] 超时设置: {self.timeout}秒")
                        
                        # 调用Ragas的single_turn_ascore方法，添加超时保护
                        # ragas的日志已经通过loguru拦截器统一输出，无需额外配置
                        score = await asyncio.wait_for(
                            metric.single_turn_ascore(ragas_sample),
                            timeout=self.timeout
                        )
                        progress_task.cancel()  # 完成后取消进度监控
                        try:
                            await progress_task
                        except asyncio.CancelledError:
                            pass
                        
                        metric_duration = time.time() - metric_start_time
                        score_value = float(score)
                        scores[metric_name] = score_value
                        
                        # 生成评分理由（基于指标类型和分数）
                        if metric_name == "Faithfulness":
                            # Faithfulness: 0.0-1.0，表示响应中忠实于上下文的主张比例
                            if score_value >= 0.8:
                                reason = f"响应高度忠实于上下文（{score_value:.1%}的主张得到支持，满分1.0）"
                            elif score_value >= 0.5:
                                reason = f"响应部分忠实于上下文（{score_value:.1%}的主张得到支持，满分1.0）"
                            elif score_value > 0:
                                reason = f"响应忠实度较低（仅{score_value:.1%}的主张得到支持，满分1.0），可能存在幻觉"
                            else:
                                reason = "响应完全不忠实于上下文（得分0.0/1.0），存在严重幻觉"
                        elif metric_name == "ResponseRelevancy":
                            # ResponseRelevancy: 0.0-1.0，表示响应与问题的相关性
                            if score_value >= 0.8:
                                reason = f"响应高度相关（相关性得分: {score_value:.1%}，满分1.0）"
                            elif score_value >= 0.5:
                                reason = f"响应部分相关（相关性得分: {score_value:.1%}，满分1.0），可能遗漏部分信息"
                            elif score_value > 0:
                                reason = f"响应相关性较低（相关性得分: {score_value:.1%}，满分1.0），可能未充分回答问题"
                            else:
                                reason = "响应与问题不相关（得分0.0/1.0），可能完全偏离主题"
                        elif metric_name == "ContextPrecision":
                            # ContextPrecision: 0.0-1.0，衡量检索到的上下文中与问题相关的比例
                            if score_value >= 0.8:
                                reason = f"检索到的上下文高度精确（{score_value:.1%}的上下文与问题相关，满分1.0）"
                            elif score_value >= 0.5:
                                reason = f"检索到的上下文部分精确（{score_value:.1%}的上下文与问题相关，满分1.0），存在无关上下文"
                            elif score_value > 0:
                                reason = f"检索到的上下文精确度较低（仅{score_value:.1%}的上下文与问题相关，满分1.0），存在较多噪声"
                            else:
                                reason = "检索到的上下文完全不相关（得分0.0/1.0），检索质量差"
                        elif metric_name == "ContextRecall":
                            # ContextRecall: 0.0-1.0，衡量检索到的上下文覆盖标准答案的程度
                            if score_value >= 0.8:
                                reason = f"检索到的上下文高度完整（覆盖了{score_value:.1%}的标准答案内容，满分1.0）"
                            elif score_value >= 0.5:
                                reason = f"检索到的上下文部分完整（覆盖了{score_value:.1%}的标准答案内容，满分1.0），遗漏部分信息"
                            elif score_value > 0:
                                reason = f"检索到的上下文完整性较低（仅覆盖{score_value:.1%}的标准答案内容，满分1.0），遗漏较多信息"
                            else:
                                reason = "检索到的上下文完全不包含标准答案内容（得分0.0/1.0），检索召回率低"
                        elif metric_name == "ContextEntityRecall":
                            # ContextEntityRecall: 0.0-1.0，衡量检索到的上下文中包含标准答案中实体的比例
                            if score_value >= 0.8:
                                reason = f"检索到的上下文包含大部分实体（{score_value:.1%}的标准答案实体在上下文中，满分1.0）"
                            elif score_value >= 0.5:
                                reason = f"检索到的上下文包含部分实体（{score_value:.1%}的标准答案实体在上下文中，满分1.0），遗漏部分实体"
                            elif score_value > 0:
                                reason = f"检索到的上下文实体覆盖率较低（仅{score_value:.1%}的标准答案实体在上下文中，满分1.0），遗漏较多实体"
                            else:
                                reason = "检索到的上下文不包含标准答案中的实体（得分0.0/1.0），实体召回率低"
                        elif metric_name == "AnswerCorrectness":
                            # AnswerCorrectness: 0.0-1.0，衡量答案的正确程度
                            if score_value >= 0.8:
                                reason = f"答案高度正确（正确性得分: {score_value:.1%}，满分1.0）"
                            elif score_value >= 0.5:
                                reason = f"答案部分正确（正确性得分: {score_value:.1%}，满分1.0），存在部分错误"
                            elif score_value > 0:
                                reason = f"答案正确性较低（正确性得分: {score_value:.1%}，满分1.0），存在较多错误"
                            else:
                                reason = "答案完全不正确（得分0.0/1.0）"
                        elif metric_name == "AnswerAccuracy":
                            # AnswerAccuracy: 0.0-1.0，衡量答案的准确程度
                            if score_value >= 0.8:
                                reason = f"答案高度准确（准确性得分: {score_value:.1%}，满分1.0）"
                            elif score_value >= 0.5:
                                reason = f"答案部分准确（准确性得分: {score_value:.1%}，满分1.0），存在偏差"
                            elif score_value > 0:
                                reason = f"答案准确性较低（准确性得分: {score_value:.1%}，满分1.0），存在较大偏差"
                            else:
                                reason = "答案完全不准确（得分0.0/1.0）"
                        elif metric_name == "ContextRelevance":
                            # ContextRelevance: 0.0-1.0，衡量检索到的上下文与问题的相关性
                            if score_value >= 0.8:
                                reason = f"检索到的上下文高度相关（相关性得分: {score_value:.1%}，满分1.0）"
                            elif score_value >= 0.5:
                                reason = f"检索到的上下文部分相关（相关性得分: {score_value:.1%}，满分1.0），存在无关内容"
                            elif score_value > 0:
                                reason = f"检索到的上下文相关性较低（相关性得分: {score_value:.1%}，满分1.0），存在较多无关内容"
                            else:
                                reason = "检索到的上下文与问题不相关（得分0.0/1.0）"
                        elif metric_name == "ResponseGroundedness":
                            # ResponseGroundedness: 0.0-1.0，衡量响应基于上下文的程度
                            if score_value >= 0.8:
                                reason = f"响应高度基于上下文（基础性得分: {score_value:.1%}，满分1.0）"
                            elif score_value >= 0.5:
                                reason = f"响应部分基于上下文（基础性得分: {score_value:.1%}，满分1.0），存在未基于上下文的内容"
                            elif score_value > 0:
                                reason = f"响应基础性较低（基础性得分: {score_value:.1%}，满分1.0），较多内容未基于上下文"
                            else:
                                reason = "响应完全不基于上下文（得分0.0/1.0），可能存在幻觉"
                        else:
                            # 其他指标的通用理由
                            if score_value >= 0.8:
                                reason = f"得分较高（{score_value:.4f}/1.0），表现良好"
                            elif score_value >= 0.5:
                                reason = f"得分中等（{score_value:.4f}/1.0），表现一般"
                            elif score_value > 0:
                                reason = f"得分较低（{score_value:.4f}/1.0），需要改进"
                            else:
                                reason = f"得分: {score_value:.4f}（满分: 1.0），表现较差"
                        
                        reasoning[metric_name] = reason
                        logger.info(f"[{idx}/{total_metrics}] ✅ 指标 {metric_name} 评估完成，得分: {score_value:.4f}/1.0 (耗时: {metric_duration:.2f}秒)")
                        logger.info(f"[{idx}/{total_metrics}]   评分理由: {reason}")
                    except asyncio.TimeoutError:
                        progress_task.cancel()
                        try:
                            await progress_task
                        except asyncio.CancelledError:
                            pass
                        metric_duration = time.time() - metric_start_time
                        error_msg = f"评估超时（{self.timeout}秒）"
                        scores[metric_name] = 0.0
                        errors[metric_name] = error_msg
                        logger.warning(f"[{idx}/{total_metrics}] ⏱️ 指标 {metric_name} 评估超时 (耗时: {metric_duration:.2f}秒)")
                        logger.warning(f"[{idx}/{total_metrics}] 超时原因分析：")
                        logger.warning(f"[{idx}/{total_metrics}]   1. LLM API响应过慢（当前超时设置: {self.timeout}秒）")
                        logger.warning(f"[{idx}/{total_metrics}]   2. Ragas的prompt较长，需要更多处理时间")
                        logger.warning(f"[{idx}/{total_metrics}]   3. 建议：在配置文件中增加timeout值（如180秒或240秒）")
                        logger.warning(f"[{idx}/{total_metrics}]   4. 检查网络连接和API服务状态")
                    except Exception as e:
                        progress_task.cancel()
                        try:
                            await progress_task
                        except asyncio.CancelledError:
                            pass
                        metric_duration = time.time() - metric_start_time
                        scores[metric_name] = 0.0
                        
                        # 检查是否是429限流错误
                        error_str = str(e)
                        if "429" in error_str or "Too Many Requests" in error_str:
                            error_msg = f"API限流错误（429 Too Many Requests）: {error_str}"
                            logger.error(f"[{idx}/{total_metrics}] 🚫 指标 {metric_name} 评估失败 - API限流")
                            logger.error(f"[{idx}/{total_metrics}] 建议：")
                            logger.error(f"[{idx}/{total_metrics}]   1. 检查API配额是否充足")
                            logger.error(f"[{idx}/{total_metrics}]   2. 减少并发请求数量")
                            logger.error(f"[{idx}/{total_metrics}]   3. 增加请求之间的延迟")
                            logger.error(f"[{idx}/{total_metrics}]   4. 联系API提供商提升配额")
                        else:
                            error_msg = str(e)
                            logger.warning(f"[{idx}/{total_metrics}] ❌ 指标 {metric_name} 评估失败 (耗时: {metric_duration:.2f}秒): {e}")
                        
                        errors[metric_name] = error_msg
                        # 记录详细的错误信息以便调试
                        import traceback
                        logger.debug(f"指标 {metric_name} 评估失败详情:\n{traceback.format_exc()}")
                except Exception as e:
                    metric_duration = time.time() - metric_start_time
                    logger.error(f"[{idx}/{total_metrics}] 指标 {metric_name} 评估过程异常 (耗时: {metric_duration:.2f}秒): {e}")
                    scores[metric_name] = 0.0
                    errors[metric_name] = str(e)

            # 评估完成统计
            eval_duration = time.time() - eval_start_time
            success_count = len(scores) - len(errors)
            logger.info(f"评估完成，成功: {success_count}/{total_metrics}, 失败: {len(errors)}/{total_metrics}, 总耗时: {eval_duration:.2f}秒")

            # 如果有错误，记录在metadata中
            error_msg = None
            if errors:
                error_msg = f"部分指标评估失败: {errors}"
                logger.warning(error_msg)

            # 准备响应文本（如果超过200字则截断）
            response_display = eval_sample.response
            response_full_length = len(eval_sample.response)
            if len(response_display) > 200:
                response_display = response_display[:200] + "..."
            
            # 准备召回内容（缩略显示，每个context最多显示100字）
            contexts_display = []
            for i, ctx in enumerate(eval_sample.contexts):
                if ctx and ctx.strip():
                    ctx_preview = ctx[:100] + "..." if len(ctx) > 100 else ctx
                    contexts_display.append(f"上下文{i+1}: {ctx_preview}")
                else:
                    contexts_display.append(f"上下文{i+1}: (空)")
            
            # 合并metadata，添加响应完整长度信息
            result_metadata = {
                **(eval_sample.metadata or {}),
                "response_full_length": response_full_length,
            }

            return SampleResult(
                scores=scores,
                reasoning=reasoning,
                error=error_msg,
                user_input=eval_sample.user_input,
                response=response_display,  # 截断后的响应
                reference=eval_sample.reference,
                contexts=contexts_display,  # 缩略后的上下文列表
                metadata=result_metadata,
            )

        except Exception as e:
            eval_duration = time.time() - eval_start_time
            logger.error(f"评估过程中发生错误 (耗时: {eval_duration:.2f}秒): {e}")
            import traceback
            logger.debug(f"评估错误详情:\n{traceback.format_exc()}")
            return SampleResult(error=str(e))
