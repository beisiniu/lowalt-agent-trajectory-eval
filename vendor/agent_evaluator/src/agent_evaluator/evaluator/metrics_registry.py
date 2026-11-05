"""指标注册表（简化配置）"""

from typing import Any

__all__ = [
    "create_metric",
    "create_metrics",
    "expand_metric_categories",
    "METRIC_CATEGORIES",
    "PERFORMANCE_METRICS",
]

# 基础指标
from ragas.metrics import (
    AnswerAccuracy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    ResponseRelevancy,
)

# 尝试导入更多指标（某些指标可能在不同版本的Ragas中可用）
try:
    from ragas.metrics import ContextEntityRecall
except ImportError:
    ContextEntityRecall = None

try:
    from ragas.metrics import NoiseSensitivity
except ImportError:
    NoiseSensitivity = None

try:
    from ragas.metrics import ContextRelevance
except ImportError:
    ContextRelevance = None

try:
    from ragas.metrics import ResponseGroundedness
except ImportError:
    ResponseGroundedness = None

try:
    from ragas.metrics import SemanticSimilarity
except ImportError:
    SemanticSimilarity = None

try:
    from ragas.metrics import BleuScore
except ImportError:
    BleuScore = None

try:
    from ragas.metrics import RougeScore
except ImportError:
    RougeScore = None

try:
    from ragas.metrics import ChrfScore
except ImportError:
    ChrfScore = None

try:
    from ragas.metrics import ExactMatch
except ImportError:
    ExactMatch = None

try:
    from ragas.metrics import StringPresence
except ImportError:
    StringPresence = None

try:
    from ragas.metrics._string import NonLLMStringSimilarity
except ImportError:
    NonLLMStringSimilarity = None

try:
    from ragas.metrics import AspectCritic
except ImportError:
    AspectCritic = None

try:
    from ragas.metrics import SimpleCriteriaScore
except ImportError:
    SimpleCriteriaScore = None

try:
    from ragas.metrics import RubricsScore
except ImportError:
    RubricsScore = None

try:
    from ragas.metrics import SummarizationScore
except ImportError:
    SummarizationScore = None

try:
    from ragas.metrics import LLMSQLEquivalence
except ImportError:
    LLMSQLEquivalence = None

try:
    from ragas.metrics import DataCompyScore
except ImportError:
    DataCompyScore = None

# 多轮对话指标
try:
    from ragas.metrics import TopicAdherenceScore
except ImportError:
    TopicAdherenceScore = None

try:
    from ragas.metrics import ToolCallAccuracy
except ImportError:
    ToolCallAccuracy = None

try:
    from ragas.metrics import AgentGoalAccuracyWithReference
except ImportError:
    AgentGoalAccuracyWithReference = None

try:
    from ragas.metrics import AgentGoalAccuracyWithoutReference
except ImportError:
    AgentGoalAccuracyWithoutReference = None


def create_metric(
    metric_name: str, 
    llm: Any, 
    embeddings: Any | None = None,
    strictness: int | None = None,
) -> Any:
    """
    根据指标名称创建Ragas指标对象

    Args:
        metric_name: 指标名称（如 "faithfulness", "relevancy"等）
        llm: 评估用的LLM对象
        embeddings: 评估用的embeddings对象（可选）
        strictness: ResponseRelevancy的strictness参数（生成问题的数量，默认3，建议2以减少超时）

    Returns:
        Ragas指标对象

    Raises:
        ValueError: 不支持的指标名称
    """
    metric_name_lower = metric_name.lower().strip()

    # 映射指标名称到Ragas类
    # 注意：某些指标可能在不同版本的Ragas中不可用
    metric_map: dict[str, type | None] = {
        # RAG核心指标
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "response_relevancy": ResponseRelevancy,
        "relevancy": ResponseRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
        "context_entity_recall": ContextEntityRecall,
        "noise_sensitivity": NoiseSensitivity,
        # Nvidia高效指标
        "context_relevance": ContextRelevance,
        "response_groundedness": ResponseGroundedness,
        "answer_correctness": AnswerCorrectness,
        "answer_accuracy": AnswerAccuracy,
        # 文本相似度指标
        "semantic_similarity": SemanticSimilarity,
        "bleu_score": BleuScore,
        "bleu": BleuScore,
        "rouge_score": RougeScore,
        "rouge": RougeScore,
        "chrf_score": ChrfScore,
        "chrf": ChrfScore,
        # 字符串匹配指标
        "exact_match": ExactMatch,
        "string_presence": StringPresence,
        "non_llm_string_similarity": NonLLMStringSimilarity,
        # 自定义评分指标
        "aspect_critic": AspectCritic,
        "simple_criteria_score": SimpleCriteriaScore,
        "rubrics_score": RubricsScore,
        # 专项领域指标
        "summarization_score": SummarizationScore,
        "llm_sql_equivalence": LLMSQLEquivalence,
        "data_compy_score": DataCompyScore,
        # 多轮对话指标（需要MultiTurnSample）
        "topic_adherence_score": TopicAdherenceScore,
        "tool_call_accuracy": ToolCallAccuracy,
        "agent_goal_accuracy": AgentGoalAccuracyWithReference,
        "agent_goal_accuracy_with_reference": AgentGoalAccuracyWithReference,
        "agent_goal_accuracy_without_reference": AgentGoalAccuracyWithoutReference,
    }
    
    # 过滤掉不可用的指标
    available_metrics = {k: v for k, v in metric_map.items() if v is not None}

    metric_class = available_metrics.get(metric_name_lower)
    if not metric_class:
        available_names = sorted(available_metrics.keys())
        raise ValueError(
            f"不支持的指标: {metric_name}。支持的指标: {', '.join(available_names)}"
        )

    # 创建指标对象
    # 根据指标类型，某些需要embeddings，某些只需要llm，某些不需要任何参数
    
    # 需要embeddings的指标
    if metric_name_lower in ["answer_relevancy", "response_relevancy", "relevancy", "semantic_similarity"]:
        if embeddings is None:
            raise ValueError(f"指标 {metric_name} 需要 embeddings 参数")
        
        if metric_name_lower in ["answer_relevancy", "response_relevancy", "relevancy"]:
            # ResponseRelevancy支持strictness参数
            if strictness is not None:
                metric_instance = metric_class(llm=llm, embeddings=embeddings, strictness=strictness)
            else:
                # 默认使用2而不是3，以减少LLM调用次数和超时风险
                metric_instance = metric_class(llm=llm, embeddings=embeddings, strictness=2)
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"已创建 {metric_name} 指标，strictness={metric_instance.strictness if hasattr(metric_instance, 'strictness') else 'N/A'}")
            return metric_instance
        else:
            # SemanticSimilarity只需要embeddings
            return metric_class(embeddings=embeddings)
    
    # ToolCallAccuracy不需要任何参数
    elif metric_name_lower == "tool_call_accuracy":
        return metric_class()
    
    # 需要LLM的指标
    elif metric_name_lower in [
        "faithfulness", "context_precision", "context_recall", 
        "context_entity_recall", "noise_sensitivity",
        "context_relevance", "response_groundedness",
        "answer_correctness", "answer_accuracy",
        "aspect_critic", "simple_criteria_score", "rubrics_score",
        "summarization_score", "llm_sql_equivalence",
        "topic_adherence_score",
        "agent_goal_accuracy", "agent_goal_accuracy_with_reference",
        "agent_goal_accuracy_without_reference"
    ]:
        return metric_class(llm=llm)
    
    # 不需要参数的指标（非LLM指标）
    else:
        # BleuScore, RougeScore, ChrfScore, ExactMatch, StringPresence, NonLLMStringSimilarity, DataCompyScore
        return metric_class()


# 指标分类定义（根据智能体测评指标文档）
METRIC_CATEGORIES: dict[str, list[str]] = {
    "rag": [
        # RAG核心指标
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "noise_sensitivity",
        "response_relevancy",
        "faithfulness",
        # Nvidia高效指标（RAG相关）
        "context_relevance",
        "response_groundedness",
        "answer_correctness",
        "answer_accuracy",
    ],
    "agent": [
        # 智能体或工具使用指标
        "topic_adherence_score",
        "tool_call_accuracy",
        "agent_goal_accuracy",
        "agent_goal_accuracy_with_reference",
        "agent_goal_accuracy_without_reference",
    ],
    "llm": [
        # 文本相似度指标
        "semantic_similarity",
        # n-gram重叠指标
        "bleu_score",
        "rouge_score",
        "chrf_score",
        # 字符串匹配指标
        "exact_match",
        "string_presence",
        "non_llm_string_similarity",
        # 自定义评分指标
        "aspect_critic",
        "simple_criteria_score",
        "rubrics_score",
        # 专项领域指标
        "summarization_score",
        "llm_sql_equivalence",
        "data_compy_score",
    ],
}

# 性能指标（自动收集，不需要配置）
PERFORMANCE_METRICS = [
    "total_time",
    "time_to_first_token",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "streaming_latency",
]


def expand_metric_categories(metric_names: list[str]) -> list[str]:
    """
    展开指标类别为具体指标名称列表
    
    Args:
        metric_names: 指标名称或类别列表（如 ["rag", "context_precision"]）
    
    Returns:
        展开后的指标名称列表（去重）
    
    Examples:
        >>> expand_metric_categories(["rag", "context_precision"])
        ["context_precision", "context_recall", "context_entity_recall", ...]
    """
    expanded = []
    for name in metric_names:
        name_lower = name.lower().strip()
        
        # 如果是类别，展开为具体指标
        if name_lower in METRIC_CATEGORIES:
            expanded.extend(METRIC_CATEGORIES[name_lower])
        else:
            # 否则作为具体指标名称
            expanded.append(name)
    
    # 去重并保持顺序
    seen = set()
    result = []
    for metric in expanded:
        if metric not in seen:
            seen.add(metric)
            result.append(metric)
    
    return result


def create_metrics(
    metric_names: list[str], llm: Any, embeddings: Any | None = None
) -> list[Any]:
    """
    批量创建指标对象，支持类别配置
    
    Args:
        metric_names: 指标名称或类别列表（如 ["rag", "context_precision"]）
        llm: 评估用的LLM对象
        embeddings: 评估用的embeddings对象（可选）
    
    Returns:
        Ragas指标对象列表
    
    Examples:
        >>> create_metrics(["rag", "context_precision"], llm, embeddings)
        [ContextPrecision(...), ContextRecall(...), ...]
    """
    from agent_evaluator.utils.logger import get_logger
    
    logger = get_logger(__name__)
    
    # 先展开类别
    expanded_names = expand_metric_categories(metric_names)
    
    # 需要embeddings的指标列表
    metrics_requiring_embeddings = ["answer_relevancy", "response_relevancy", "relevancy", "semantic_similarity"]
    
    # 多轮对话指标列表（需要MultiTurnSample，当前数据格式不支持）
    multi_turn_metrics = [
        "topic_adherence_score",
        "tool_call_accuracy",
        "agent_goal_accuracy",
        "agent_goal_accuracy_with_reference",
        "agent_goal_accuracy_without_reference",
    ]
    
    # 创建指标，跳过需要embeddings但embeddings为None的指标，以及多轮对话指标
    metrics = []
    skipped_metrics = []
    
    for name in expanded_names:
        name_lower = name.lower().strip()
        
        # 检查是否为多轮对话指标
        if name_lower in multi_turn_metrics:
            skipped_metrics.append(name)
            logger.warning(f"跳过指标 {name}：该指标是多轮对话指标，需要MultiTurnSample数据格式，当前数据为单轮对话格式，无法评估")
            continue
        
        # 检查是否需要embeddings
        if name_lower in metrics_requiring_embeddings and embeddings is None:
            skipped_metrics.append(name)
            logger.warning(f"跳过指标 {name}：该指标需要 embeddings 参数，但未配置 embeddings")
            continue
        
        try:
            metric = create_metric(name, llm, embeddings)
            metrics.append(metric)
        except Exception as e:
            logger.warning(f"跳过指标 {name}：创建失败 - {e}")
            skipped_metrics.append(name)
    
    if not metrics:
        # 分析跳过的原因
        multi_turn_skipped = [m for m in skipped_metrics if m.lower() in multi_turn_metrics]
        embeddings_skipped = [m for m in skipped_metrics if m.lower() in metrics_requiring_embeddings]
        other_skipped = [m for m in skipped_metrics if m.lower() not in multi_turn_metrics and m.lower() not in metrics_requiring_embeddings]
        
        error_parts = []
        suggestions = []
        
        if multi_turn_skipped:
            error_parts.append(f"多轮对话指标（需要MultiTurnSample数据格式）: {', '.join(multi_turn_skipped)}")
            suggestions.append("这些指标需要多轮对话数据格式（MultiTurnSample），当前数据为单轮对话格式。建议：1) 使用 'rag' 或 'llm' 类别的指标（支持单轮对话）；2) 或准备多轮对话格式的数据")
        
        if embeddings_skipped:
            error_parts.append(f"需要 embeddings 的指标: {', '.join(embeddings_skipped)}")
            suggestions.append("这些指标需要 embeddings 参数。建议：在配置文件中添加 embeddings_model 配置")
        
        if other_skipped:
            error_parts.append(f"其他原因: {', '.join(other_skipped)}")
        
        error_msg = "无法创建任何指标。"
        if error_parts:
            error_msg += "\n跳过的指标：" + "；".join(error_parts) + "。"
        if suggestions:
            error_msg += "\n\n建议：" + "\n".join(f"  • {s}" for s in suggestions)
        
        raise ValueError(error_msg)
    
    if skipped_metrics:
        skipped_reasons = []
        multi_turn_skipped = [m for m in skipped_metrics if m.lower() in multi_turn_metrics]
        embeddings_skipped = [m for m in skipped_metrics if m.lower() in metrics_requiring_embeddings]
        other_skipped = [m for m in skipped_metrics if m.lower() not in multi_turn_metrics and m.lower() not in metrics_requiring_embeddings]
        
        if multi_turn_skipped:
            skipped_reasons.append(f"{len(multi_turn_skipped)} 个多轮对话指标")
        if embeddings_skipped:
            skipped_reasons.append(f"{len(embeddings_skipped)} 个需要 embeddings 的指标")
        if other_skipped:
            skipped_reasons.append(f"{len(other_skipped)} 个其他指标")
        
        reason_str = "、".join(skipped_reasons)
        logger.info(f"成功创建 {len(metrics)} 个指标，跳过 {len(skipped_metrics)} 个指标（{reason_str}）")
    
    return metrics
