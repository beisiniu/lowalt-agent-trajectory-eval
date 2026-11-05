"""命令行入口"""

import asyncio
import sys
from pathlib import Path

import warnings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper

# 注意：LangchainLLMWrapper已弃用，但由于我们需要支持自定义base_url（如智谱API）
# 而llm_factory需要直接使用OpenAI客户端，无法方便地支持自定义base_url
# 因此继续使用LangchainLLMWrapper，并抑制弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.llms")

from agent_evaluator.adapters.dify import DifyAdapter
from agent_evaluator.core.config import EvalConfig
from agent_evaluator.evaluator.executor import EvaluatorExecutor
from agent_evaluator.evaluator.metrics_registry import create_metrics
from agent_evaluator.loaders.config_loader import load_config
from agent_evaluator.loaders.dataset_loader import load_dataset
from agent_evaluator.reporters.console import ConsoleReporter
from agent_evaluator.reporters.csv_reporter import CSVReporter
from agent_evaluator.reporters.html_reporter import HTMLReporter
from agent_evaluator.reporters.json_reporter import JSONReporter
from agent_evaluator.runner import EvaluationRunner
from agent_evaluator.utils.logger import get_logger, setup_logger

# 临时初始化日志系统（在加载配置前）
setup_logger()
logger = get_logger(__name__)


def create_llm(config: EvalConfig):
    """创建LLM对象"""
    llm_config = config.evaluator_llm

    # 获取base_url，如果为None则使用默认值
    base_url = llm_config.base_url or "https://api.openai.com/v1"
    
    # 获取timeout，默认120秒（Ragas的prompt较长，智谱API响应较慢，需要更长的超时时间）
    timeout = llm_config.timeout if llm_config.timeout is not None else 120.0
    
    # 获取max_retries，默认3次
    max_retries = llm_config.max_retries if llm_config.max_retries is not None else 3

    if llm_config.provider == "openai":
        # 配置重试机制，特别是针对429限流错误
        # max_retries: 最大重试次数，默认3次
        # 对于429错误，OpenAI客户端会自动重试，但我们需要确保有足够的重试次数
        chat_llm = ChatOpenAI(
            model=llm_config.model,
            api_key=llm_config.api_key,
            base_url=base_url,
            timeout=timeout,  # 添加超时设置
            max_retries=max_retries,  # 增加重试次数以应对429限流
        )
        wrapped_llm = LangchainLLMWrapper(chat_llm)
        
        logger.info(f"已创建LLM: 模型={llm_config.model}, API端点={base_url}, 超时={timeout}秒, 最大重试={max_retries}次")
        return wrapped_llm
    elif llm_config.provider == "langgenius":
        chat_llm = ChatOpenAI(
            model=llm_config.model,
            api_key=llm_config.api_key,
            base_url=base_url or "https://api.dify.ai/v1",
            timeout=timeout,  # 添加超时设置
            max_retries=max_retries,  # 增加重试次数以应对429限流
        )
        wrapped_llm = LangchainLLMWrapper(chat_llm)
        
        logger.info(f"已创建LLM (Langgenius): 模型={llm_config.model}, API端点={base_url or 'https://api.dify.ai/v1'}, 超时={timeout}秒, 最大重试={max_retries}次")
        return wrapped_llm
    else:
        raise ValueError(f"不支持的LLM提供商: {llm_config.provider}")


def create_embeddings(config: EvalConfig):
    """创建Embeddings对象（如果需要）"""
    from agent_evaluator.evaluator.metrics_registry import expand_metric_categories
    
    llm_config = config.evaluator_llm
    
    # 先展开类别，再检查是否需要embeddings
    expanded_metrics = expand_metric_categories(config.metrics)
    
    # 检查是否需要embeddings
    metrics_requiring_embeddings = ["answer_relevancy", "response_relevancy", "relevancy", "semantic_similarity"]
    needs_embeddings = any(
        metric.lower() in metrics_requiring_embeddings
        for metric in expanded_metrics
    )

    if not needs_embeddings:
        logger.debug("当前配置的指标不需要 embeddings，跳过创建")
        return None

    # 获取base_url，如果为None则使用默认值
    base_url = llm_config.base_url or "https://api.openai.com/v1"
    
    # 获取timeout，默认120秒（与LLM保持一致）
    timeout = llm_config.timeout if llm_config.timeout is not None else 120.0

    if llm_config.provider == "openai":
        # 优先使用配置文件中指定的embeddings模型
        if llm_config.embeddings_model:
            logger.info(f"使用配置的embeddings模型: {llm_config.embeddings_model}")
            return OpenAIEmbeddings(
                api_key=llm_config.api_key,
                base_url=base_url,
                model=llm_config.embeddings_model,
                timeout=timeout,  # 添加超时设置
            )
        # 如果没有配置，自动检测智谱API
        elif "bigmodel.cn" in base_url or "zhipu" in base_url.lower():
            # 智谱API需要使用特定的embeddings模型
            # 智谱的embeddings模型名称通常是 embedding-2
            logger.info("检测到智谱API，自动使用智谱embeddings模型: embedding-2")
            return OpenAIEmbeddings(
                api_key=llm_config.api_key,
                base_url=base_url,
                model="embedding-2",  # 智谱的embeddings模型
                timeout=timeout,  # 添加超时设置
            )
        else:
            # 标准OpenAI API，使用默认模型
            logger.debug("使用OpenAI默认embeddings模型")
            return OpenAIEmbeddings(
                api_key=llm_config.api_key,
                base_url=base_url,
                timeout=timeout,  # 添加超时设置
            )
    elif llm_config.provider == "langgenius":
        # Langgenius也支持指定embeddings模型
        embeddings_model = llm_config.embeddings_model
        if embeddings_model:
            logger.info(f"使用配置的embeddings模型: {embeddings_model}")
            return OpenAIEmbeddings(
                api_key=llm_config.api_key,
                base_url=base_url or "https://api.dify.ai/v1",
                model=embeddings_model,
                timeout=timeout,  # 添加超时设置
            )
        else:
            # 使用默认embeddings模型
            logger.debug("使用默认embeddings模型")
            return OpenAIEmbeddings(
                api_key=llm_config.api_key,
                base_url=base_url or "https://api.dify.ai/v1",
                timeout=timeout,  # 添加超时设置
            )
    else:
        raise ValueError(f"不支持的Embeddings提供商: {llm_config.provider}")


def create_adapter(config: EvalConfig):
    """创建平台适配器"""
    platform = config.platform.lower()

    if platform == "dify":
        api_config = {
            "api_key": config.api_config.api_key,
            "base_url": config.api_config.base_url,
            "timeout": config.api_config.timeout,
            "app_id": config.api_config.app_id,
        }
        return DifyAdapter(api_config, show_streaming_content=config.log.show_streaming_content)

    if platform == "langgraph":
        from agent_evaluator.adapters.langgraph import LangGraphAdapter

        api_config = {
            "api_key": config.api_config.api_key,
            "base_url": config.api_config.base_url,
            "timeout": config.api_config.timeout,
            **(config.api_config.model_extra or {}),
        }
        return LangGraphAdapter(api_config)

    raise ValueError(f"不支持的平台: {platform}")


def create_reporters(config: EvalConfig):
    """创建报告器列表"""
    reporters = []
    save_path = Path(config.output.save_path)

    for fmt in config.output.format:
        if fmt == "console":
            reporters.append(ConsoleReporter())
        elif fmt == "json":
            reporters.append(JSONReporter())
        elif fmt == "html":
            reporters.append(HTMLReporter())
        elif fmt == "csv":
            reporters.append(CSVReporter())
        else:
            logger.warning(f"不支持的输出格式: {fmt}")

    return reporters, save_path


async def run_evaluation(config_path: str):
    """运行评估"""
    try:
        # 加载配置
        logger.info(f"加载配置文件: {config_path}")
        config = load_config(config_path)
        
        # 根据配置文件重新初始化日志系统
        # setup_logger会自动拦截ragas、langchain等库的logging输出
        setup_logger(level=config.log.level, format_type=config.log.format)
        logger.info(f"日志级别设置为: {config.log.level}, 格式: {config.log.format}")
        logger.info(f"已启用Ragas/LangChain/OpenAI日志拦截，所有日志将通过loguru统一输出")
        if config.log.show_streaming_content:
            logger.info("已启用流式输出详细内容显示")
        
        logger.debug(f"配置加载成功: platform={config.platform}, metrics={config.metrics}")

        # 加载数据集
        logger.info(f"加载数据集: {config.dataset}")
        test_samples = load_dataset(config.dataset)
        logger.info(f"已加载 {len(test_samples)} 个测试样本")

        # 创建LLM和Embeddings
        logger.info("初始化评估器LLM...")
        llm = create_llm(config)
        embeddings = create_embeddings(config)
        if embeddings:
            logger.info("已创建Embeddings对象")

        # 创建指标（支持类别配置）
        from agent_evaluator.evaluator.metrics_registry import expand_metric_categories
        
        expanded_metrics = expand_metric_categories(config.metrics)
        if expanded_metrics != config.metrics:
            logger.info(f"配置的指标/类别: {', '.join(config.metrics)}")
            logger.info(f"展开后的指标: {', '.join(expanded_metrics)}")
        else:
            logger.info(f"创建评估指标: {', '.join(config.metrics)}")
        
        metrics = create_metrics(config.metrics, llm, embeddings)
        logger.info(f"已创建 {len(metrics)} 个评估指标对象")
        
        # 性能指标说明
        logger.debug("性能指标（total_time, time_to_first_token, total_tokens等）将自动收集，无需配置")

        # 创建评估执行器
        # 超时时间计算：每个指标可能需要多次LLM调用
        # - Faithfulness: 2次LLM调用（生成statements + 验证statements）
        # - ResponseRelevancy: strictness次LLM调用（默认2次）
        # 假设每次LLM调用需要30-40秒，设置超时时间
        # 使用配置中的timeout，如果没有配置则使用默认值180秒
        # 评估器超时时间应该比LLM API超时时间更长，因为一个指标可能需要多次LLM调用
        llm_timeout = config.evaluator_llm.timeout if config.evaluator_llm.timeout is not None else 120.0
        # 评估器超时时间 = LLM超时时间 * 2（因为一个指标可能需要2次LLM调用）
        # 但最少180秒，最多600秒（10分钟）
        evaluator_timeout = max(180.0, min(llm_timeout * 2, 600.0))
        evaluator = EvaluatorExecutor(
            metrics=metrics, 
            llm=llm, 
            embeddings=embeddings,
            timeout=evaluator_timeout
        )
        logger.info(f"评估器超时时间设置为: {evaluator_timeout}秒（基于LLM超时时间: {llm_timeout}秒）")

        # 创建适配器
        logger.info(f"初始化平台适配器: {config.platform}")
        adapter = create_adapter(config)

        # 创建运行器
        runner = EvaluationRunner(
            adapter=adapter,
            evaluator=evaluator,
            stream=config.stream,
        )

        # 执行评估
        logger.info("开始评估...")
        report = await runner.evaluate_batch(test_samples)

        # 生成报告
        logger.info("生成报告...")
        reporters, save_path = create_reporters(config)
        save_path.mkdir(parents=True, exist_ok=True)

        for reporter in reporters:
            reporter.generate(report)
            if hasattr(reporter, "save") and reporter.__class__.__name__ != "ConsoleReporter":
                # 生成文件名
                if isinstance(reporter, JSONReporter):
                    file_path = save_path / "report.json"
                elif isinstance(reporter, HTMLReporter):
                    file_path = save_path / "report.html"
                elif isinstance(reporter, CSVReporter):
                    file_path = save_path / "report.csv"
                else:
                    continue

                reporter.save(report, str(file_path))
                logger.info(f"报告已保存: {file_path}")

        logger.success(f"评估完成！总样本数: {report.total_samples}, 成功: {report.total_samples - report.failed_samples}, 失败: {report.failed_samples}")
        return 0

    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1


def main():
    """主函数"""
    if len(sys.argv) < 2:
        logger.error("用法: agent-eval run <config.yml>")
        sys.exit(1)

    config_path = sys.argv[1]
    exit_code = asyncio.run(run_evaluation(config_path))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
