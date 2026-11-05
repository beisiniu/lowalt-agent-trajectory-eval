"""日志配置模块"""

import logging
import sys
from loguru import logger


class InterceptHandler(logging.Handler):
    """
    拦截标准logging模块的输出，转发到loguru
    """
    def emit(self, record):
        # 获取对应的loguru级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用者信息
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logger(level: str = "INFO", format_type: str = "detailed") -> None:
    """
    配置loguru日志系统，并拦截标准logging模块的输出

    Args:
        level: 日志级别，默认为INFO
        format_type: 日志格式类型：detailed(详细), simple(简单), json(JSON格式)
    """
    # 移除默认的handler
    logger.remove()

    # 根据格式类型选择不同的格式
    if format_type == "json":
        # JSON格式，方便日志收集和分析
        format_str = "{time} | {level} | {name}:{function}:{line} | {message}"
        serialize = True
    elif format_type == "simple":
        # 简单格式
        format_str = "<level>{level: <8}</level> | <level>{message}</level>"
        serialize = False
    else:
        # 详细格式（默认）
        format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        serialize = False

    # 添加控制台输出handler
    logger.add(
        sys.stderr,
        format=format_str,
        level=level,
        colorize=(format_type != "json"),  # JSON格式不启用颜色
        serialize=serialize,
    )
    
    # 拦截标准logging模块的输出，转发到loguru
    # 这样ragas、langchain等使用logging的库的日志也会通过loguru输出
    intercept_handler = InterceptHandler()
    
    # 设置ragas相关库的日志级别
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(level, logging.INFO)
    
    # 为ragas、langchain、openai等库配置日志拦截
    for logger_name in ["ragas", "langchain", "langchain_openai", "openai", "httpx"]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [intercept_handler]
        logging_logger.setLevel(log_level)
        logging_logger.propagate = False


def get_logger(name: str = None):
    """
    获取logger实例

    Args:
        name: logger名称，通常使用模块名

    Returns:
        logger实例
    """
    if name:
        return logger.bind(name=name)
    return logger

