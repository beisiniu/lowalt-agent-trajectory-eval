"""YML配置加载器"""

from pathlib import Path

import yaml

from agent_evaluator.core.config import EvalConfig


def load_config(config_path: str | Path) -> EvalConfig:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        EvalConfig对象

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置格式错误
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not config_dict:
        raise ValueError("配置文件为空")

    try:
        return EvalConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"配置格式错误: {e}") from e
