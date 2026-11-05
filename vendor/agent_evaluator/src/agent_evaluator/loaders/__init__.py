"""配置和数据加载器"""

from agent_evaluator.loaders.config_loader import load_config
from agent_evaluator.loaders.dataset_loader import load_dataset

__all__ = [
    "load_config",
    "load_dataset",
]
