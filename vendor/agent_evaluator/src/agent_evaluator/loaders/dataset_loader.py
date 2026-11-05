"""测试数据集加载器"""

import json
from pathlib import Path
from typing import Any

from agent_evaluator.core.sample import TestSample


def load_dataset(dataset_path: str | Path) -> list[TestSample]:
    """
    加载JSON格式的数据集

    Args:
        dataset_path: 数据集文件路径

    Returns:
        TestSample列表

    Raises:
        FileNotFoundError: 数据集文件不存在
        ValueError: 数据集格式错误
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("数据集必须是JSON数组格式")

    samples: list[TestSample] = []
    for i, item in enumerate(data):
        try:
            # 支持Ragas格式（retrieved_contexts）和我们的格式（reference_contexts）
            reference_contexts = item.get("reference_contexts") or item.get("retrieved_contexts")

            sample = TestSample(
                user_input=item.get("user_input", ""),
                reference=item.get("reference"),
                reference_contexts=reference_contexts,
                metadata={
                    "index": i,
                    **{k: v for k, v in item.items() if k not in ["user_input", "reference", "reference_contexts", "retrieved_contexts"]},
                },
            )
            samples.append(sample)
        except Exception as e:
            raise ValueError(f"数据集第{i+1}项格式错误: {e}") from e

    if not samples:
        raise ValueError("数据集为空")

    return samples
