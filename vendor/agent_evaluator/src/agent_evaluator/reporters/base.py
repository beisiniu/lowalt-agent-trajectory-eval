"""基础Reporter接口"""

from abc import ABC, abstractmethod

from agent_evaluator.core.result import EvalReport


class BaseReporter(ABC):
    """报告器基类"""

    @abstractmethod
    def generate(self, report: EvalReport) -> str:
        """
        生成报告

        Args:
            report: 评估报告

        Returns:
            报告内容（字符串或文件路径）
        """
        pass

    @abstractmethod
    def save(self, report: EvalReport, path: str) -> None:
        """
        保存报告到文件

        Args:
            report: 评估报告
            path: 保存路径
        """
        pass
