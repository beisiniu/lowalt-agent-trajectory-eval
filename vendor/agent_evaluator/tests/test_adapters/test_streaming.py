"""测试StreamingAccumulator"""

import pytest

from agent_evaluator.adapters.streaming import StreamingAccumulator


class TestStreamingAccumulator:
    """测试流式累积器"""

    def test_accumulate_message_events(self):
        """测试累积message事件"""
        accumulator = StreamingAccumulator()
        start_time = 0.0

        # 模拟第一个message事件
        event1 = {
            "event": "message",
            "message_id": "test-123",
            "answer": "Hello",
            "created_at": 1000,
        }
        accumulator.accumulate(event1, 0.1)

        assert accumulator.answer == "Hello"
        assert accumulator.first_token_time == 0.1
        assert accumulator.last_token_time == 0.1

        # 模拟第二个message事件
        event2 = {
            "event": "message",
            "message_id": "test-123",
            "answer": " World",
            "created_at": 1001,
        }
        accumulator.accumulate(event2, 0.2)

        assert accumulator.answer == "Hello World"
        assert accumulator.first_token_time == 0.1
        assert accumulator.last_token_time == 0.2
        assert len(accumulator.token_timestamps) == 2

    def test_accumulate_workflow_finished(self):
        """测试累积workflow_finished事件"""
        accumulator = StreamingAccumulator()
        start_time = 0.0

        event = {
            "event": "workflow_finished",
            "task_id": "test-123",
            "data": {
                "id": "workflow-123",
                "elapsed_time": 1.5,
                "total_tokens": 100,
                "execution_metadata": {
                    "total_tokens": 100,
                    "total_price": 0.01,
                    "currency": "USD",
                },
            },
        }
        accumulator.accumulate(event, 1.5)

        assert accumulator.elapsed_time == 1.5
        assert accumulator.total_tokens == 100
        assert accumulator.total_price == 0.01
        assert accumulator.currency == "USD"

    def test_to_performance_metrics(self):
        """测试转换为性能指标"""
        accumulator = StreamingAccumulator()
        start_time = 0.0

        # 累积一些事件
        accumulator.accumulate(
            {"event": "message", "answer": "Hello"},
            0.1,
        )
        accumulator.accumulate(
            {"event": "message", "answer": " World"},
            0.2,
        )
        accumulator.accumulate(
            {
                "event": "workflow_finished",
                "data": {
                    "elapsed_time": 0.5,
                    "total_tokens": 50,
                },
            },
            0.5,
        )

        metrics = accumulator.to_performance_metrics(start_time)

        assert metrics.total_time == 0.5
        assert metrics.time_to_first_token == 0.1
        assert metrics.total_tokens == 50
        assert len(metrics.streaming_latency) == 1  # 两个token之间有一个延迟

    def test_to_adapter_response(self):
        """测试转换为AdapterResponse"""
        accumulator = StreamingAccumulator()
        start_time = 0.0

        accumulator.accumulate(
            {"event": "message", "answer": "Test answer"},
            0.1,
        )
        accumulator.accumulate(
            {
                "event": "workflow_finished",
                "data": {
                    "elapsed_time": 0.5,
                    "total_tokens": 100,
                },
            },
            0.5,
        )

        response = accumulator.to_adapter_response(start_time)

        assert response.answer == "Test answer"
        assert response.performance is not None
        assert response.performance.total_time == 0.5
        assert response.performance.total_tokens == 100

