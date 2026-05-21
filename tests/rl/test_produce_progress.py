"""ProduceProgress 的深模块契约测试。

Good Tests:
- 通过 ProduceProgress 的公开构造和方法验证“累计 target / consumed / metrics”的领域行为。
- 测试描述 progress 对 producer/consumer 可见的行为结果，而不是 manager 或 strategy 如何调用它。
- 这些测试应在字段内部实现调整后仍保持稳定，只要公开方法语义不变。

Bad Tests:
- 不测试 AgentLoopManager 的私有 `_produce_progress` 字段推进。
- 不通过 mock manager/strategy 调用顺序来间接证明 progress 行为。

本文件主要覆盖的 public 行为:
- global progress 按 future step 维护绝对累计 target。
- local progress 表达单次共卡 produce_batch，不污染 global progress。
- consumer 按实际取出的 Rollout Group 推进 consumed 和 next consumer step。
- state_dict/load_state_dict 原地恢复状态，metrics 被读取一次后清零。
"""

import unittest

from xtuner.v1.rl.agent_loop_manager import ProduceProgress


class TestProduceProgress(unittest.TestCase):
    def test_global_progress_accumulates_absolute_targets_by_future_step(self):
        # 验证 global progress 按 future step 累计绝对 target，而不是只记录当前 batch 缺口。
        progress = ProduceProgress.build(["task_a", "task_b"])

        def allocate(batch_size: int, step: int) -> dict[str, int]:
            self.assertEqual(batch_size, 4)
            return {"task_a": step, "task_b": batch_size - step}

        current_sizes = progress.ensure_target_upto(
            batch_size=4,
            future_step=2,
            allocate_batch_sizes=allocate,
        )

        self.assertEqual(current_sizes, {"task_a": 2, "task_b": 2})
        self.assertEqual(progress.target_samples, {"task_a": 3, "task_b": 5})
        self.assertEqual(progress.target_upto_future_step, 2)

    def test_consumption_records_actual_taken_groups_and_next_consumer_step(self):
        # 验证 consumer 只按实际取出的 rollout group 数更新 consumed，并推进下一消费 step。
        progress = ProduceProgress.build(["task_a", "task_b"])

        progress.begin_consume(2)
        progress.mark_consumed({"task_a": 1, "task_b": 2})
        progress.finish_consume(2)

        self.assertEqual(progress.next_consumer_step, 3)
        self.assertEqual(progress.consumed_samples, {"task_a": 1, "task_b": 2})

    def test_producer_future_step_advances_independently_from_consumer_step(self):
        # 验证 producer future step 是独立的生产进度，不会被 consumer step 更新隐式推进。
        progress = ProduceProgress.build(["task_a"])

        progress.begin_consume(5)
        progress.finish_consume(5)
        self.assertEqual(progress.producer_future_step, 1)

        progress.advance_future_step()
        self.assertEqual(progress.producer_future_step, 2)

    def test_local_progress_keeps_global_window_untouched(self):
        # 验证共卡 local progress 只表达本次 produce_batch，不污染非共卡 global progress。
        global_progress = ProduceProgress.build(["task_a", "task_b"])
        global_progress.ensure_target_upto(
            batch_size=4,
            future_step=2,
            allocate_batch_sizes=lambda batch_size, step: {"task_a": step, "task_b": batch_size - step},
        )

        local_progress = ProduceProgress.build_local(["task_a", "task_b"], {"task_a": 1, "task_b": 3}, 7)

        self.assertEqual(local_progress.next_consumer_step, 7)
        self.assertEqual(local_progress.producer_future_step, 7)
        self.assertEqual(local_progress.target_samples, {"task_a": 1, "task_b": 3})
        self.assertEqual(global_progress.target_samples, {"task_a": 3, "task_b": 5})

    def test_load_state_dict_updates_existing_dicts_in_place(self):
        # 验证 resume/load 原地更新 dict，避免 strategy 或 context 持有的旧引用失效。
        progress = ProduceProgress.build(["task_a", "task_b"])
        consumed_ref = progress.consumed_samples
        target_ref = progress.target_samples
        raw_sum_ref = progress.raw_rewards_sum
        produced_samples_ref = progress.produced_samples

        progress.load_state_dict(
            {
                "next_consumer_step": 8,
                "producer_future_step": 9,
                "consumed_samples": {"task_a": 4, "task_b": 5},
                "target_samples": {"task_a": 6, "task_b": 7},
                "target_upto_future_step": 10,
                "raw_rewards_sum": {"task_a": 1.25, "task_b": 0.75},
                "raw_rewards_count": {"task_a": 2, "task_b": 1},
                "produced_samples": {"task_a": 3, "task_b": 4},
                "produced_tokens": {"task_a": 30, "task_b": 40},
                "produce_time_s": 2.5,
            }
        )

        self.assertIs(progress.consumed_samples, consumed_ref)
        self.assertIs(progress.target_samples, target_ref)
        self.assertIs(progress.raw_rewards_sum, raw_sum_ref)
        self.assertIs(progress.produced_samples, produced_samples_ref)
        self.assertEqual(progress.state_dict()["target_samples"], {"task_a": 6, "task_b": 7})
        self.assertEqual(progress.raw_rewards_sum, {"task_a": 1.25, "task_b": 0.75})
        self.assertEqual(progress.produced_samples, {"task_a": 3, "task_b": 4})
        self.assertEqual(progress.produce_time_s, 2.5)

    def test_metrics_are_consumed_once_and_reset(self):
        # 验证 producer 侧统计被 trainer 读取一次后清零，避免后续 step 重复上报。
        progress = ProduceProgress.build(["task_a"])
        progress.add_raw_rewards("task_a", 1.25, 2)
        progress.add_produced("task_a", samples=3, tokens=30)
        progress.add_produce_time(0.5)

        self.assertEqual(progress.consume_raw_rewards("task_a"), (1.25, 2))
        self.assertEqual(progress.consume_produced("task_a"), (3, 30))
        self.assertEqual(progress.consume_produce_time(), 0.5)
        self.assertEqual(progress.consume_raw_rewards("task_a"), (0.0, 0))
        self.assertEqual(progress.consume_produced("task_a"), (0, 0))
        self.assertEqual(progress.consume_produce_time(), 0.0)
