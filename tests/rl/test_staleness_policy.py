import unittest

from pydantic import ValidationError

from xtuner.v1.data_proto import Status
from xtuner.v1.rl.agent_loop_manager import AsyncProduceStrategyConfig, calculate_stale_threshold
from xtuner.v1.rl.agent_loop_manager.producer import expire_group_if_needed


class _State:
    def __init__(self, seq_staleness: int, status: Status = Status.COMPLETED):
        self.uid = "state"
        self.seq_staleness = seq_staleness
        self.status = status


class TestStalenessPolicy(unittest.TestCase):
    def test_max_staleness_zero_uses_sync_interval_as_threshold(self):
        # max_staleness=0 表示只接受同步间隔内天然存在的最小滞后。
        self.assertEqual(calculate_stale_threshold(max_staleness=0, sync_weights_interval=4), 4)
        strategy = AsyncProduceStrategyConfig(max_staleness=0).build(sync_weights_interval=4)

        self.assertFalse(strategy.is_model_expired(train_step=8, model_step=4))
        self.assertTrue(strategy.is_model_expired(train_step=9, model_step=4))

    def test_max_staleness_one_allows_one_extra_sync_interval(self):
        self.assertEqual(calculate_stale_threshold(max_staleness=1, sync_weights_interval=4), 8)
        strategy = AsyncProduceStrategyConfig(max_staleness=1).build(sync_weights_interval=4)

        self.assertFalse(strategy.is_model_expired(train_step=12, model_step=4))
        self.assertTrue(strategy.is_model_expired(train_step=13, model_step=4))

    def test_negative_max_staleness_is_invalid(self):
        with self.assertRaises(ValidationError):
            AsyncProduceStrategyConfig(max_staleness=-1)

    def test_expire_group_requires_positive_step_threshold(self):
        with self.assertRaisesRegex(ValueError, "stale_threshold must be positive"):
            expire_group_if_needed([_State(seq_staleness=0)], stale_threshold=0)

        group = [_State(seq_staleness=4)]
        expire_group_if_needed(group, stale_threshold=4)
        self.assertEqual(group[0].status, Status.EXPIRED)


if __name__ == "__main__":
    unittest.main()
