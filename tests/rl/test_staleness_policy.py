"""Staleness 配置与 mask 行为测试。

覆盖整组过期阈值的配置和校验，以及 token 级 staleness mask 的纯函数行为。
"""

import unittest

from pydantic import ValidationError

from xtuner.v1.data_proto.rl_data import RolloutState, calculate_effective_response_mask, reset_rollout_response
from xtuner.v1.rl.agent_loop_manager import (
    AsyncProduceStrategyConfig,
    DisaggAsyncProduceStrategyConfig,
)
from xtuner.v1.rl.agent_loop_manager.produce_utils import calculate_stale_threshold


class TestStalenessPolicy(unittest.TestCase):
    """整组 staleness 阈值、异步策略配置和非法参数校验。"""

    def test_max_staleness_zero_uses_sync_interval_as_threshold(self):
        # max_staleness=0 表示只接受同步间隔内天然存在的最小滞后。
        self.assertEqual(calculate_stale_threshold(max_staleness=0, sync_weights_interval=4), 4)
        strategy = DisaggAsyncProduceStrategyConfig(max_staleness=0).build(sync_weights_interval=4)

        self.assertFalse(strategy.is_model_expired(train_step=8, model_step=4))
        self.assertTrue(strategy.is_model_expired(train_step=9, model_step=4))

    def test_max_staleness_one_allows_one_extra_sync_interval(self):
        # max_staleness=1 表示额外接受一个权重同步周期的滞后。
        self.assertEqual(calculate_stale_threshold(max_staleness=1, sync_weights_interval=4), 8)
        strategy = DisaggAsyncProduceStrategyConfig(
            max_staleness=1,
            enable_partial_rollout=True,
        ).build(sync_weights_interval=4)

        self.assertFalse(strategy.is_model_expired(train_step=12, model_step=4))
        self.assertTrue(strategy.is_model_expired(train_step=13, model_step=4))

    def test_negative_max_staleness_is_invalid(self):
        # Pydantic 配置层必须拒绝负的整组 staleness。
        with self.assertRaises(ValidationError):
            AsyncProduceStrategyConfig(max_staleness=-1)
        with self.assertRaises(ValidationError):
            DisaggAsyncProduceStrategyConfig(max_staleness=-1)

    def test_async_strategies_precompute_token_stale_threshold(self):
        # colocated 和 disaggregated 异步策略应使用相同的 token 阈值换算。
        for config_cls in (
            AsyncProduceStrategyConfig,
            DisaggAsyncProduceStrategyConfig,
        ):
            with self.subTest(config_cls=config_cls.__name__):
                base_kwargs = {"max_staleness": 1, "enable_partial_rollout": True}
                self.assertIsNone(config_cls(**base_kwargs).build(sync_weights_interval=4).token_stale_threshold)
                self.assertEqual(
                    config_cls(max_token_staleness=0, **base_kwargs)
                    .build(sync_weights_interval=4)
                    .token_stale_threshold,
                    4,
                )
                self.assertEqual(
                    config_cls(max_token_staleness=1, **base_kwargs)
                    .build(sync_weights_interval=4)
                    .token_stale_threshold,
                    8,
                )

class TestTokenStalenessMask(unittest.TestCase):
    """Token 级 staleness mask 的阈值与 semantic mask 行为。"""

    def test_token_staleness_threshold_can_be_relaxed(self):
        # token threshold 放宽一个同步周期后，旧周期 token 应从 masked 变为可训练。
        for token_stale_threshold, expected in ((4, [0, 1]), (8, [1, 1])):
            with self.subTest(token_stale_threshold=token_stale_threshold):
                state = self._state(response_model_steps=[0, 4])

                mask = calculate_effective_response_mask(
                    state,
                    current_train_step=5,
                    token_stale_threshold=token_stale_threshold,
                )

                self.assertEqual(mask, expected)
                self.assertIsNone(state.response_mask)

    def test_token_staleness_intersects_semantic_response_mask(self):
        # 最终 mask 必须同时满足 semantic mask 和 token staleness mask。
        state = self._state(response_model_steps=[0, 4], response_mask=[1, 0])

        mask = calculate_effective_response_mask(
            state,
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertEqual(mask, [0, 0])

    def test_rerolled_state_without_semantic_mask_uses_token_staleness_only(self):
        state = reset_rollout_response(self._state(response_model_steps=[0, 4], response_mask=[0, 1]))
        state.response_ids = [3, 4]
        state.response_model_steps = [4, 4]

        mask = calculate_effective_response_mask(
            state,
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertIsNone(state.response_mask)
        self.assertEqual(mask, [1, 1])

    @staticmethod
    def _state(
        *,
        response_model_steps: list[int] | None,
        response_mask: list[int] | None = None,
    ) -> RolloutState:
        return RolloutState(
            rollout_id=1,
            group_id=1,
            message=[{"role": "user", "content": "prompt"}],
            prompt_ids=[1, 2],
            response_ids=[3, 4],
            response_model_steps=response_model_steps,
            response_mask=response_mask,
        )


if __name__ == "__main__":
    unittest.main()
