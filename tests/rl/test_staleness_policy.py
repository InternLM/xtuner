"""Staleness 配置与 mask 行为测试。

覆盖整组过期阈值的配置和校验，以及 token 级 staleness mask 的纯函数行为。
"""

import unittest
from unittest.mock import patch

import ray
from pydantic import ValidationError

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    calculate_group_effective_response_masks,
    discard_rollout_state,
    reset_rollout_response,
)
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

                masks = calculate_group_effective_response_masks(
                    [state],
                    current_train_step=5,
                    token_stale_threshold=token_stale_threshold,
                )

                self.assertEqual(masks, [expected])
                self.assertIsNone(state.response_mask)

    def test_token_staleness_intersects_semantic_response_mask(self):
        # 最终 mask 必须同时满足 semantic mask 和 token staleness mask。
        state = self._state(response_model_steps=[0, 4], response_mask=[1, 0])

        masks = calculate_group_effective_response_masks(
            [state],
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertEqual(masks, [[0, 0]])

    def test_rerolled_state_without_semantic_mask_uses_token_staleness_only(self):
        state = reset_rollout_response(self._state(response_model_steps=[0, 4], response_mask=[0, 1]))
        state.response_ids = [3, 4]
        state.response_model_steps = [4, 4]

        masks = calculate_group_effective_response_masks(
            [state],
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertIsNone(state.response_mask)
        self.assertEqual(masks, [[1, 1]])

    def test_reset_rollout_response_only_clears_fields(self):
        """Reset must not implicitly release an ObjectRef owned by its caller."""
        state = self._state(response_model_steps=[0, 4])
        state.routed_experts = object()
        state.routed_experts_owner = "rollout"

        with patch("xtuner.v1.rl.utils.ray_utils.free_object_refs") as free_refs:
            reset_rollout_response(state)

        free_refs.assert_not_called()
        self.assertIsNone(state.routed_experts)
        self.assertIsNone(state.routed_experts_owner)

    def test_release_owned_routed_experts_only_frees_direct_rollout_refs(self):
        from xtuner.v1.data_proto.rl_data import release_owned_routed_experts

        class FakeObjectRef:
            pass

        direct = self._state(response_model_steps=[0])
        direct_ref = FakeObjectRef()
        direct.routed_experts = direct_ref
        direct.routed_experts_owner = "rollout"
        borrowed = self._state(response_model_steps=[0])
        borrowed.routed_experts = FakeObjectRef()
        borrowed.routed_experts_owner = "trace_store"

        with (
            patch.object(ray, "ObjectRef", FakeObjectRef),
            patch("xtuner.v1.rl.utils.ray_utils.free_object_refs") as free_refs,
        ):
            release_owned_routed_experts(direct)
            release_owned_routed_experts(borrowed)

        free_refs.assert_called_once_with(direct_ref)
        self.assertIsNone(direct.routed_experts)
        self.assertIsNone(direct.routed_experts_owner)
        self.assertIsNotNone(borrowed.routed_experts)
        self.assertEqual(borrowed.routed_experts_owner, "trace_store")

    def test_discard_trace_store_state_detaches_without_freeing_trace_ref(self):
        state = self._state(response_model_steps=[0])
        state.routed_experts = object()
        state.routed_experts_owner = "trace_store"

        with patch("xtuner.v1.rl.utils.ray_utils.free_object_refs") as free_refs:
            discarded = discard_rollout_state(state, release_refs=True)

        free_refs.assert_not_called()
        self.assertIsNone(discarded.routed_experts)
        self.assertIsNone(discarded.routed_experts_owner)

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
