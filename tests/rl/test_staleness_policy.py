"""Staleness 配置与 mask 行为测试。

覆盖整组过期阈值的配置和校验，以及 token 级 staleness 往 labels 烙制的行为。
"""

import unittest

from pydantic import ValidationError

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    calculate_group_effective_response_masks,
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
    """Token 级 staleness mask 的阈值与语义监督行为（labels 尾段 / agentic 全序列两种形态）。"""

    def test_token_staleness_threshold_can_be_relaxed(self):
        # token threshold 放宽一个同步周期后，旧周期 token 应从 masked 变为可训练。
        for token_stale_threshold, expected_mask, expected_labels in (
            (4, [0, 1], [-100, -100, -100, 4]),
            (8, [1, 1], [-100, -100, 3, 4]),
        ):
            with self.subTest(token_stale_threshold=token_stale_threshold):
                state = self._state(response_model_steps=[0, 4])

                masks = calculate_group_effective_response_masks(
                    [state],
                    current_train_step=5,
                    token_stale_threshold=token_stale_threshold,
                )

                self.assertEqual(masks, [expected_mask])
                # 过期 token 的 response 段 label 被烙成 -100；全新鲜则保持原 labels。
                self.assertEqual(state.labels, expected_labels)

    def test_repeated_calls_converge(self):
        # staleness 只增不减：先在旧 step 烙一次，再在新 step 重算，最终 labels
        # 与一次性按新 step 烙制的结果完全一致（replay buffer 逐轮检查同理）。
        incremental = self._state(response_model_steps=[0, 4])
        calculate_group_effective_response_masks([incremental], current_train_step=5, token_stale_threshold=4)
        calculate_group_effective_response_masks([incremental], current_train_step=9, token_stale_threshold=4)

        oneshot = self._state(response_model_steps=[0, 4])
        calculate_group_effective_response_masks([oneshot], current_train_step=9, token_stale_threshold=4)

        self.assertEqual(incremental.labels, oneshot.labels)
        self.assertEqual(oneshot.labels, [-100, -100, -100, -100])

    def test_token_staleness_intersects_semantic_labels(self):
        # 最终有效掩码必须同时反映语义监督（labels 尾段 -100 位）与 token staleness。
        state = self._state(response_model_steps=[0, 4], labels=[-100, -100, 3, -100])

        masks = calculate_group_effective_response_masks(
            [state],
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertEqual(masks, [[0, 0]])
        self.assertEqual(state.labels, [-100, -100, -100, -100])

    def test_rerolled_state_uses_token_staleness_only(self):
        # 重 roll 后 canonicalize 重建训练字段：response 段全部受监督时，
        # 有效掩码只由 token staleness 决定。
        state = reset_rollout_response(self._state(response_model_steps=[0, 4]))
        state.response_ids = [3, 4]
        state.response_model_steps = [4, 4]
        state.labels = [-100, -100, 3, 4]

        masks = calculate_group_effective_response_masks(
            [state],
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertEqual(masks, [[1, 1]])
        self.assertEqual(state.labels, [-100, -100, 3, 4])

    def test_reset_clears_canonical_train_fields(self):
        # 回归：reset 必须清掉上一轮 canonicalize 写入的 input_ids/labels，
        # 否则重 roll 后基类 canonicalize 跳过重建，训练字段与新生成的 response 错位。
        state = self._state(response_model_steps=[0, 4])
        state.input_ids = [1, 2, 3, 4]

        reset_rollout_response(state)

        self.assertIsNone(state.input_ids)
        self.assertIsNone(state.labels)

    def test_full_sequence_agentic_stale_clears_all_supervised_labels(self):
        # agentic 全序列形态：response_ids 是 input_ids 去掉 prompt 前缀的连续后缀（含环境
        # token）。steps 由 replay buffer 回填为单一 model_step 时全部过期，全部监督位清零。
        state = RolloutState(
            rollout_id=1,
            group_id=1,
            message=[{"role": "user", "content": "prompt"}],
            prompt_ids=[1, 2],
            response_ids=[30, -1, 40, -1, 32],
            response_model_steps=[0, 0, 0, 0, 0],
            input_ids=[1, 2, 30, -1, 40, -1, 32],
            labels=[-100, -100, 30, -100, 40, -100, 32],
        )

        masks = calculate_group_effective_response_masks(
            [state],
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertEqual(masks, [[0, 0, 0, 0, 0]])
        self.assertEqual(state.labels, [-100] * 7)

    def test_full_sequence_agentic_per_token_stale_clears_only_stale_tokens(self):
        # agentic 全序列形态同样按 token 级 staleness 判定：仅过期 token 的监督位被清零，
        # 新鲜 token 原样保留，环境 token 的洞位（semantic=0）不受 staleness 影响。
        state = RolloutState(
            rollout_id=1,
            group_id=1,
            message=[{"role": "user", "content": "prompt"}],
            prompt_ids=[1, 2],
            response_ids=[30, -1, 40, -1, 32],
            response_model_steps=[0, 4, 4, 4, 4],
            input_ids=[1, 2, 30, -1, 40, -1, 32],
            labels=[-100, -100, 30, -100, 40, -100, 32],
        )

        masks = calculate_group_effective_response_masks(
            [state],
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertEqual(masks, [[0, 0, 1, 0, 1]])
        self.assertEqual(state.labels, [-100, -100, -100, -100, 40, -100, 32])

    def test_full_sequence_agentic_fresh_keeps_labels(self):
        # agentic 全序列形态且 token 新鲜：labels 保持不变，掩码即语义掩码（不会触发过期）。
        state = RolloutState(
            rollout_id=1,
            group_id=1,
            message=[{"role": "user", "content": "prompt"}],
            prompt_ids=[1, 2],
            response_ids=[30, -1, 40, -1, 32],
            response_model_steps=[4, 4, 4, 4, 4],
            input_ids=[1, 2, 30, -1, 40, -1, 32],
            labels=[-100, -100, 30, -100, 40, -100, 32],
        )

        masks = calculate_group_effective_response_masks(
            [state],
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertEqual(masks, [[1, 0, 1, 0, 1]])
        self.assertEqual(state.labels, [-100, -100, 30, -100, 40, -100, 32])

    def test_zero_prompt_suffix_state_is_eligible(self):
        # response_ids 允许覆盖整个序列（prompt 前缀长度为 0）：len(labels) ==
        # len(response_ids) 的后缀形态必须仍参与 staleness 烙制，不被 guard 跳过。
        state = RolloutState(
            rollout_id=1,
            group_id=1,
            message=[{"role": "user", "content": "prompt"}],
            response_ids=[3, 4],
            response_model_steps=[0, 4],
            labels=[3, 4],
        )

        masks = calculate_group_effective_response_masks(
            [state],
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertEqual(masks, [[0, 1]])
        self.assertEqual(state.labels, [-100, 4])

    def test_canonicalized_prompt_response_state_is_still_eligible(self):
        # canonicalize 后的 prompt+response 样本（reasoning loop 产出）仍参与 staleness。
        canonical = self._state(response_model_steps=[0, 4], labels=[-100, -100, 3, -100])
        canonical.input_ids = [1, 2, 3, 4]
        canonical.logprobs = [0.0, 0.0, -0.1, -0.2]

        masks = calculate_group_effective_response_masks(
            [canonical],
            current_train_step=5,
            token_stale_threshold=4,
        )

        self.assertEqual(masks, [[0, 0]])
        self.assertEqual(canonical.labels, [-100, -100, -100, -100])

    @staticmethod
    def _state(
        *,
        response_model_steps: list[int] | None,
        labels: list[int] | None = None,
    ) -> RolloutState:
        state = RolloutState(
            rollout_id=1,
            group_id=1,
            message=[{"role": "user", "content": "prompt"}],
            prompt_ids=[1, 2],
            response_ids=[3, 4],
            response_model_steps=response_model_steps,
        )
        # prompt+response 形态：response_ids 是序列的连续尾段，labels 为全序列监督。
        state.labels = labels if labels is not None else [-100, -100, 3, 4]
        return state


if __name__ == "__main__":
    unittest.main()
