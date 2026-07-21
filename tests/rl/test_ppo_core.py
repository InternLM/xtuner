import pytest
import torch

from xtuner.v1.rl.ppo import (
    action_gae,
    align_next_token_data,
    align_single_turn,
    build_group_loss_masks,
    compute_ppo_targets,
    deterministic_truncated_keep_mask,
    normalize_advantages,
    terminal_rewards,
)
from xtuner.v1.rl.trainer.ppo_config import PPOConfig
from xtuner.v1.rl.trainer.worker import calculate_actor_global_flat_ev


class TestNextTokenAlignment:
    def test_input_labels_align_actions_to_predictor_positions(self) -> None:
        result = align_next_token_data(
            input_ids=torch.tensor([10, 11, 20, 21]),
            labels=torch.tensor([-100, -100, 20, -100]),
        )

        assert result.model_input_ids.tolist() == [10, 11, 20]
        assert result.shifted_labels.tolist() == [-100, 20, -100]
        assert result.action_mask.tolist() == [False, True, False]

    def test_single_turn_first_action_is_predicted_by_last_prompt_token(self) -> None:
        result = align_single_turn(
            prompt_ids=torch.tensor([10, 11, 12]),
            response_ids=torch.tensor([20, 21, 22]),
            response_mask=torch.tensor([1, 0, 1]),
        )

        assert result.model_input_ids.tolist() == [10, 11, 12, 20, 21]
        assert result.shifted_labels.tolist() == [-100, -100, 20, -100, 22]
        assert result.action_mask.tolist() == [False, False, True, False, True]
        assert result.model_input_ids[2].item() == 12


class TestActionOnlyGAE:
    def test_terminal_rewards_do_not_merge_adjacent_equal_rewards(self) -> None:
        action_mask = torch.tensor([[False, True, False, True, False, True, False, True]])
        rewards = terminal_rewards(
            reward_scores=torch.tensor([1.0, 1.0]),
            action_mask=action_mask,
            cu_seq_lens=torch.tensor([0, 4, 8]),
        )

        assert rewards.tolist() == [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]

    def test_actor_and_critic_use_different_lambdas_and_skip_observation(self) -> None:
        values = torch.tensor([[0.1, 7.0, 0.2, 8.0, 0.3]])
        action_mask = torch.tensor([[True, False, True, False, True]])
        targets = compute_ppo_targets(
            old_values=values,
            reward_scores=torch.tensor([1.0]),
            action_mask=action_mask,
            cu_seq_lens=torch.tensor([0, 5]),
            actor_lambda=0.95,
            critic_lambda=1.0,
        )

        torch.testing.assert_close(
            targets.actor_advantages,
            torch.tensor([[0.82675, 0.0, 0.765, 0.0, 0.7]]),
        )
        torch.testing.assert_close(
            targets.critic_advantages,
            torch.tensor([[0.9, 0.0, 0.8, 0.0, 0.7]]),
        )
        torch.testing.assert_close(
            targets.critic_returns,
            torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]]),
        )

    @pytest.mark.parametrize(
        ("action_count", "alpha"),
        [(3, 0.05), (2, 0.5)],
    )
    def test_length_adaptive_lambda_clamps_at_zero(self, action_count: int, alpha: float) -> None:
        values = torch.zeros(1, action_count)
        rewards = torch.zeros_like(values)
        rewards[0, -1] = 1.0
        advantages = action_gae(
            old_values=values,
            token_rewards=rewards,
            action_mask=torch.ones_like(values, dtype=torch.bool),
            cu_seq_lens=torch.tensor([0, action_count]),
            gae_lambda=0.95,
            length_adaptive_alpha=alpha,
        )
        expected = torch.zeros_like(values)
        expected[0, -1] = 1.0

        torch.testing.assert_close(advantages, expected)

    def test_length_adaptive_lambda_is_per_trajectory_and_action_only(self) -> None:
        values = torch.zeros(1, 10)
        action_mask = torch.tensor(
            [[True, False, True, True, False, True, False, True, False, True]]
        )
        boundaries = torch.tensor([0, 3, 10])
        rewards = terminal_rewards(torch.tensor([1.0, 1.0]), action_mask, boundaries)
        advantages = action_gae(
            old_values=values,
            token_rewards=rewards,
            action_mask=action_mask,
            cu_seq_lens=boundaries,
            gae_lambda=0.95,
            length_adaptive_alpha=0.4,
        )

        torch.testing.assert_close(
            advantages,
            torch.tensor([[0.0, 0.0, 1.0, 0.052734375, 0.0, 0.140625, 0.0, 0.375, 0.0, 1.0]]),
        )

    def test_length_adaptive_lambda_changes_actor_only(self) -> None:
        values = torch.zeros(1, 5)
        action_mask = torch.tensor([[True, False, True, False, True]])
        targets = compute_ppo_targets(
            old_values=values,
            reward_scores=torch.tensor([1.0]),
            action_mask=action_mask,
            cu_seq_lens=torch.tensor([0, 5]),
            actor_lambda=0.95,
            critic_lambda=1.0,
            actor_length_adaptive_alpha=0.4,
        )

        torch.testing.assert_close(
            targets.actor_advantages,
            torch.tensor([[1.0 / 36.0, 0.0, 1.0 / 6.0, 0.0, 1.0]]),
        )
        torch.testing.assert_close(
            targets.critic_advantages,
            torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]]),
        )
        torch.testing.assert_close(
            targets.critic_returns,
            torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]]),
        )

    @pytest.mark.parametrize("alpha", [0.0, -0.1, float("nan"), float("inf")])
    def test_action_gae_rejects_invalid_length_adaptive_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            action_gae(
                old_values=torch.zeros(1, 1),
                token_rewards=torch.zeros(1, 1),
                action_mask=torch.ones(1, 1, dtype=torch.bool),
                cu_seq_lens=torch.tensor([0, 1]),
                length_adaptive_alpha=alpha,
            )

    def test_padding_trajectory_without_actions_is_skipped(self) -> None:
        values = torch.tensor([[0.1, 9.0, 0.2, 8.0, 7.0]])
        action_mask = torch.tensor([[True, False, True, False, False]])
        rewards = terminal_rewards(
            reward_scores=torch.tensor([1.0, 0.0]),
            action_mask=action_mask,
            cu_seq_lens=torch.tensor([0, 3, 5]),
        )
        advantages = action_gae(
            old_values=values,
            token_rewards=rewards,
            action_mask=action_mask,
            cu_seq_lens=torch.tensor([0, 3, 5]),
            length_adaptive_alpha=0.05,
        )

        assert rewards.tolist() == [[0.0, 0.0, 1.0, 0.0, 0.0]]
        torch.testing.assert_close(advantages[:, 3:], torch.zeros(1, 2))

    def test_padding_trajectory_rejects_nonzero_reward(self) -> None:
        with pytest.raises(ValueError, match="non-zero reward"):
            terminal_rewards(
                reward_scores=torch.tensor([1.0]),
                action_mask=torch.tensor([[False, False]]),
                cu_seq_lens=torch.tensor([0, 2]),
            )


class TestPPOMasks:
    def test_uniform_reward_group_trains_only_critic(self) -> None:
        action_masks = [torch.tensor([False, True, True]), torch.tensor([True, False])]
        masks = build_group_loss_masks(action_masks, rewards=[1.0, 1.0])

        assert masks.is_uniform
        assert [mask.tolist() for mask in masks.actor] == [
            [False, False, False],
            [False, False],
        ]
        assert [mask.tolist() for mask in masks.critic] == [
            [False, True, True],
            [True, False],
        ]

    def test_uniform_reward_group_can_train_actor(self) -> None:
        action_masks = [torch.tensor([False, True, True]), torch.tensor([True, False])]
        masks = build_group_loss_masks(
            action_masks,
            rewards=[1.0, 1.0],
            train_actor_on_uniform_groups=True,
        )

        assert masks.is_uniform
        assert [mask.tolist() for mask in masks.actor] == [
            [False, True, True],
            [True, False],
        ]
        assert [mask.tolist() for mask in masks.critic] == [
            [False, True, True],
            [True, False],
        ]

    def test_uniform_reward_group_can_be_dropped_from_critic(self) -> None:
        action_masks = [torch.tensor([False, True]), torch.tensor([True, False])]
        masks = build_group_loss_masks(
            action_masks,
            rewards=[1.0, 1.0],
            keep_uniform_groups=False,
        )

        assert masks.is_uniform
        assert masks.critic_is_uniform
        assert not any(mask.any() for mask in masks.actor)
        assert not any(mask.any() for mask in masks.critic)

    def test_actor_and_critic_truncation_eligibility_are_independent(self) -> None:
        action_masks = [torch.tensor([True]), torch.tensor([True]), torch.tensor([True])]
        masks = build_group_loss_masks(
            action_masks,
            rewards=[0.0, 1.0, 0.0],
            sample_eligible=[True, True, False],
            critic_sample_eligible=[True, True, True],
        )

        assert masks.sample_eligible == (True, True, False)
        assert masks.critic_sample_eligible == (True, True, True)
        assert [mask.item() for mask in masks.actor] == [True, True, False]
        assert [mask.item() for mask in masks.critic] == [True, True, True]

    def test_deterministic_selection_keeps_one_length_trajectory(self) -> None:
        reasons = ["length", "stop", "length", "length"]
        rollout_ids = ["a", "normal", "b", "c"]
        first = deterministic_truncated_keep_mask(
            reasons,
            rollout_ids,
            selection_seed=17,
            step=4,
            group_id="group-1",
        )
        second = deterministic_truncated_keep_mask(
            reasons,
            rollout_ids,
            selection_seed=17,
            step=4,
            group_id="group-1",
        )

        assert first == second
        assert first[1]
        assert sum(first[index] for index in (0, 2, 3)) == 1

    @pytest.mark.parametrize(
        ("limit", "expected_retained"),
        [(None, 3), (0, 0), (1, 1), (2, 2)],
    )
    def test_truncated_limit_is_configurable(
        self,
        limit: int | None,
        expected_retained: int,
    ) -> None:
        mask = deterministic_truncated_keep_mask(
            ["length", "stop", "length", "length"],
            ["a", "normal", "b", "c"],
            max_truncated_per_group=limit,
        )

        assert mask[1]
        assert sum(mask[index] for index in (0, 2, 3)) == expected_retained

    def test_negative_truncated_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            deterministic_truncated_keep_mask(
                ["length"],
                ["a"],
                max_truncated_per_group=-1,
            )


class TestPPOConfig:
    def test_critic_selection_defaults_keep_uniform_and_all_truncated(self) -> None:
        config = PPOConfig()

        assert config.keep_uniform_groups is True
        assert config.train_actor_on_uniform_groups is False
        assert config.max_truncated_per_group is None

    def test_negative_truncated_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            PPOConfig(max_truncated_per_group=-1)

    @pytest.mark.parametrize("alpha", [0.0, -0.1, float("nan"), float("inf")])
    def test_invalid_length_adaptive_alpha_is_rejected(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            PPOConfig(actor_length_adaptive_alpha=alpha)


class TestAdvantageNormalization:
    def test_local_masked_population_normalization(self) -> None:
        normalized = normalize_advantages(
            torch.tensor([1.0, 2.0, 3.0, 100.0]),
            torch.tensor([True, True, True, False]),
            distributed=False,
        )

        torch.testing.assert_close(
            normalized,
            torch.tensor([-1.2247449, 0.0, 1.2247449, 0.0]),
        )


class TestActorGlobalFlatEV:
    def test_uses_raw_actor_advantages_and_masked_lambda_returns(self) -> None:
        explained_variance = calculate_actor_global_flat_ev(
            [torch.tensor([0.0, 1.0, 2.0, 100.0])],
            [torch.tensor([0.0, 1.0, 2.0, -100.0])],
            [torch.tensor([True, True, True, False])],
            distributed=False,
        )

        assert explained_variance == pytest.approx(0.75)

    @pytest.mark.parametrize(
        ("advantages", "old_values", "mask"),
        [
            ([1.0], [0.0], [True]),
            ([0.0, 0.0], [1.0, 1.0], [True, True]),
            ([1.0, 2.0], [0.0, 0.0], [False, False]),
        ],
    )
    def test_returns_none_when_explained_variance_is_undefined(
        self,
        advantages: list[float],
        old_values: list[float],
        mask: list[bool],
    ) -> None:
        explained_variance = calculate_actor_global_flat_ev(
            [torch.tensor(advantages)],
            [torch.tensor(old_values)],
            [torch.tensor(mask)],
            distributed=False,
        )

        assert explained_variance is None
