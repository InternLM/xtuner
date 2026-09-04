"""Tests for GAE over packed RL trajectories."""

import pytest
import torch

from xtuner.v1.rl.advantage import (
    GAEAdvantageConfig,
    GAEEstimator,
    TokenLevelAdvantageEstimator,
    action_gae,
    normalize_advantages,
    terminal_token_rewards,
)


def naive_action_gae(
    values: torch.Tensor,
    token_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Straightforward per-token reference implementation.

    Deliberately written as the textbook backward recursion so the vectorized
    implementation can be differentially tested against it.
    """
    flat_values = values.reshape(-1).float()
    flat_rewards = token_rewards.reshape(-1).float()
    flat_mask = action_mask.reshape(-1).bool()
    bounds = [int(b) for b in cu_seq_lens.tolist()]
    advantages = torch.zeros_like(flat_values)

    for start, end in zip(bounds[:-1], bounds[1:]):
        action_indices = [i for i in range(start, end) if bool(flat_mask[i])]
        next_value = 0.0
        next_advantage = 0.0
        for index in reversed(action_indices):
            delta = float(flat_rewards[index]) + gamma * next_value - float(flat_values[index])
            advantage = delta + gamma * gae_lambda * next_advantage
            advantages[index] = advantage
            next_value = float(flat_values[index])
            next_advantage = advantage

    return advantages.reshape(values.shape)


def _packed_batch(
    response_lengths: list[int],
    prompt_length: int = 3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (values, action_mask, cu_seq_lens, reward_scores) for a packed batch."""
    generator = torch.Generator().manual_seed(seed)
    values_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    bounds = [0]
    for response_length in response_lengths:
        total = prompt_length + response_length
        values_parts.append(torch.randn(total, generator=generator))
        mask = torch.zeros(total, dtype=torch.bool)
        mask[prompt_length:] = True
        mask_parts.append(mask)
        bounds.append(bounds[-1] + total)

    values = torch.cat(values_parts).unsqueeze(0)
    action_mask = torch.cat(mask_parts).unsqueeze(0)
    cu_seq_lens = torch.tensor(bounds, dtype=torch.int32)
    reward_scores = torch.randn(len(response_lengths), generator=generator)
    return values, action_mask, cu_seq_lens, reward_scores


class TestTerminalTokenRewards:
    def test_reward_lands_on_last_action_token(self) -> None:
        action_mask = torch.tensor([[False, True, True, False]])
        rewards = terminal_token_rewards(
            reward_scores=torch.tensor([2.5]),
            action_mask=action_mask,
            cu_seq_lens=torch.tensor([0, 4]),
        )
        # Position 2 is the final action token; position 3 is not an action.
        assert rewards.tolist() == [[0.0, 0.0, 2.5, 0.0]]

    def test_each_packed_trajectory_gets_its_own_terminal(self) -> None:
        action_mask = torch.tensor([[False, True, False, True, False, True, False, True]])
        rewards = terminal_token_rewards(
            reward_scores=torch.tensor([1.0, 1.0]),
            action_mask=action_mask,
            cu_seq_lens=torch.tensor([0, 4, 8]),
        )
        # Equal rewards must not merge into one; each trajectory terminates.
        assert rewards.tolist() == [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]

    def test_reward_count_must_match_trajectory_count(self) -> None:
        with pytest.raises(ValueError, match="one value per trajectory"):
            terminal_token_rewards(
                reward_scores=torch.tensor([1.0]),
                action_mask=torch.ones(1, 8, dtype=torch.bool),
                cu_seq_lens=torch.tensor([0, 4, 8]),
            )

    def test_nonzero_reward_without_action_token_raises(self) -> None:
        with pytest.raises(ValueError, match="no controllable action token"):
            terminal_token_rewards(
                reward_scores=torch.tensor([1.0]),
                action_mask=torch.zeros(1, 4, dtype=torch.bool),
                cu_seq_lens=torch.tensor([0, 4]),
            )

    def test_zero_reward_without_action_token_is_allowed(self) -> None:
        rewards = terminal_token_rewards(
            reward_scores=torch.tensor([0.0]),
            action_mask=torch.zeros(1, 4, dtype=torch.bool),
            cu_seq_lens=torch.tensor([0, 4]),
        )
        assert rewards.tolist() == [[0.0, 0.0, 0.0, 0.0]]


class TestActionGAE:
    def test_known_values(self) -> None:
        # Three actions at indices 0, 2, 4; observation tokens at 1 and 3 carry
        # large values that must not enter the recursion.
        values = torch.tensor([[0.1, 7.0, 0.2, 8.0, 0.3]])
        action_mask = torch.tensor([[True, False, True, False, True]])
        token_rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])

        advantages = action_gae(values, token_rewards, action_mask, torch.tensor([0, 5]), gamma=1.0, gae_lambda=1.0)

        # Terminal: 1.0 - 0.3 = 0.7; then 0.2 -> (0.0 + 0.3 - 0.2) + 0.7 = 0.8;
        # then 0.1 -> (0.0 + 0.2 - 0.1) + 0.8 = 0.9.
        torch.testing.assert_close(advantages, torch.tensor([[0.9, 0.0, 0.8, 0.0, 0.7]]))

    def test_observation_tokens_receive_no_advantage(self) -> None:
        values, action_mask, cu_seq_lens, scores = _packed_batch([6, 4])
        token_rewards = terminal_token_rewards(scores, action_mask, cu_seq_lens)
        advantages = action_gae(values, token_rewards, action_mask, cu_seq_lens)
        assert bool((advantages[~action_mask.bool()] == 0).all())

    @pytest.mark.parametrize("gamma", [1.0, 0.99])
    @pytest.mark.parametrize("gae_lambda", [1.0, 0.95, 0.0])
    @pytest.mark.parametrize("response_lengths", [[5], [4, 7], [1, 9, 3], [2, 2, 2, 2]])
    def test_matches_naive_reference(self, gamma: float, gae_lambda: float, response_lengths: list[int]) -> None:
        """The vectorized scan must equal the textbook per-token recursion."""
        values, action_mask, cu_seq_lens, scores = _packed_batch(response_lengths, seed=len(response_lengths))
        token_rewards = terminal_token_rewards(scores, action_mask, cu_seq_lens)

        actual = action_gae(values, token_rewards, action_mask, cu_seq_lens, gamma=gamma, gae_lambda=gae_lambda)
        expected = naive_action_gae(
            values, token_rewards, action_mask, cu_seq_lens, gamma=gamma, gae_lambda=gae_lambda
        )

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_matches_naive_reference_with_interleaved_observations(self) -> None:
        """Agentic trajectories interleave observation tokens mid-response."""
        torch.manual_seed(3)
        values = torch.randn(1, 40)
        action_mask = torch.zeros(1, 40, dtype=torch.bool)
        # Two trajectories, each: prompt, actions, observation, actions.
        action_mask[0, 3:10] = True
        action_mask[0, 13:20] = True
        action_mask[0, 23:28] = True
        action_mask[0, 32:40] = True
        cu_seq_lens = torch.tensor([0, 20, 40])
        token_rewards = terminal_token_rewards(torch.tensor([1.0, -1.0]), action_mask, cu_seq_lens)

        actual = action_gae(values, token_rewards, action_mask, cu_seq_lens, gamma=1.0, gae_lambda=0.95)
        expected = naive_action_gae(values, token_rewards, action_mask, cu_seq_lens, 1.0, 0.95)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_terminal_action_bootstraps_from_zero(self) -> None:
        # A single action whose value is 0.4 and reward 1.0 has advantage
        # 1.0 - 0.4, i.e. no bootstrap from a following state.
        values = torch.tensor([[0.4]])
        advantages = action_gae(
            values,
            torch.tensor([[1.0]]),
            torch.tensor([[True]]),
            torch.tensor([0, 1]),
        )
        torch.testing.assert_close(advantages, torch.tensor([[0.6]]))

    def test_trajectory_boundaries_reset_the_recursion(self) -> None:
        """Rewards must not leak across packed trajectory boundaries."""
        values = torch.zeros(1, 4)
        action_mask = torch.ones(1, 4, dtype=torch.bool)
        # Only the second trajectory is rewarded.
        token_rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

        advantages = action_gae(values, token_rewards, action_mask, torch.tensor([0, 2, 4]), gamma=1.0, gae_lambda=1.0)

        assert advantages[0, 0].item() == 0.0
        assert advantages[0, 1].item() == 0.0
        assert advantages[0, 2].item() == 1.0
        assert advantages[0, 3].item() == 1.0

    def test_no_action_tokens_yields_zeros(self) -> None:
        advantages = action_gae(
            torch.randn(1, 6),
            torch.zeros(1, 6),
            torch.zeros(1, 6, dtype=torch.bool),
            torch.tensor([0, 6]),
        )
        torch.testing.assert_close(advantages, torch.zeros(1, 6))

    def test_accepts_one_dimensional_input(self) -> None:
        advantages = action_gae(
            torch.tensor([0.4]),
            torch.tensor([1.0]),
            torch.tensor([True]),
            torch.tensor([0, 1]),
        )
        assert advantages.shape == (1,)

    def test_long_sequence_stays_finite(self) -> None:
        """lambda**t underflows for long responses; the scan must not.

        A closed-form (gamma*lambda)**t weighting would underflow to zero here,
        silently zeroing early advantages.
        """
        length = 8192
        values = torch.zeros(1, length)
        action_mask = torch.ones(1, length, dtype=torch.bool)
        token_rewards = torch.zeros(1, length)
        token_rewards[0, -1] = 1.0

        advantages = action_gae(
            values, token_rewards, action_mask, torch.tensor([0, length]), gamma=1.0, gae_lambda=0.95
        )

        assert bool(torch.isfinite(advantages).all())
        # With gamma=1 the terminal advantage is exactly the reward.
        torch.testing.assert_close(advantages[0, -1], torch.tensor(1.0))

    @pytest.mark.parametrize(("name", "value"), [("gamma", -0.1), ("gamma", 1.1), ("gae_lambda", 2.0)])
    def test_out_of_range_discounts_raise(self, name: str, value: float) -> None:
        with pytest.raises(ValueError, match=f"{name} must be in"):
            action_gae(
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([0, 2]),
                **{name: value},
            )

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            action_gae(
                torch.zeros(1, 4),
                torch.zeros(1, 3),
                torch.ones(1, 4, dtype=torch.bool),
                torch.tensor([0, 4]),
            )

    @pytest.mark.parametrize(
        "cu_seq_lens",
        [torch.tensor([1, 4]), torch.tensor([0, 3]), torch.tensor([0, 2, 2, 4]), torch.tensor([0])],
    )
    def test_invalid_boundaries_raise(self, cu_seq_lens: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="cu_seq_lens"):
            action_gae(
                torch.zeros(1, 4),
                torch.zeros(1, 4),
                torch.ones(1, 4, dtype=torch.bool),
                cu_seq_lens,
            )


class TestGAEEstimator:
    def test_returns_equal_advantage_plus_value_on_actions(self) -> None:
        values, action_mask, cu_seq_lens, scores = _packed_batch([5, 3])
        token_rewards = terminal_token_rewards(scores, action_mask, cu_seq_lens)
        estimator = GAEEstimator(gamma=1.0, gae_lambda=0.95)

        advantages, returns = estimator.compute(values, token_rewards, action_mask, cu_seq_lens)

        mask = action_mask.bool()
        torch.testing.assert_close(returns[mask], (advantages + values.float())[mask])
        assert bool((returns[~mask] == 0).all())

    def test_lambda_one_returns_are_reward_to_go(self) -> None:
        """With gamma=lambda=1 the return target is the undiscounted reward-to-go."""
        values = torch.tensor([[0.1, 0.2, 0.3]])
        action_mask = torch.ones(1, 3, dtype=torch.bool)
        token_rewards = torch.tensor([[0.0, 0.0, 1.0]])

        _, returns = GAEEstimator(gamma=1.0, gae_lambda=1.0).compute(
            values, token_rewards, action_mask, torch.tensor([0, 3])
        )

        torch.testing.assert_close(returns, torch.ones(1, 3))

    def test_outputs_are_detached(self) -> None:
        values = torch.randn(1, 4, requires_grad=True)
        advantages, returns = GAEEstimator().compute(
            values,
            torch.zeros(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            torch.tensor([0, 4]),
        )
        assert not advantages.requires_grad
        assert not returns.requires_grad

    def test_is_token_level_estimator(self) -> None:
        # The trainer dispatches on this to decide whether to build a critic.
        assert isinstance(GAEEstimator(), TokenLevelAdvantageEstimator)

    def test_config_builds_estimator(self) -> None:
        estimator = GAEAdvantageConfig(gamma=0.99, gae_lambda=0.9).build()
        assert isinstance(estimator, GAEEstimator)
        assert estimator.gamma == 0.99
        assert estimator.gae_lambda == 0.9

    def test_config_rejects_out_of_range_discounts(self) -> None:
        with pytest.raises(ValueError):
            GAEAdvantageConfig(gamma=1.5).build()


class TestNormalizeAdvantages:
    def test_masked_positions_stay_zero(self) -> None:
        advantages = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[True, True, False, False]])
        normalized = normalize_advantages(advantages, mask)
        assert normalized[0, 2].item() == 0.0
        assert normalized[0, 3].item() == 0.0

    def test_statistics_use_only_masked_in_tokens(self) -> None:
        advantages = torch.tensor([[1.0, 3.0, 1000.0]])
        mask = torch.tensor([[True, True, False]])
        normalized = normalize_advantages(advantages, mask)
        # Mean 2, population std 1 -> [-1, +1]; the outlier must not shift them.
        torch.testing.assert_close(normalized[0, :2], torch.tensor([-1.0, 1.0]), atol=1e-5, rtol=1e-4)

    def test_zero_mean_unit_variance(self) -> None:
        torch.manual_seed(0)
        advantages = torch.randn(1, 256) * 5 + 3
        mask = torch.ones(1, 256, dtype=torch.bool)
        normalized = normalize_advantages(advantages, mask)
        assert abs(float(normalized.mean())) < 1e-5
        assert abs(float(normalized.std(unbiased=False)) - 1.0) < 1e-4

    def test_empty_mask_returns_zeros(self) -> None:
        normalized = normalize_advantages(torch.randn(1, 4), torch.zeros(1, 4, dtype=torch.bool))
        torch.testing.assert_close(normalized, torch.zeros(1, 4))

    def test_constant_advantages_do_not_produce_nan(self) -> None:
        normalized = normalize_advantages(torch.full((1, 8), 2.0), torch.ones(1, 8, dtype=torch.bool))
        assert bool(torch.isfinite(normalized).all())

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            normalize_advantages(torch.zeros(1, 4), torch.ones(1, 3, dtype=torch.bool))
