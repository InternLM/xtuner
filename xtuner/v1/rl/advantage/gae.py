"""Generalized advantage estimation for PPO.

A trajectory reward is placed on its last controllable action. ``cu_seq_lens``
resets the recursion at trajectory boundaries, and observation tokens are skipped.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import torch

from xtuner.v1.data_proto.rl_data import RolloutState

from .base import AdvantageEstimator


def terminal_rewards(
    reward_scores: torch.Tensor,
    action_mask: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Place each trajectory reward on its last controllable action."""
    if action_mask.ndim != 1:
        raise ValueError(f"action_mask must have shape [T], got {action_mask.shape}")
    flat_mask = action_mask.bool()
    boundaries = [int(value) for value in cu_seq_lens.detach().cpu().tolist()]
    flat_scores = reward_scores.reshape(-1).to(device=flat_mask.device, dtype=torch.float32)
    if flat_scores.numel() != len(boundaries) - 1:
        raise ValueError(
            f"reward_scores must contain one value per trajectory, got {flat_scores.numel()} for "
            f"{len(boundaries) - 1} trajectories"
        )

    flat_rewards = torch.zeros(flat_mask.shape, dtype=torch.float32, device=flat_mask.device)

    for sample_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        action_indices = torch.nonzero(flat_mask[start:end], as_tuple=False).flatten()
        if action_indices.numel() == 0:
            if flat_scores[sample_idx].item() == 0.0:
                continue
            raise ValueError(f"Trajectory {sample_idx} has no controllable action token but has a non-zero reward.")
        terminal_idx = start + int(action_indices[-1].item())
        flat_rewards[terminal_idx] = flat_scores[sample_idx]
    return flat_rewards


def action_gae(
    old_values: torch.Tensor,
    token_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    length_adaptive_alpha: float | None = None,
) -> torch.Tensor:
    """Compute GAE on controllable actions, resetting at ``cu_seq_lens``.

    When ``length_adaptive_alpha`` is set, the per-trajectory lambda becomes
    ``clamp(1 - 1 / (alpha * action_length), 0, 1)`` instead of ``gae_lambda``.
    """
    if old_values.ndim != 1 or old_values.shape != token_rewards.shape or old_values.shape != action_mask.shape:
        raise ValueError(
            "old_values, token_rewards, and action_mask must have shape [T] and match, got "
            f"{old_values.shape}, {token_rewards.shape}, and {action_mask.shape}"
        )

    flat_values = old_values.detach().float()
    flat_rewards = token_rewards.detach().to(device=flat_values.device, dtype=torch.float32)
    flat_mask = action_mask.to(device=flat_values.device).bool()
    boundaries = [int(value) for value in cu_seq_lens.detach().cpu().tolist()]
    flat_advantages = torch.zeros_like(flat_values, dtype=torch.float32)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        action_indices = torch.nonzero(flat_mask[start:end], as_tuple=False).flatten() + start
        if action_indices.numel() == 0:
            continue
        next_value = torch.zeros((), dtype=torch.float32, device=flat_values.device)
        next_advantage = torch.zeros((), dtype=torch.float32, device=flat_values.device)
        trajectory_lambda = gae_lambda
        if length_adaptive_alpha is not None:
            action_length = int(action_indices.numel())
            trajectory_lambda = max(0.0, min(1.0, 1.0 - 1.0 / (length_adaptive_alpha * action_length)))
        for action_idx_tensor in action_indices.flip(0):
            action_idx = int(action_idx_tensor.item())
            delta = flat_rewards[action_idx] + gamma * next_value - flat_values[action_idx]
            advantage = delta + gamma * trajectory_lambda * next_advantage
            flat_advantages[action_idx] = advantage
            next_value = flat_values[action_idx]
            next_advantage = advantage

    return flat_advantages


def normalize_actor_advantages(
    advantages: Sequence[Sequence[float]],
    masks: Sequence[Sequence[bool]],
    eps: float = 1e-8,
) -> list[list[float]]:
    """Normalize actor advantages with population moments over the full batch.

    The controller already holds every sample, so these moments are the global statistic an all-reduce would produce.
    Critic returns are left unchanged. Masked positions become zero.
    """
    flat_values: list[float] = []
    flat_mask: list[bool] = []
    for sample_advantages, sample_mask in zip(advantages, masks):
        if len(sample_advantages) != len(sample_mask):
            raise ValueError(
                f"advantage length {len(sample_advantages)} does not match mask length {len(sample_mask)}"
            )
        flat_values.extend(float(value) for value in sample_advantages)
        flat_mask.extend(bool(value) for value in sample_mask)
    values = torch.tensor(flat_values, dtype=torch.float32)
    valid_mask = torch.tensor(flat_mask, dtype=torch.bool)
    valid_values = values.masked_select(valid_mask)
    count = int(valid_mask.sum().item())
    if count == 0:
        normalized = torch.zeros_like(values)
    else:
        mean = valid_values.sum() / count
        variance = torch.clamp(valid_values.square().sum() / count - mean.square(), min=0.0)
        normalized = torch.where(valid_mask, (values - mean) / (variance.sqrt() + eps), 0.0)
    cursor = 0
    result: list[list[float]] = []
    for sample_advantages in advantages:
        width = len(sample_advantages)
        result.append(normalized[cursor : cursor + width].tolist())
        cursor += width
    return result


def normalize_with_moments(
    advantages: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    moments: torch.Tensor,
) -> list[torch.Tensor]:
    """Standardize actor advantages with pre-aggregated global moments.

    The worker accumulates per-replica moments and all-reduces them, then hands the
    aggregated ``moments`` here so the normalization math lives with the estimator.
    Critic returns are left unchanged. Masked positions become zero.

    Args:
        advantages (Sequence[torch.Tensor]): Per-sample actor advantage tensors.
        masks (Sequence[torch.Tensor]): Per-sample supervised-token masks with
            shapes matching ``advantages``.
        moments (torch.Tensor): Aggregated tensor of ``[sum, square_sum, count]``
            over kept tokens.

    Returns:
        list[torch.Tensor]: Normalized advantage tensors; masked positions are zero.
    """
    count = moments[2]
    if float(count.item()) <= 0:
        # No supervised token anywhere in the batch: skip normalization and leave the
        # advantages as produced by the GAE pass (matches the previous worker behavior).
        return list(advantages)
    mean = moments[0] / count
    variance = torch.clamp(moments[1] / count - mean.square(), min=0.0)
    scale = (variance.sqrt() + 1e-8).float()
    mean_value = mean.float()
    return [
        torch.where(
            mask,
            (advantage.float() - mean_value) / scale,
            torch.zeros_like(advantage.float()),
        )
        for advantage, mask in zip(advantages, masks)
    ]


class GAEEstimator(AdvantageEstimator):
    """Token GAE with one discount for actor advantage and critic return."""

    def __init__(
        self,
        gae_gamma: float = 1.0,
        gae_lambda: float = 0.95,
        reward_scope: Literal["segment", "session"] = "segment",
        normalize_actor_advantage: bool = True,
    ) -> None:
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.reward_scope = reward_scope
        self.normalize_actor_advantage = normalize_actor_advantage

    def compute(self, rewards: torch.Tensor, group: list[Any]) -> torch.Tensor:
        del rewards, group
        raise NotImplementedError(
            "GAEEstimator.compute does not estimate group-scalar advantages. "
            "Use compute_session_gae after the critic value pass."
        )

    def compute_segment_gae(
        self,
        values: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return actor advantages and critic returns for one packed batch.

        Args:
            values (torch.Tensor): Flattened critic values of shape ``[T]``.
            scores (torch.Tensor): One reward score per packed trajectory.
            mask (torch.Tensor): Flattened action mask of shape ``[T]``.
            cu_seq_lens (torch.Tensor): Cumulative sequence lengths marking
                trajectory boundaries.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Actor advantages and critic
                returns, both of shape ``[T]``.
        """
        frozen_values = values.detach().float()
        rewards = terminal_rewards(scores, mask, cu_seq_lens).to(frozen_values.device)
        advantages = action_gae(
            frozen_values,
            rewards,
            mask,
            cu_seq_lens,
            gamma=self.gae_gamma,
            gae_lambda=self.gae_lambda,
        )
        critic_returns = torch.where(
            mask.to(frozen_values.device).bool(),
            advantages + frozen_values,
            0.0,
        )
        return advantages.detach(), critic_returns.detach()

    def compute_session_gae(
        self,
        states: Sequence[RolloutState],
        values: Sequence[Sequence[float]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Return actor advantages and critic returns over session-grouped
        segments.

        Segments that share a session are concatenated into one trajectory and the
        reward is placed on the session's last controllable action. When
        ``normalize_actor_advantage`` is set, actor advantages are standardized with
        population moments over the full batch; critic returns stay unnormalized.

        Args:
            states (Sequence[RolloutState]): Flat rollout states in training order.
            values (Sequence[Sequence[float]]): One value row per state from the
                critic forward pass.

        Returns:
            tuple[list[list[float]], list[list[float]]]: Actor advantages and critic
                returns, one row per state.
        """
        if len(states) != len(values):
            raise ValueError(f"states and values must have the same length, got {len(states)} and {len(values)}")
        action_masks: list[list[bool]] = []
        for state, sample_values in zip(states, values):
            labels = state.labels or []
            mask = [label != -100 for label in labels[1:]]
            if len(mask) != len(sample_values):
                raise ValueError(f"shifted label length {len(mask)} does not match value length {len(sample_values)}")
            action_masks.append(mask)

        # Concatenate segments that share a session into one trajectory, ordered by
        # ``agent_trace_segment_index``, and place the session reward on its last
        # controllable action.
        advantages: list[list[float] | None] = [None] * len(states)
        returns: list[list[float] | None] = [None] * len(states)
        groups: dict[Any, list[int]] = {}
        for index, state in enumerate(states):
            if state.session_id is not None:
                key: tuple[str, Any] = ("session", state.session_id)
            elif state.rollout_id is not None:
                key = ("rollout", state.rollout_id)
            else:
                key = ("index", index)
            groups.setdefault(key, []).append(index)
        for members in groups.values():
            segment_order: list[tuple[int, int]] = []
            for index in members:
                raw = states[index].extra_fields.get("agent_trace_segment_index")
                segment_index = raw if isinstance(raw, int) else index
                segment_order.append((segment_index, index))
            ordered = [index for _, index in sorted(segment_order)]
            flat_values: list[float] = []
            flat_mask: list[bool] = []
            widths: list[int] = []
            for index in ordered:
                flat_values.extend(float(value) for value in values[index])
                flat_mask.extend(action_masks[index])
                widths.append(len(values[index]))
            reward = 0.0
            for index in reversed(ordered):
                sample_reward = states[index].reward
                if sample_reward is not None and "score" in sample_reward:
                    reward = float(sample_reward["score"])
                    break
            if not flat_values:
                split_advantages: list[list[float]] = [[] for _ in ordered]
                split_returns: list[list[float]] = [[] for _ in ordered]
            else:
                sample_actor_advantages, sample_critic_returns = self.compute_segment_gae(
                    torch.tensor(flat_values, dtype=torch.float32),
                    torch.tensor([reward], dtype=torch.float32),
                    torch.tensor(flat_mask, dtype=torch.bool),
                    torch.tensor([0, len(flat_values)], dtype=torch.int32),
                )
                flat_advantages = sample_actor_advantages.reshape(-1).tolist()
                flat_returns = sample_critic_returns.reshape(-1).tolist()
                split_advantages = []
                split_returns = []
                cursor = 0
                for width in widths:
                    split_advantages.append(flat_advantages[cursor : cursor + width])
                    split_returns.append(flat_returns[cursor : cursor + width])
                    cursor += width
            for index, sample_advantages, sample_returns in zip(ordered, split_advantages, split_returns):
                advantages[index] = sample_advantages
                returns[index] = sample_returns
        actor_advantages: list[list[float]] = []
        critic_returns: list[list[float]] = []
        for maybe_advantages, maybe_returns in zip(advantages, returns):
            if maybe_advantages is None or maybe_returns is None:
                raise RuntimeError("Session GAE did not cover every sample")
            actor_advantages.append(maybe_advantages)
            critic_returns.append(maybe_returns)

        if self.normalize_actor_advantage:
            actor_advantages = normalize_actor_advantages(actor_advantages, action_masks)
        else:
            actor_advantages = [
                [advantage if keep else 0.0 for advantage, keep in zip(sample_advantages, sample_mask)]
                for sample_advantages, sample_mask in zip(actor_advantages, action_masks)
            ]
        return actor_advantages, critic_returns


class DecoupledGAEEstimator(AdvantageEstimator):
    """Token GAE with independent actor and critic discount parameters.

    Decoupled GAE as proposed in `VAPO: Efficient and Reliable Reinforcement Learning
    for Advanced Reasoning Tasks <https://arxiv.org/abs/2504.05118>`_.

    The actor advantage recursion uses ``(actor_gamma, actor_lambda)`` and the critic
    TD(lambda) return recursion uses ``(critic_gamma, critic_lambda)``; defaults follow
    the paper (actor lambda 0.95, critic lambda 1.0). ``actor_length_adaptive_alpha``
    implements the paper's Length-Adaptive GAE, replacing the actor lambda with a
    per-trajectory ``clamp(1 - 1 / (alpha * action_length), 0, 1)``.
    ``actor_positive_scale``/``actor_negative_scale`` rescale positive and non-positive
    actor advantages before normalization. Only segment-scope GAE is supported.
    """

    reward_scope: Literal["segment"] = "segment"

    def __init__(
        self,
        normalize_actor_advantage: bool = True,
        actor_gamma: float = 1.0,
        actor_lambda: float = 0.95,
        critic_gamma: float = 1.0,
        critic_lambda: float = 1.0,
        actor_length_adaptive_alpha: float | None = None,
        actor_positive_scale: float = 1.0,
        actor_negative_scale: float = 1.0,
    ) -> None:
        self.normalize_actor_advantage = normalize_actor_advantage
        self.actor_gamma = actor_gamma
        self.actor_lambda = actor_lambda
        self.critic_gamma = critic_gamma
        self.critic_lambda = critic_lambda
        self.actor_length_adaptive_alpha = actor_length_adaptive_alpha
        self.actor_positive_scale = actor_positive_scale
        self.actor_negative_scale = actor_negative_scale

    def compute(self, rewards: torch.Tensor, group: list[Any]) -> torch.Tensor:
        del rewards, group
        raise NotImplementedError(
            "DecoupledGAEEstimator.compute does not estimate group-scalar advantages. "
            "Use compute_segment_gae after the critic value pass."
        )

    def compute_segment_gae(
        self,
        values: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return decoupled actor advantages and critic returns for one packed
        batch.

        Args:
            values (torch.Tensor): Flattened critic values of shape ``[T]``.
            scores (torch.Tensor): One reward score per packed trajectory.
            mask (torch.Tensor): Flattened action mask of shape ``[T]``.
            cu_seq_lens (torch.Tensor): Cumulative sequence lengths marking
                trajectory boundaries.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Actor advantages and critic
                returns, both of shape ``[T]``.
        """
        frozen_values = values.detach().float()
        rewards = terminal_rewards(scores, mask, cu_seq_lens).to(frozen_values.device)
        actor_advantages = action_gae(
            frozen_values,
            rewards,
            mask,
            cu_seq_lens,
            gamma=self.actor_gamma,
            gae_lambda=self.actor_lambda,
            length_adaptive_alpha=self.actor_length_adaptive_alpha,
        )
        actor_advantages = torch.where(
            actor_advantages > 0,
            actor_advantages * self.actor_positive_scale,
            actor_advantages * self.actor_negative_scale,
        )
        critic_advantages = action_gae(
            frozen_values,
            rewards,
            mask,
            cu_seq_lens,
            gamma=self.critic_gamma,
            gae_lambda=self.critic_lambda,
        )
        critic_returns = torch.where(
            mask.to(frozen_values.device).bool(),
            critic_advantages + frozen_values,
            0.0,
        )
        return actor_advantages.detach(), critic_returns.detach()
