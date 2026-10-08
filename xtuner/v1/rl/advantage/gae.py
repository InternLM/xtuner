"""Generalized advantage estimation for PPO.

A trajectory reward is placed on its last controllable action. ``cu_seq_lens``
resets the recursion at trajectory boundaries, and observation tokens are skipped.
"""

from __future__ import annotations

from typing import Any, Sequence

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
) -> torch.Tensor:
    """Compute GAE on controllable actions, resetting at ``cu_seq_lens``."""
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
        for action_idx_tensor in action_indices.flip(0):
            action_idx = int(action_idx_tensor.item())
            delta = flat_rewards[action_idx] + gamma * next_value - flat_values[action_idx]
            advantage = delta + gamma * gae_lambda * next_advantage
            flat_advantages[action_idx] = advantage
            next_value = flat_values[action_idx]
            next_advantage = advantage

    return flat_advantages


def compute_ppo_targets(
    old_values: torch.Tensor,
    reward_scores: torch.Tensor,
    action_mask: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    gae_gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return actor advantages and critic returns from one GAE pass."""
    if old_values.shape != action_mask.shape:
        raise ValueError(
            f"old_values and action_mask must have the same shape, got {old_values.shape} and {action_mask.shape}"
        )
    frozen_values = old_values.detach().float()
    rewards = terminal_rewards(reward_scores, action_mask, cu_seq_lens).to(frozen_values.device)
    advantages = action_gae(
        frozen_values,
        rewards,
        action_mask,
        cu_seq_lens,
        gamma=gae_gamma,
        gae_lambda=gae_lambda,
    )
    critic_returns = torch.where(
        action_mask.to(frozen_values.device).bool(),
        advantages + frozen_values,
        0.0,
    )
    return advantages.detach(), critic_returns.detach()


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


class GAEEstimator(AdvantageEstimator):
    """Token GAE with one discount for actor advantage and critic return."""

    def __init__(
        self,
        gae_gamma: float = 1.0,
        gae_lambda: float = 0.95,
        reward_scope: str = "segment",
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
            "Use compute_gae after the critic value pass."
        )

    def compute_gae(
        self,
        states: Sequence[RolloutState],
        values: Sequence[Sequence[float]],
        rollout_idx: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Return actor advantages and critic returns.

        ``reward_scope="segment"`` treats each sample as its own trajectory.
        ``reward_scope="session"`` concatenates segments that share a session and
        places the reward on the session's last controllable action.
        """
        del rollout_idx
        if len(states) != len(values):
            raise ValueError(f"states and values must have the same length, got {len(states)} and {len(values)}")
        action_masks: list[list[bool]] = []
        for state, sample_values in zip(states, values):
            labels = state.labels or []
            mask = [label != -100 for label in labels[1:]]
            if len(mask) != len(sample_values):
                raise ValueError(f"shifted label length {len(mask)} does not match value length {len(sample_values)}")
            action_masks.append(mask)
        if self.reward_scope == "segment":
            actor_advantages, critic_returns = self._compute_segment_gae(states, values, action_masks)
        elif self.reward_scope == "session":
            actor_advantages, critic_returns = self._compute_session_gae(states, values, action_masks)
        else:
            raise ValueError(f"Unsupported reward_scope {self.reward_scope!r}")
        if self.normalize_actor_advantage:
            actor_advantages = normalize_actor_advantages(actor_advantages, action_masks)
        else:
            actor_advantages = [
                [advantage if keep else 0.0 for advantage, keep in zip(sample_advantages, sample_mask)]
                for sample_advantages, sample_mask in zip(actor_advantages, action_masks)
            ]
        return actor_advantages, critic_returns

    def _compute_segment_gae(
        self,
        states: Sequence[RolloutState],
        values: Sequence[Sequence[float]],
        action_masks: Sequence[list[bool]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Treat each sample as its own trajectory."""
        advantages: list[list[float]] = []
        returns: list[list[float]] = []
        for state, sample_values, sample_mask in zip(states, values, action_masks):
            if len(sample_values) != len(sample_mask):
                raise ValueError(f"values length {len(sample_values)} does not match mask length {len(sample_mask)}")
            if not sample_values:
                advantages.append([])
                returns.append([])
                continue
            reward = state.reward
            score = float(reward["score"]) if reward is not None and "score" in reward else 0.0
            actor_advantages, critic_returns = compute_ppo_targets(
                torch.tensor(sample_values, dtype=torch.float32),
                torch.tensor([score], dtype=torch.float32),
                torch.tensor(sample_mask, dtype=torch.bool),
                torch.tensor([0, len(sample_values)], dtype=torch.int32),
                gae_gamma=self.gae_gamma,
                gae_lambda=self.gae_lambda,
            )
            advantages.append(actor_advantages.reshape(-1).tolist())
            returns.append(critic_returns.reshape(-1).tolist())
        return advantages, returns

    def _compute_session_gae(
        self,
        states: Sequence[RolloutState],
        values: Sequence[Sequence[float]],
        action_masks: Sequence[list[bool]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Concatenate segments that share a session into one trajectory."""
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
                actor_advantages, critic_returns = compute_ppo_targets(
                    torch.tensor(flat_values, dtype=torch.float32),
                    torch.tensor([reward], dtype=torch.float32),
                    torch.tensor(flat_mask, dtype=torch.bool),
                    torch.tensor([0, len(flat_values)], dtype=torch.int32),
                    gae_gamma=self.gae_gamma,
                    gae_lambda=self.gae_lambda,
                )
                flat_advantages = actor_advantages.reshape(-1).tolist()
                flat_returns = critic_returns.reshape(-1).tolist()
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
        filled_advantages: list[list[float]] = []
        filled_returns: list[list[float]] = []
        for maybe_advantages, maybe_returns in zip(advantages, returns):
            if maybe_advantages is None or maybe_returns is None:
                raise RuntimeError("Session GAE did not cover every sample")
            filled_advantages.append(maybe_advantages)
            filled_returns.append(maybe_returns)
        return filled_advantages, filled_returns
