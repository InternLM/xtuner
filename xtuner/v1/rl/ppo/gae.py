from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PPOTargets:
    """Frozen actor and critic targets computed from one rollout batch.

    Args:
        actor_advantages (torch.Tensor): Action-only GAE using the actor lambda.
        critic_advantages (torch.Tensor): Action-only GAE using the critic lambda.
        critic_returns (torch.Tensor): Critic regression targets.
        token_rewards (torch.Tensor): Sparse rewards with one terminal reward per trajectory.
    """

    actor_advantages: torch.Tensor
    critic_advantages: torch.Tensor
    critic_returns: torch.Tensor
    token_rewards: torch.Tensor


def terminal_rewards(
    reward_scores: torch.Tensor,
    action_mask: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Place every trajectory reward on its final controllable action position.

    Args:
        reward_scores (torch.Tensor): One scalar reward per packed trajectory.
        action_mask (torch.Tensor): Boolean-like token mask with shape ``[T]`` or ``[1, T]``.
        cu_seq_lens (torch.Tensor): Cumulative packed boundaries ``[0, ..., T]``.

    Returns:
        torch.Tensor: Sparse float32 token rewards with the same shape as ``action_mask``.
    """
    flat_mask = _flatten_token_tensor(action_mask, "action_mask").bool()
    boundaries = _validate_boundaries(cu_seq_lens, flat_mask.numel())
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
    return flat_rewards.reshape(action_mask.shape)


def action_gae(
    old_values: torch.Tensor,
    token_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> torch.Tensor:
    """Compute GAE along controllable actions while skipping observation tokens.

    Packed trajectory boundaries reset the recursion. The final action in every
    trajectory is terminal and therefore uses a zero bootstrap value.

    Args:
        old_values (torch.Tensor): Frozen value estimates with shape ``[T]`` or ``[1, T]``.
        token_rewards (torch.Tensor): Token rewards with the same shape as ``old_values``.
        action_mask (torch.Tensor): Boolean-like controllable-action mask.
        cu_seq_lens (torch.Tensor): Cumulative packed boundaries ``[0, ..., T]``.
        gamma (float): Reward discount factor.
        gae_lambda (float): Generalized advantage estimation lambda.

    Returns:
        torch.Tensor: Float32 advantages at action positions and zero elsewhere.
    """
    _validate_discount("gamma", gamma)
    _validate_discount("gae_lambda", gae_lambda)
    if old_values.shape != token_rewards.shape or old_values.shape != action_mask.shape:
        raise ValueError(
            "old_values, token_rewards, and action_mask must have the same shape, got "
            f"{old_values.shape}, {token_rewards.shape}, and {action_mask.shape}"
        )

    flat_values = _flatten_token_tensor(old_values, "old_values").detach().float()
    flat_rewards = (
        _flatten_token_tensor(token_rewards, "token_rewards")
        .detach()
        .to(device=flat_values.device, dtype=torch.float32)
    )
    flat_mask = _flatten_token_tensor(action_mask, "action_mask").to(device=flat_values.device).bool()
    boundaries = _validate_boundaries(cu_seq_lens, flat_values.numel())
    flat_advantages = torch.zeros_like(flat_values, dtype=torch.float32)

    for sample_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
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

    return flat_advantages.reshape(old_values.shape)


def compute_ppo_targets(
    old_values: torch.Tensor,
    reward_scores: torch.Tensor,
    action_mask: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    actor_gamma: float = 1.0,
    actor_lambda: float = 0.95,
    critic_gamma: float = 1.0,
    critic_lambda: float = 1.0,
) -> PPOTargets:
    """Compute decoupled actor GAE and critic returns from terminal rewards.

    Args:
        old_values (torch.Tensor): Frozen value estimates with shape ``[T]`` or ``[1, T]``.
        reward_scores (torch.Tensor): One terminal score per packed trajectory.
        action_mask (torch.Tensor): Boolean-like controllable-action mask.
        cu_seq_lens (torch.Tensor): Cumulative packed boundaries ``[0, ..., T]``.
        actor_gamma (float): Actor GAE discount factor.
        actor_lambda (float): Actor GAE lambda.
        critic_gamma (float): Critic GAE discount factor.
        critic_lambda (float): Critic GAE lambda.

    Returns:
        PPOTargets: Detached float32 actor advantages, critic advantages, returns, and sparse rewards.
    """
    if old_values.shape != action_mask.shape:
        raise ValueError(
            f"old_values and action_mask must have the same shape, got {old_values.shape} and {action_mask.shape}"
        )
    frozen_values = old_values.detach().float()
    rewards = terminal_rewards(reward_scores, action_mask, cu_seq_lens).to(frozen_values.device)
    actor_advantages = action_gae(
        frozen_values,
        rewards,
        action_mask,
        cu_seq_lens,
        gamma=actor_gamma,
        gae_lambda=actor_lambda,
    )
    critic_advantages = action_gae(
        frozen_values,
        rewards,
        action_mask,
        cu_seq_lens,
        gamma=critic_gamma,
        gae_lambda=critic_lambda,
    )
    critic_returns = torch.where(
        action_mask.to(frozen_values.device).bool(),
        critic_advantages + frozen_values,
        0.0,
    )
    return PPOTargets(
        actor_advantages=actor_advantages.detach(),
        critic_advantages=critic_advantages.detach(),
        critic_returns=critic_returns.detach(),
        token_rewards=rewards.detach(),
    )


def _flatten_token_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim not in (1, 2) or (tensor.ndim == 2 and tensor.size(0) != 1):
        raise ValueError(f"{name} must have shape [T] or [1, T], got {tensor.shape}")
    return tensor.reshape(-1)


def _validate_boundaries(cu_seq_lens: torch.Tensor, num_tokens: int) -> list[int]:
    if cu_seq_lens.ndim != 1:
        raise ValueError(f"cu_seq_lens must be one-dimensional, got {cu_seq_lens.shape}")
    boundaries = [int(value) for value in cu_seq_lens.detach().cpu().tolist()]
    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != num_tokens:
        raise ValueError(f"cu_seq_lens must start at 0 and end at {num_tokens}, got {boundaries}")
    if any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError(f"cu_seq_lens boundaries must be strictly increasing, got {boundaries}")
    return boundaries


def _validate_discount(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
