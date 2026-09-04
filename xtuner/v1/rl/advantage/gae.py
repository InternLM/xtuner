"""Generalized Advantage Estimation over packed RL trajectories."""

import torch

from xtuner.v1.rl.advantage.base import TokenLevelAdvantageEstimator


def terminal_token_rewards(
    reward_scores: torch.Tensor,
    action_mask: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Place each trajectory's scalar reward on its final action token.

    PPO treats a completion as an episode whose reward arrives at termination,
    so the sequence-level score becomes a single non-zero token reward. GAE then
    propagates it backwards.

    Args:
        reward_scores (torch.Tensor): One scalar reward per packed trajectory.
        action_mask (torch.Tensor): Boolean-like mask shaped ``[T]`` or ``[1, T]``.
        cu_seq_lens (torch.Tensor): Cumulative packed boundaries ``[0, ..., T]``.

    Returns:
        torch.Tensor: Sparse float32 token rewards shaped like ``action_mask``.

    Raises:
        ValueError: If the reward count does not match the trajectory count, or
            a trajectory carries a non-zero reward but has no action token.
    """
    flat_mask = _flatten(action_mask, "action_mask").bool()
    bounds = _validate_bounds(cu_seq_lens, flat_mask.numel())
    scores = reward_scores.reshape(-1).to(device=flat_mask.device, dtype=torch.float32)
    num_traj = bounds.numel() - 1
    if scores.numel() != num_traj:
        raise ValueError(
            f"reward_scores must hold one value per trajectory, got {scores.numel()} for {num_traj} trajectories"
        )

    flat_rewards = torch.zeros(flat_mask.shape, dtype=torch.float32, device=flat_mask.device)
    action_idx, segment = _action_index(flat_mask, bounds)
    if action_idx.numel() > 0:
        # The last action index within each trajectory. `scatter_reduce` with
        # amax picks it without a Python loop over trajectories.
        terminal = torch.full((num_traj,), -1, dtype=torch.long, device=flat_mask.device)
        terminal.scatter_reduce_(0, segment, action_idx, reduce="amax", include_self=True)
    else:
        terminal = torch.full((num_traj,), -1, dtype=torch.long, device=flat_mask.device)

    empty = terminal < 0
    if bool(empty.any()) and bool((scores[empty] != 0).any()):
        bad = int(torch.nonzero(empty & (scores != 0), as_tuple=False).flatten()[0].item())
        raise ValueError(f"Trajectory {bad} has a non-zero reward but no controllable action token.")

    keep = ~empty
    flat_rewards[terminal[keep]] = scores[keep]
    return flat_rewards.reshape(action_mask.shape)


def action_gae(
    values: torch.Tensor,
    token_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> torch.Tensor:
    """Compute GAE along action tokens, skipping observation tokens.

    The recursion runs over the tokens the policy controls, so interleaved
    observation tokens (tool output, retrieved context) neither receive an
    advantage nor discount the chain. Packed boundaries reset the recursion, and
    each trajectory's last action is terminal, bootstrapping from zero.

    The scan is vectorized across trajectories: actions are gathered into a
    ``[num_trajectories, max_actions]`` matrix and reduced column by column, so
    the Python loop length is the longest action count rather than the token
    count. This is bitwise-identical to a per-token loop but roughly two orders
    of magnitude faster on realistic batches.

    Args:
        values (torch.Tensor): Frozen value predictions, ``[T]`` or ``[1, T]``.
        token_rewards (torch.Tensor): Per-token rewards, same shape as ``values``.
        action_mask (torch.Tensor): Boolean-like action mask, same shape.
        cu_seq_lens (torch.Tensor): Cumulative packed boundaries ``[0, ..., T]``.
        gamma (float): Discount factor.
        gae_lambda (float): GAE lambda, trading bias against variance.

    Returns:
        torch.Tensor: Float32 advantages, zero outside action positions.
    """
    _validate_unit_interval("gamma", gamma)
    _validate_unit_interval("gae_lambda", gae_lambda)
    if values.shape != token_rewards.shape or values.shape != action_mask.shape:
        raise ValueError(
            "values, token_rewards and action_mask must have the same shape, got "
            f"{values.shape}, {token_rewards.shape} and {action_mask.shape}"
        )

    flat_values = _flatten(values, "values").detach().float()
    device = flat_values.device
    flat_rewards = _flatten(token_rewards, "token_rewards").detach().to(device=device, dtype=torch.float32)
    flat_mask = _flatten(action_mask, "action_mask").to(device=device).bool()
    bounds = _validate_bounds(cu_seq_lens, flat_values.numel()).to(device)

    advantages = torch.zeros_like(flat_values)
    action_idx, segment = _action_index(flat_mask, bounds)
    if action_idx.numel() == 0:
        return advantages.reshape(values.shape)

    num_traj = bounds.numel() - 1
    counts = torch.bincount(segment, minlength=num_traj)
    max_actions = int(counts.max().item())
    # Rank of each action within its own trajectory.
    offsets = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(action_idx.numel(), device=device) - offsets[segment]

    dense_values = torch.zeros((num_traj, max_actions), dtype=torch.float32, device=device)
    dense_rewards = torch.zeros_like(dense_values)
    is_action = torch.zeros((num_traj, max_actions), dtype=torch.bool, device=device)
    dense_values[segment, rank] = flat_values[action_idx]
    dense_rewards[segment, rank] = flat_rewards[action_idx]
    is_action[segment, rank] = True

    # V(s_{t+1}) along the action chain; the terminal action bootstraps from 0.
    next_values = torch.zeros_like(dense_values)
    next_values[:, :-1] = dense_values[:, 1:]
    has_next = torch.zeros_like(is_action)
    has_next[:, :-1] = is_action[:, 1:]
    next_values = torch.where(has_next, next_values, torch.zeros_like(next_values))

    delta = torch.where(
        is_action,
        dense_rewards + gamma * next_values - dense_values,
        torch.zeros_like(dense_values),
    )

    dense_adv = torch.zeros_like(delta)
    carry = torch.zeros(num_traj, dtype=torch.float32, device=device)
    coef = gamma * gae_lambda
    for column in range(max_actions - 1, -1, -1):
        carry = delta[:, column] + coef * carry
        dense_adv[:, column] = carry

    advantages[action_idx] = dense_adv[segment, rank]
    return advantages.reshape(values.shape)


class GAEEstimator(TokenLevelAdvantageEstimator):
    """Generalized Advantage Estimation. https://arxiv.org/abs/1506.02438

    Computes token-level advantages from a learned value function::

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lambda * A_{t+1}
        R_t     = A_t + V(s_t)

    Unlike the group-baseline estimators, this needs a critic, so it is invoked
    by the training worker after the critic forward pass.

    Args:
        gamma (float): Discount factor. Defaults to 1.0, standard for RLHF where
            episodes are short and undiscounted.
        gae_lambda (float): GAE lambda. Defaults to 0.95.
    """

    def __init__(self, gamma: float = 1.0, gae_lambda: float = 0.95) -> None:
        _validate_unit_interval("gamma", gamma)
        _validate_unit_interval("gae_lambda", gae_lambda)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute(
        self,
        values: torch.Tensor,
        token_rewards: torch.Tensor,
        action_mask: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and value-regression targets.

        Args:
            values (torch.Tensor): Frozen critic predictions, ``[T]`` or ``[1, T]``.
            token_rewards (torch.Tensor): Per-token rewards, same shape.
            action_mask (torch.Tensor): Boolean-like action mask, same shape.
            cu_seq_lens (torch.Tensor): Cumulative packed boundaries.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Detached advantages and returns,
                both zero outside action positions.
        """
        frozen_values = values.detach().float()
        advantages = action_gae(
            frozen_values,
            token_rewards,
            action_mask,
            cu_seq_lens,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        # returns = advantage + value is the standard TD(lambda) target; zeroing
        # non-action positions keeps them out of the value loss.
        mask = action_mask.to(device=advantages.device).bool()
        returns = torch.where(mask, advantages + frozen_values, torch.zeros_like(advantages))
        return advantages.detach(), returns.detach()

    def __repr__(self) -> str:
        return f"GAEEstimator(gamma={self.gamma}, gae_lambda={self.gae_lambda})"


def _flatten(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim not in (1, 2) or (tensor.ndim == 2 and tensor.size(0) != 1):
        raise ValueError(f"{name} must have shape [T] or [1, T], got {tuple(tensor.shape)}")
    return tensor.reshape(-1)


def _validate_bounds(cu_seq_lens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if cu_seq_lens.ndim != 1:
        raise ValueError(f"cu_seq_lens must be one-dimensional, got {tuple(cu_seq_lens.shape)}")
    bounds = cu_seq_lens.detach().to(torch.long)
    if bounds.numel() < 2:
        raise ValueError(f"cu_seq_lens must hold at least two boundaries, got {bounds.tolist()}")
    if int(bounds[0].item()) != 0 or int(bounds[-1].item()) != num_tokens:
        raise ValueError(f"cu_seq_lens must start at 0 and end at {num_tokens}, got {bounds.tolist()}")
    if not bool((bounds[1:] > bounds[:-1]).all()):
        raise ValueError(f"cu_seq_lens must be strictly increasing, got {bounds.tolist()}")
    return bounds


def _action_index(flat_mask: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the flat indices of action tokens and the trajectory each belongs to."""
    action_idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
    segment = torch.searchsorted(bounds.to(action_idx.device), action_idx, right=True) - 1
    return action_idx, segment


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
