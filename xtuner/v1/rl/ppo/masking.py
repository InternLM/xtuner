import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class GroupLossMasks:
    """Actor and critic masks for one rollout group.

    Args:
        actor (tuple[torch.Tensor, ...]): Per-trajectory actor masks.
        critic (tuple[torch.Tensor, ...]): Per-trajectory critic masks.
        is_uniform (bool): Whether Actor-eligible trajectories all have the same reward.
        sample_eligible (tuple[bool, ...]): Actor eligibility after truncation filtering.
        critic_is_uniform (bool): Whether Critic-eligible trajectories all have the same reward.
        critic_sample_eligible (tuple[bool, ...]): Critic eligibility after truncation filtering.
    """

    actor: tuple[torch.Tensor, ...]
    critic: tuple[torch.Tensor, ...]
    is_uniform: bool
    sample_eligible: tuple[bool, ...]
    critic_is_uniform: bool
    critic_sample_eligible: tuple[bool, ...]


def deterministic_truncated_keep_mask(
    finish_reasons: Sequence[str | None],
    rollout_ids: Sequence[str | int],
    *,
    selection_seed: int = 0,
    step: int = 0,
    group_id: str | int = "",
    max_truncated_per_group: int | None = 1,
) -> list[bool]:
    """Keep a configured number of max-length trajectories using a stable hash.

    Only ``finish_reason == "length"`` participates in this selection. Other
    invalid finish reasons are left unchanged for the caller to handle.

    Args:
        finish_reasons (Sequence[str | None]): Per-trajectory finish reasons.
        rollout_ids (Sequence[str | int]): Stable and unique rollout identifiers.
        selection_seed (int): Stable selection seed.
        step (int): Collector or training step included in the hash.
        group_id (str | int): Stable rollout group identifier.
        max_truncated_per_group (int | None): Maximum retained length-finished
            trajectories. ``None`` keeps all of them.

    Returns:
        list[bool]: Per-trajectory keep decisions.
    """
    if len(finish_reasons) != len(rollout_ids):
        raise ValueError(
            f"finish_reasons and rollout_ids must have the same length, got {len(finish_reasons)} and "
            f"{len(rollout_ids)}"
        )
    if max_truncated_per_group is not None and max_truncated_per_group < 0:
        raise ValueError("max_truncated_per_group must be non-negative or None")

    truncated_indices = [idx for idx, reason in enumerate(finish_reasons) if reason == "length"]
    truncated_ids = [str(rollout_ids[idx]) for idx in truncated_indices]
    if len(truncated_ids) != len(set(truncated_ids)):
        raise ValueError("rollout_ids for truncated trajectories must be unique.")

    keep = [True] * len(finish_reasons)
    if not truncated_indices or max_truncated_per_group is None:
        return keep

    def selection_key(index: int) -> bytes:
        payload = f"{selection_seed}\x1f{step}\x1f{group_id}\x1f{rollout_ids[index]}".encode()
        return hashlib.sha256(payload).digest()

    selected = set(sorted(truncated_indices, key=selection_key)[:max_truncated_per_group])
    for index in truncated_indices:
        keep[index] = index in selected
    return keep


def build_group_loss_masks(
    action_masks: Sequence[torch.Tensor],
    rewards: Sequence[float],
    sample_eligible: Sequence[bool] | None = None,
    *,
    critic_sample_eligible: Sequence[bool] | None = None,
    keep_uniform_groups: bool = True,
) -> GroupLossMasks:
    """Build independent Actor and Critic masks for one rollout group.

    Uniform Actor groups are always masked from policy training. Critic
    eligibility is independent, and uniform Critic groups are retained only
    when ``keep_uniform_groups`` is enabled.

    Args:
        action_masks (Sequence[torch.Tensor]): Per-trajectory controllable-action masks.
        rewards (Sequence[float]): Per-trajectory scalar rewards.
        sample_eligible (Sequence[bool] | None): Actor eligibility after
            invalid/truncated selection.
        critic_sample_eligible (Sequence[bool] | None): Independent Critic
            eligibility. When omitted, it follows Actor eligibility for
            backward compatibility.
        keep_uniform_groups (bool): Whether uniform-reward groups train Critic.

    Returns:
        GroupLossMasks: Actor masks, critic masks, and group-level eligibility metadata.
    """
    if not action_masks:
        raise ValueError("action_masks must not be empty.")
    if len(action_masks) != len(rewards):
        raise ValueError(
            f"action_masks and rewards must have the same length, got {len(action_masks)} and {len(rewards)}"
        )
    if any(not math.isfinite(float(reward)) for reward in rewards):
        raise ValueError("All rewards must be finite.")

    if sample_eligible is None:
        eligible = (True,) * len(action_masks)
    else:
        if len(sample_eligible) != len(action_masks):
            raise ValueError(
                f"sample_eligible must match action_masks, got {len(sample_eligible)} and {len(action_masks)}"
            )
        eligible = tuple(bool(value) for value in sample_eligible)

    if critic_sample_eligible is None:
        critic_eligible = eligible
    else:
        if len(critic_sample_eligible) != len(action_masks):
            raise ValueError(
                "critic_sample_eligible must match action_masks, got "
                f"{len(critic_sample_eligible)} and {len(action_masks)}"
            )
        critic_eligible = tuple(bool(value) for value in critic_sample_eligible)

    bool_action_masks = tuple(mask.bool() for mask in action_masks)
    actor_candidates = [
        idx for idx, (mask, is_eligible) in enumerate(zip(bool_action_masks, eligible)) if is_eligible and mask.any()
    ]
    unique_rewards = {float(rewards[idx]) for idx in actor_candidates}
    is_uniform = len(unique_rewards) < 2
    actor_masks = tuple(
        mask if is_eligible and not is_uniform else torch.zeros_like(mask)
        for mask, is_eligible in zip(bool_action_masks, eligible)
    )
    critic_candidates = [
        idx
        for idx, (mask, is_eligible) in enumerate(zip(bool_action_masks, critic_eligible))
        if is_eligible and mask.any()
    ]
    critic_unique_rewards = {float(rewards[idx]) for idx in critic_candidates}
    critic_is_uniform = len(critic_unique_rewards) < 2
    critic_masks = tuple(
        mask if is_eligible and (keep_uniform_groups or not critic_is_uniform) else torch.zeros_like(mask)
        for mask, is_eligible in zip(bool_action_masks, critic_eligible)
    )
    return GroupLossMasks(
        actor=actor_masks,
        critic=critic_masks,
        is_uniform=is_uniform,
        sample_eligible=eligible,
        critic_is_uniform=critic_is_uniform,
        critic_sample_eligible=critic_eligible,
    )
