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
        is_uniform (bool): Whether eligible trajectories all have the same reward.
        sample_eligible (tuple[bool, ...]): Sample-level eligibility after truncation filtering.
    """

    actor: tuple[torch.Tensor, ...]
    critic: tuple[torch.Tensor, ...]
    is_uniform: bool
    sample_eligible: tuple[bool, ...]


def deterministic_truncated_keep_mask(
    finish_reasons: Sequence[str | None],
    rollout_ids: Sequence[str | int],
    *,
    selection_seed: int = 0,
    step: int = 0,
    group_id: str | int = "",
) -> list[bool]:
    """Keep exactly one max-length trajectory in a group using a stable hash.

    Only ``finish_reason == "length"`` participates in this selection. Other
    invalid finish reasons are left unchanged for the caller to handle.

    Args:
        finish_reasons (Sequence[str | None]): Per-trajectory finish reasons.
        rollout_ids (Sequence[str | int]): Stable and unique rollout identifiers.
        selection_seed (int): Stable selection seed.
        step (int): Collector or training step included in the hash.
        group_id (str | int): Stable rollout group identifier.

    Returns:
        list[bool]: Per-trajectory keep decisions.
    """
    if len(finish_reasons) != len(rollout_ids):
        raise ValueError(
            f"finish_reasons and rollout_ids must have the same length, got {len(finish_reasons)} and "
            f"{len(rollout_ids)}"
        )

    truncated_indices = [idx for idx, reason in enumerate(finish_reasons) if reason == "length"]
    truncated_ids = [str(rollout_ids[idx]) for idx in truncated_indices]
    if len(truncated_ids) != len(set(truncated_ids)):
        raise ValueError("rollout_ids for truncated trajectories must be unique.")

    keep = [True] * len(finish_reasons)
    if not truncated_indices:
        return keep

    def selection_key(index: int) -> bytes:
        payload = f"{selection_seed}\x1f{step}\x1f{group_id}\x1f{rollout_ids[index]}".encode()
        return hashlib.sha256(payload).digest()

    selected = min(truncated_indices, key=selection_key)
    for index in truncated_indices:
        keep[index] = index == selected
    return keep


def build_group_loss_masks(
    action_masks: Sequence[torch.Tensor],
    rewards: Sequence[float],
    sample_eligible: Sequence[bool] | None = None,
) -> GroupLossMasks:
    """Build critic-preserving and uniform-aware masks for one rollout group.

    Eligible trajectories always retain their action tokens for critic training.
    If their rewards are uniform, the entire group is masked from actor training.

    Args:
        action_masks (Sequence[torch.Tensor]): Per-trajectory controllable-action masks.
        rewards (Sequence[float]): Per-trajectory scalar rewards.
        sample_eligible (Sequence[bool] | None): Eligibility after invalid/truncated selection.

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

    bool_action_masks = tuple(mask.bool() for mask in action_masks)
    critic_masks = tuple(
        mask if is_eligible else torch.zeros_like(mask) for mask, is_eligible in zip(bool_action_masks, eligible)
    )
    actor_candidates = [
        idx for idx, (mask, is_eligible) in enumerate(zip(bool_action_masks, eligible)) if is_eligible and mask.any()
    ]
    unique_rewards = {float(rewards[idx]) for idx in actor_candidates}
    is_uniform = len(unique_rewards) < 2
    actor_masks = tuple(
        mask if is_eligible and not is_uniform else torch.zeros_like(mask)
        for mask, is_eligible in zip(bool_action_masks, eligible)
    )
    return GroupLossMasks(
        actor=actor_masks,
        critic=critic_masks,
        is_uniform=is_uniform,
        sample_eligible=eligible,
    )
