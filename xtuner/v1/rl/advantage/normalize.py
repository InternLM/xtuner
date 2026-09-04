"""Advantage normalization across the global training batch."""

import torch
import torch.distributed as dist


def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
    process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Normalize masked advantages using global mean and variance.

    Moments are reduced over the whole process group so every rank applies the
    same affine transform; normalizing per rank would make the effective step
    size depend on how data happened to be sharded.

    Sequence parallelism needs no special handling: SP ranks hold replicas of
    the same data, so the summed statistics and the token count are both scaled
    by the SP size and the ratios that define mean and variance are unchanged.

    Args:
        advantages (torch.Tensor): Dense per-token advantages.
        mask (torch.Tensor): Boolean-like mask selecting the tokens to normalize
            over, with the same shape as ``advantages``.
        eps (float): Added to the standard deviation for numerical stability.
        process_group (dist.ProcessGroup | None): Group to reduce over. Defaults
            to the default group.

    Returns:
        torch.Tensor: Float32 normalized advantages, zero at masked-out positions.
    """
    if advantages.shape != mask.shape:
        raise ValueError(f"advantages and mask must have the same shape, got {advantages.shape} and {mask.shape}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    values = advantages.float()
    valid = mask.to(device=values.device).bool()
    selected = values.masked_select(valid)

    # One fused all-reduce: sum, sum of squares, count.
    stats = torch.stack((selected.sum(), selected.square().sum(), valid.sum().to(torch.float32)))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=process_group)

    count = stats[2]
    if float(count.item()) == 0.0:
        return torch.zeros_like(values)

    mean = stats[0] / count
    variance = torch.clamp(stats[1] / count - mean.square(), min=0.0)
    normalized = (values - mean) / (variance.sqrt() + eps)
    return torch.where(valid, normalized, torch.zeros_like(normalized))
