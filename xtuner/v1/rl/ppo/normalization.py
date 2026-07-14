import torch
import torch.distributed as dist


def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
    distributed: bool = True,
    process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Normalize masked advantages with local or distributed population moments.

    Args:
        advantages (torch.Tensor): Dense advantages.
        mask (torch.Tensor): Boolean-like eligibility mask with the same shape.
        eps (float): Numerical stability term added to the standard deviation.
        distributed (bool): Use global moments when torch.distributed is initialized.
        process_group (dist.ProcessGroup | None): Optional process group for global moments.

    Returns:
        torch.Tensor: Float32 normalized advantages, with zero at masked positions.
    """
    if advantages.shape != mask.shape:
        raise ValueError(f"advantages and mask must have the same shape, got {advantages.shape} and {mask.shape}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    values = advantages.float()
    valid_mask = mask.to(device=values.device).bool()
    valid_values = values.masked_select(valid_mask)
    stats = torch.stack(
        (
            valid_values.sum(),
            valid_values.square().sum(),
            valid_mask.sum().to(torch.float32),
        )
    )
    if distributed and dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=process_group)

    count = stats[2]
    if count.item() == 0:
        return torch.zeros_like(values)
    mean = stats[0] / count
    variance = torch.clamp(stats[1] / count - mean.square(), min=0.0)
    normalized = (values - mean) / (variance.sqrt() + eps)
    return torch.where(valid_mask, normalized, 0.0)
