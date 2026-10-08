"""Learning-rate schedule and critic pass helpers shared by PPO workers."""

from __future__ import annotations

import math

import torch

from xtuner.v1.config.optim import LRConfig
from xtuner.v1.engine.train_engine import TrainEngine


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_cfg: LRConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the PPO scheduler.

    ``warmup_ratio < 1`` is a fraction of ``total_steps``. A value of 1 or
    greater is an absolute warmup update count.
    """
    if total_steps <= 0:
        raise ValueError(f"scheduler total_steps must be positive, got {total_steps}")
    warmup_steps = int(lr_cfg.warmup_ratio * total_steps) if lr_cfg.warmup_ratio < 1 else int(lr_cfg.warmup_ratio)
    warmup_steps = min(warmup_steps, total_steps)
    base_lr = float(optimizer.defaults["lr"])
    min_factor = float(lr_cfg.lr_min) / base_lr if base_lr > 0 else 1.0

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        if lr_cfg.lr_type == "constant":
            return 1.0
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        if lr_cfg.lr_type == "linear":
            return 1.0 - progress * (1.0 - min_factor)
        if lr_cfg.lr_type == "cosine":
            return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported lr type: {lr_cfg.lr_type}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def optimizer_step_succeeds(engine: TrainEngine, grad_norm: torch.Tensor) -> bool:
    """Return whether ``TrainEngine.step_optimizer`` will apply the update."""
    if not torch.isfinite(grad_norm):
        return False
    threshold = engine.optim_cfg.skip_grad_norm_threshold
    return threshold is None or bool(grad_norm <= threshold)


def partition_indices(indices: list[int], max_updates: int) -> list[list[int]]:
    """Split ``indices`` into at most ``max_updates`` contiguous chunks."""
    if not indices:
        return []
    if max_updates <= 0:
        raise ValueError(f"max_updates must be positive, got {max_updates}")
    num_updates = min(len(indices), max_updates)
    quotient, remainder = divmod(len(indices), num_updates)
    chunks = []
    offset = 0
    for chunk_index in range(num_updates):
        chunk_size = quotient + int(chunk_index < remainder)
        chunks.append(indices[offset : offset + chunk_size])
        offset += chunk_size
    return chunks


def pass_order(num_batches: int, rollout_idx: int, pass_index: int, seed: int) -> list[int]:
    """Keep the first critic pass in pack order and shuffle later passes."""
    if num_batches <= 0:
        return []
    if pass_index == 0:
        return list(range(num_batches))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + rollout_idx * 1_000_003 + pass_index)
    return torch.randperm(num_batches, generator=generator).tolist()
