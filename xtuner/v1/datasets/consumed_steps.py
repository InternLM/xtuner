"""Track consumed samples for checkpointing; aggregate across DP only (not
SP/TP)."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


def reduce_sum_across_dp_group(dp_mesh: DeviceMesh | None, local_value: int) -> int:
    """Sum ``local_value`` over the DP process group (one contribution per
    data-parallel replica).

    Ranks that only differ in SP/TP see identical data batches and must not be summed with the global world group; see
    Training notes for SP+DP.
    """
    if dp_mesh is None or dp_mesh.size() <= 1:
        return int(local_value)
    if not dist.is_available() or not dist.is_initialized():
        return int(local_value)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    tensor = torch.tensor([local_value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
    return int(tensor.item())


class ConsumedStepsTracker:
    """Holds per-resume totals and per-rank local accumulation; checkpoint
    total uses DP-only reduction."""

    __slots__ = ("_dp_mesh", "_init_steps", "_local_steps")

    def __init__(self, dp_mesh: DeviceMesh | None) -> None:
        self._dp_mesh = dp_mesh
        self._init_steps = 0
        self._local_steps = 0

    def record(self, n: int) -> None:
        self._local_steps += int(n)

    def set_init_from_checkpoint(self, total: int) -> None:
        """After loading a checkpoint: global total consumed so far; reset session-local accumulation."""
        self._init_steps = int(total)
        self._local_steps = 0

    def total_for_checkpoint(self) -> int:
        """Global consumed sample count including this session (collective over
        DP group)."""
        return self._init_steps + reduce_sum_across_dp_group(self._dp_mesh, self._local_steps)


def apply_old_ckpt_init_steps(sampler: object, sampler_state: dict, train_state_total: int | None) -> None:
    """If the sampler checkpoint predates ``total_consumed_steps``, copy the
    total from ``train_state``."""
    if train_state_total is None:
        return
    if sampler_state.get("total_consumed_steps") is not None:
        return
    consumed: ConsumedStepsTracker | None = getattr(sampler, "_consumed", None)
    if consumed is not None:
        consumed.set_init_from_checkpoint(train_state_total)
