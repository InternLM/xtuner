"""CustomSampler: consumes packs in a user-supplied global order.

Sampler config file formats
----------------------------
JSONL (single line):
    [3, 1, 7, 2, 0, 5, 4, 6]

NPY:
    sampler_order.npy  – 1-D int64 array of pack indices.

The global order may be longer than the number of packs (over-sampling) or
shorter (only consume a subset).  The sampler round-ups the total length to
the nearest multiple of ``global_batch_size * world_size`` by repeating tail
elements, then slices indices for the local rank.
"""

import json
import math
import os
from typing import Iterator

import numpy as np
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import Sampler

from xtuner.v1.utils import get_logger

from .custom_pack import CustomPackDataset


logger = get_logger()


def _load_sampler_config(path: str) -> list[int]:
    """Load the global pack consumption order from a file.

    Supports JSONL (single line JSON array) and NPY (1-D integer array).
    """
    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.ndim != 1:
            raise ValueError(f"sampler config NPY must be 1-D, got shape {arr.shape}")
        return arr.tolist()
    else:
        # JSONL: single line containing a JSON array.
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        try:
            order = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Cannot parse sampler config as JSON array from {path}: {e}") from e
        if not isinstance(order, list):
            raise ValueError(f"Sampler config must be a JSON array, got {type(order)} in {path}.")
        return [int(x) for x in order]


class CustomSampler(Sampler):
    """Distributed sampler that consumes packs in a fixed user-defined order.

    Parameters
    ----------
    dataset:
        The :class:`CustomPackDataset` whose packs are being sampled.
    global_order:
        A pre-loaded list of pack indices specifying the global consumption
        order.  Can also be passed as a path string; in that case the file is
        loaded automatically.
    global_batch_size:
        Total batch size across all ranks.  Used for round-up behaviour.
    dp_mesh:
        Optional DeviceMesh for distributed training.  If ``None``, assumes
        single-rank training.
    seed:
        Unused in this sampler (order is deterministic), kept for API parity.
    """

    def __init__(
        self,
        dataset: CustomPackDataset,
        global_order: list[int] | str,
        global_batch_size: int,
        dp_mesh: DeviceMesh | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if dp_mesh is not None:
            self.rank = dp_mesh.get_local_rank()
            self.world_size = dp_mesh.size()
        else:
            self.rank = 0
            self.world_size = 1

        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.seed = seed  # kept for API compat, not used

        # ------------------------------------------------------------------
        # Load order from file if a path was given
        # ------------------------------------------------------------------
        if isinstance(global_order, str):
            logger.info(f"CustomSampler: loading sampler order from {global_order}.")
            global_order = _load_sampler_config(global_order)

        num_packs = len(dataset)

        # ------------------------------------------------------------------
        # Validate
        # ------------------------------------------------------------------
        invalid = [idx for idx in global_order if idx < 0 or idx >= num_packs]
        if invalid:
            raise ValueError(
                f"CustomSampler: {len(invalid)} pack index(es) out of range [0, {num_packs}). "
                f"First few invalid: {invalid[:5]}"
            )

        # ------------------------------------------------------------------
        # Round-up to multiple of global_batch_size * world_size
        # ------------------------------------------------------------------
        step_size = global_batch_size * self.world_size
        raw_len = len(global_order)
        rounded_len = math.ceil(raw_len / step_size) * step_size
        if rounded_len > raw_len:
            # Repeat tail elements
            extra = rounded_len - raw_len
            tail = global_order[-(extra % raw_len or raw_len):]
            padded: list[int] = list(global_order) + (tail * (extra // raw_len + 1))[:extra]
        else:
            padded = list(global_order)

        # Local indices for this rank: interleaved slicing
        self._local_indices: list[int] = padded[self.rank :: self.world_size]
        self.total_size = rounded_len              # global total (all ranks)
        self.num_samples = len(self._local_indices)  # per-rank

        self.epoch = 0
        self.step = 0

        # ------------------------------------------------------------------
        # Log coverage summary
        # ------------------------------------------------------------------
        used_packs = len(set(global_order))
        repeated = sum(1 for idx in set(global_order) if global_order.count(idx) > 1)
        pct = 100.0 * used_packs / num_packs if num_packs > 0 else 0.0
        logger.info(
            f"CustomSampler: global_order covers {used_packs}/{num_packs} packs ({pct:.1f}%). "
            f"({repeated} packs referenced more than once)"
        )

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[int]:
        yield from self._local_indices[self.step:]
        self.step = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def get_state_dict(self, step: int) -> dict:
        # step here is consumed_samples (global, across all ranks).
        # Convert to local epoch offset.
        local_step = step % self.total_size
        return {
            "epoch": self.epoch,
            "step": local_step,
            "world_size": self.world_size,
            "num_samples": self.num_samples,
            "total_size": self.total_size,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if self.world_size != state_dict.get("world_size"):
            logger.warning(
                f"CustomSampler: world_size mismatch: checkpoint has "
                f"{state_dict.get('world_size')}, current is {self.world_size}. "
                "Resumption may be inaccurate."
            )

        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
