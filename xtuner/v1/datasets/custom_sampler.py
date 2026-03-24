"""CustomSampler: consumes packs in a user-supplied global order.

Sampler config file format
--------------------------
Only ``.npy`` is supported: a 1-D integer array of pack indices, loaded with
``numpy.load(..., mmap_mode='r')`` so multiple processes on one machine can
share the same virtual memory mapping and avoid duplicating the full order in
RAM.

In-memory callers must pass a 1-D integer :class:`numpy.ndarray` (not a Python
``list``); a ``.npy`` path is loaded via mmap as above.

The global order may be longer than the number of packs (over-sampling) or
shorter (only consume a subset).  The sampler **rounds down** the total length
to the largest multiple of ``global_batch_size * world_size`` using a basic
slice ``global_order[:rounded_len]``, which keeps a memory-mapped array as a
view (no copy).  Then each rank takes ``effective[rank::world_size]``.
"""

from typing import Iterator

import numpy as np
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import Sampler

from xtuner.v1.utils import get_logger

from .preset_pack import PresetPackDataset


logger = get_logger()


def _load_sampler_config(path: str) -> np.memmap | np.ndarray:
    """Load the global pack consumption order from a ``.npy`` file via mmap.

    Uses read-only mmap so multiple training processes can share the same mapping and reduce peak resident memory.
    """
    if not path.endswith(".npy"):
        raise ValueError(f"CustomSampler: only .npy sampler order files are supported (mmap read). Got path {path!r}.")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 1:
        raise ValueError(f"sampler config NPY must be 1-D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"sampler config NPY must have an integer dtype, got {arr.dtype}")
    return arr


def _validate_pack_indices(order: np.ndarray, num_packs: int) -> None:
    """Ensure all indices are in ``[0, num_packs)`` without materializing huge
    lists."""
    bad = np.where((order < 0) | (order >= num_packs))[0]
    if bad.size:
        first = bad[:5]
        vals = order[first]
        raise ValueError(
            f"CustomSampler: {bad.size} pack index(es) out of range [0, {num_packs}). "
            f"First positions/values (idx -> value): {list(zip(first.tolist(), vals.tolist()))}"
        )


def _log_coverage_summary(order: np.ndarray, num_packs: int) -> None:
    uniq, counts = np.unique(order, return_counts=True)
    used_packs = int(uniq.size)
    repeated = int(np.sum(counts > 1))
    pct = 100.0 * used_packs / num_packs if num_packs > 0 else 0.0
    logger.info(
        f"CustomSampler: global_order covers {used_packs}/{num_packs} packs ({pct:.1f}%). "
        f"({repeated} packs referenced more than once)"
    )


class CustomSampler(Sampler):
    """Distributed sampler that consumes packs in a fixed user-defined order.

    Parameters
    ----------
    dataset:
        The :class:`PresetPackDataset` whose packs are being sampled.
    global_order:
        A 1-D integer :class:`numpy.ndarray` of pack indices, or a path to a
        ``.npy`` file (mmap read).  Python ``list`` is not accepted.
        When loaded from file, ``self.global_order`` is a view into the mmap
        after length round-down (still backed by the file mapping).
    global_batch_size:
        Total batch size across all ranks.  Used together with ``world_size``
        for round-down alignment.
    dp_mesh:
        Optional DeviceMesh for distributed training.  If ``None``, assumes
        single-rank training.
    seed:
        Unused in this sampler (order is deterministic), kept for API parity.
    """

    global_order: np.ndarray

    def __init__(
        self,
        dataset: PresetPackDataset,
        global_order: np.ndarray | str,
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
        # Load order from file if a path was given (mmap only)
        # ------------------------------------------------------------------
        if isinstance(global_order, str):
            logger.info(f"CustomSampler: loading sampler order (mmap) from {global_order}.")
            global_order = _load_sampler_config(global_order)

        if not isinstance(global_order, np.ndarray):
            raise TypeError(
                "CustomSampler: global_order must be a numpy.ndarray (1-D integer), "
                f"got {type(global_order).__name__}. Use np.asarray(..., dtype=np.int64) "
                "or pass a path to a .npy file."
            )
        if global_order.ndim != 1:
            raise ValueError(f"CustomSampler: global_order must be 1-D, got shape {global_order.shape}")
        if not np.issubdtype(global_order.dtype, np.integer):
            raise ValueError(f"CustomSampler: global_order must have an integer dtype, got {global_order.dtype}")

        num_packs = len(dataset)

        # ------------------------------------------------------------------
        # Validate
        # ------------------------------------------------------------------
        _validate_pack_indices(global_order, num_packs)

        # ------------------------------------------------------------------
        # Round-down to multiple of global_batch_size * world_size (basic slice)
        # ------------------------------------------------------------------
        step_size = global_batch_size * self.world_size
        raw_len = len(global_order)
        if raw_len == 0:
            raise ValueError("CustomSampler: global_order is empty.")
        rounded_len = (raw_len // step_size) * step_size
        if rounded_len == 0:
            raise ValueError(
                f"CustomSampler: global_order length {raw_len} is smaller than "
                f"global_batch_size*world_size={step_size}; "
                "cannot round down to a positive multiple. "
                "Increase the order length or decrease batch size / world size."
            )
        if rounded_len < raw_len:
            logger.info(
                f"CustomSampler: truncating global order from {raw_len} to {rounded_len} "
                f"(multiple of {step_size}, round-down)."
            )

        effective = global_order[:rounded_len]
        self.global_order = effective

        # Local indices for this rank: interleaved slicing (view if ndarray)
        self._local_indices = effective[self.rank :: self.world_size]
        self.total_size = rounded_len  # global total (all ranks)
        self.num_samples = len(self._local_indices)  # per-rank

        self.epoch = 0
        self.step = 0

        # ------------------------------------------------------------------
        # Log coverage summary
        # ------------------------------------------------------------------
        _log_coverage_summary(effective, num_packs)

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[int]:
        yield from self._local_indices[self.step :]
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
