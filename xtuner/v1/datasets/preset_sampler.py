"""PresetSampler: consumes packs in a user-supplied global order.

Sampler config file format
--------------------------
Only ``.npy`` paths are supported: a 1-D integer array of pack indices, loaded with
``numpy.load(..., mmap_mode='r')`` so multiple processes on one machine can
share the same virtual memory mapping and avoid duplicating the full order in
RAM.

The file order may be longer than the number of packs (over-sampling) or
shorter (only consume a subset).  The sampler **rounds down** the total length
to the largest multiple of ``global_batch_size * world_size`` using a basic
slice ``order[:rounded_len]``, which keeps a memory-mapped array as a
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
        raise ValueError(f"PresetSampler: only .npy sampler order files are supported (mmap read). Got path {path!r}.")
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
            f"PresetSampler: {bad.size} pack index(es) out of range [0, {num_packs}). "
            f"First positions/values (idx -> value): {list(zip(first.tolist(), vals.tolist()))}"
        )


def _log_coverage_summary(order: np.ndarray, num_packs: int) -> None:
    uniq, counts = np.unique(order, return_counts=True)
    used_packs = int(uniq.size)
    repeated = int(np.sum(counts > 1))
    pct = 100.0 * used_packs / num_packs if num_packs > 0 else 0.0
    logger.info(
        f"PresetSampler: sampler order covers {used_packs}/{num_packs} packs ({pct:.1f}%). "
        f"({repeated} packs referenced more than once)"
    )


class PresetSampler(Sampler):
    """Distributed sampler that consumes packs in a fixed user-defined order.

    Parameters
    ----------
    dataset:
        The :class:`PresetPackDataset` whose packs are being sampled.
    sampler_config_path:
        Path to a ``.npy`` file (mmap read): 1-D integer pack indices.
        After length round-down, ``self.global_order`` holds a view backed by
        that mapping.
    global_batch_size:
        Total batch size across all ranks.  Used together with ``world_size``
        for round-down alignment.
    dp_mesh:
        Optional DeviceMesh for distributed training.  If ``None``, assumes
        single-rank training.
    """

    global_order: np.ndarray

    def __init__(
        self,
        dataset: PresetPackDataset,
        sampler_config_path: str,
        global_batch_size: int,
        dp_mesh: DeviceMesh | None = None,
        round_up: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(sampler_config_path, str):
            raise TypeError(
                "PresetSampler: sampler_config_path must be a str path to a .npy file, "
                f"got {type(sampler_config_path).__name__}."
            )

        if round_up:
            round_up = False
            logger.warning(
                "PresetSampler: round_up is not supported and ignored for preset sampler, due to mmap array limitation."
            )

        if dp_mesh is not None:
            self.rank = dp_mesh.get_local_rank()
            self.world_size = dp_mesh.size()
        else:
            self.rank = 0
            self.world_size = 1

        self.dataset = dataset
        self.global_batch_size = global_batch_size

        logger.info(f"PresetSampler: loading sampler order (mmap) from {sampler_config_path}.")
        order = _load_sampler_config(sampler_config_path)

        if order.ndim != 1:
            raise ValueError(f"PresetSampler: sampler order must be 1-D, got shape {order.shape}")
        if not np.issubdtype(order.dtype, np.integer):
            raise ValueError(f"PresetSampler: sampler order must have an integer dtype, got {order.dtype}")

        num_packs = len(dataset)

        _validate_pack_indices(order, num_packs)

        raw_len = len(order)
        if raw_len == 0:
            raise ValueError("PresetSampler: sampler order is empty.")

        self.num_samples = (raw_len // self.global_batch_size) * self.global_batch_size // self.world_size
        self.total_size = self.num_samples * self.world_size

        if self.total_size == 0:
            raise ValueError(
                f"PresetSampler: sampler order length {raw_len} is smaller than "
                f"global_batch_size={self.global_batch_size}; "
                "cannot round down to a positive multiple. "
            )
        if self.total_size < raw_len:
            logger.info(
                f"PresetSampler: truncating sampler order from {raw_len} to {self.total_size} "
                f"(multiple of {self.global_batch_size}, round-down)."
            )

        self.global_order = order[: self.total_size]

        self.epoch = 0
        self.step = 0

        _log_coverage_summary(self.global_order, num_packs)

    def __iter__(self) -> Iterator[int]:
        # load order from npy → global_order → rank_view 类型均为 memmap, 子视图 的路径仍然保持
        # memmap 语义（视图、按需分页、文件后端）；单机多进程可共享同一份文件页缓存
        yield from self.global_order[self.step + self.rank : self.total_size : self.world_size]
        self.step = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def get_state_dict(self, total_consumed_steps: int) -> dict:
        # Same convention as :class:`LengthGroupedSampler`: ``step`` is the global pack offset
        # (modulo ``total_size``) into ``global_order``, shared across all ranks in the checkpoint.
        global_step = total_consumed_steps % self.total_size
        return {
            "epoch": self.epoch,
            "step": global_step,
            "world_size": self.world_size,
            "num_samples": self.num_samples,
            "total_size": self.total_size,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if self.world_size != state_dict.get("world_size"):
            logger.warning(
                f"PresetSampler: world_size mismatch: checkpoint has "
                f"{state_dict.get('world_size')}, current is {self.world_size}. "
                "Resumption may be inaccurate."
            )

        self.epoch = state_dict["epoch"]
        self.step = int(state_dict["step"])
