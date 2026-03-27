"""PresetPackDataset: loads user-provided pack configurations.

Pack config directory format (NPY directory):
    boundaries.npy: int64 array, shape (num_packs+1,) — CSR boundaries
    samples.npy:    int64 array, shape (total_slices, 6)
        columns: [path_id, sample_idx, char_start, char_end, token_start_offset, token_end_offset]
        For plain DataItem: char_start == char_end == -1
    paths.json:     JSON array of strings — path_id -> dataset_path mapping
"""

import json
import os
from typing import Literal, TypedDict, cast

import numpy as np
import torch.utils.data as tud

from xtuner.v1.utils import get_logger, profile_time

from .data_item import DataItem, LongTextDataItem
from .jsonl import JsonlDataset
from .utils import get_longest


logger = get_logger()


class PackConfig(TypedDict):
    boundaries: np.ndarray  # int64, shape (num_packs+1,)
    samples: np.ndarray  # int64, shape (total_slices, 6)
    paths: list[str]  # path_id -> dataset_path


def load_config(path: str, mmap: bool = True) -> PackConfig:
    """Load pack config from an NPY directory.

    Args:
        path (str): Directory containing boundaries.npy, samples.npy, paths.json.
        mmap (bool): If True, load int64 arrays with mmap_mode='r' to avoid loading
            large arrays into RAM. paths.json is always fully loaded.

    Returns:
        PackConfig with 'boundaries' (int64 ndarray, shape (num_packs+1,)),
        'samples' (int64 ndarray, shape (total_slices, 6)),
        'paths' (list[str]).
    """
    mmap_mode = "r" if mmap else None
    with open(os.path.join(path, "paths.json"), encoding="utf-8") as f:
        paths: list[str] = json.load(f)
    return {
        "boundaries": np.load(os.path.join(path, "boundaries.npy"), mmap_mode=mmap_mode),
        "samples": np.load(os.path.join(path, "samples.npy"), mmap_mode=mmap_mode),
        "paths": paths,
    }


class PresetPackDataset(tud.Dataset):
    """Dataset that reads pack groupings from a user-supplied NPY directory.

    The interface of ``__getitem__`` is identical to ``HardPackDataset``:
    it returns a ``list[DataItem]`` (one item per source sample slice in the
    pack).

    Parameters:
        datasets (list[JsonlDataset]): List of source datasets.
        pack_config_path (str): Path to the NPY directory containing
            boundaries.npy, samples.npy, and paths.json.
        pack_max_length (int): Expected total token count per pack.
        short_pack_strategy (str): What to do when a pack has fewer tokens than
            ``pack_max_length``. ``"error"`` raises; ``"padding"`` appends pad tokens.
        long_pack_strategy (str): What to do when a pack has more tokens than
            ``pack_max_length``. ``"error"`` raises; ``"truncate"`` truncates at
            pack_max_length during ``__getitem__``.
        mmap (bool): If True, use memory mapping for large arrays (default True).
    """

    def __init__(
        self,
        datasets: list[JsonlDataset],
        pack_config_path: str,
        pack_max_length: int,
        short_pack_strategy: Literal["error", "padding"] = "error",
        long_pack_strategy: Literal["error", "truncate"] = "error",
        mmap: bool = True,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.pack_max_length = pack_max_length
        self.short_pack_strategy = short_pack_strategy
        self.long_pack_strategy = long_pack_strategy

        config = load_config(pack_config_path, mmap=mmap)
        self._boundaries: np.ndarray = config["boundaries"]
        self._samples: np.ndarray = config["samples"]
        self._paths: list[str] = config["paths"]
        self._path_to_ds_idx: dict[str, int] = {ds.path: idx for idx, ds in enumerate(datasets)}

        self._validate_arrays()
        logger.info(f"PresetPackDataset: {len(self)} valid packs loaded.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_arrays(self) -> None:
        logger.info("PresetPackDataset._validate_arrays: start")
        boundaries = self._boundaries
        samples = self._samples
        paths = self._paths

        # Structural checks
        if int(boundaries[-1]) != len(samples):
            raise ValueError(f"boundaries[-1] ({int(boundaries[-1])}) != len(samples) ({len(samples)}).")
        if samples.ndim != 2 or samples.shape[1] != 6:
            raise ValueError(f"samples must have shape (N, 6), got {samples.shape}.")

        n_packs = int(len(boundaries) - 1)
        if n_packs == 0:
            return

        n_paths = len(paths)
        path_ids = samples[:, 0]
        s_idxs = samples[:, 1]
        char_starts = samples[:, 2]
        char_ends = samples[:, 3]
        tok_offs = samples[:, 4]
        tok_ends = samples[:, 5]

        # path_id range
        if np.any(path_ids < 0) or np.any(path_ids >= n_paths):
            raise ValueError("path_id out of range [0, len(paths)).")

        # token offset validity
        if np.any(tok_offs < 0):
            raise ValueError("token_start_offset must be >= 0.")
        if np.any(tok_ends <= tok_offs):
            raise ValueError("token_end_offset must be > token_start_offset.")

        # char range: must be both -1 (plain DataItem) or valid positive range
        mixed_mask = ~((char_starts == -1) & (char_ends == -1))
        if np.any(mixed_mask):
            cs = char_starts[mixed_mask]
            ce = char_ends[mixed_mask]
            if np.any(cs < 0) or np.any(ce <= cs):
                raise ValueError("Invalid char range: require char_start >= 0 and char_end > char_start.")

        with profile_time("PresetPackDataset._validate_arrays: sample_idx per path"):
            # 路径是否在 datasets 里
            for path_id_val in range(n_paths):
                ds_path = str(paths[path_id_val])
                if ds_path not in self._path_to_ds_idx:
                    raise ValueError(f"dataset_path '{ds_path}' not found in datasets list.")

            n_samples_per_path = np.fromiter(
                (len(self.datasets[self._path_to_ds_idx[str(paths[p])]]) for p in range(n_paths)),
                dtype=np.int64,
                count=n_paths,
            )

            # sample_idx < 0：对已读取的 s_idxs 做一次 np.any(s_idxs < 0)；若有错，用 flatnonzero 定位首行并沿用原来的报错文案。
            if np.any(s_idxs < 0):
                i = int(np.flatnonzero(s_idxs < 0)[0])
                p = int(path_ids[i])
                ds_path = str(paths[p])
                n_samples = int(n_samples_per_path[p])
                raise ValueError(f"sample_idx out of range [0, {n_samples}) for dataset '{ds_path}'.")

            # 上界检查：不再对每个 path 在 1 亿行上建 mask。对每个 path_id 用 np.maximum.at 在 O(行数) 内聚合成 max_s_idx[path_id]，
            # 再与长度仅 n_paths 的 n_samples_per_path 比较：若某 path 的最大 sample_idx 都 < n_samples，则该 path 下所有行都合法。
            max_s_idx = np.full(n_paths, -1, dtype=np.int64)
            np.maximum.at(max_s_idx, path_ids, s_idxs)
            bad_paths = max_s_idx >= n_samples_per_path
            if np.any(bad_paths):
                p = int(np.flatnonzero(bad_paths)[0])
                ds_path = str(paths[p])
                n_samples = int(n_samples_per_path[p])
                raise ValueError(f"sample_idx out of range [0, {n_samples}) for dataset '{ds_path}'.")

        # per-pack token total checks (only for error strategies)
        if self.short_pack_strategy == "error" or self.long_pack_strategy == "error":
            token_lengths = tok_ends - tok_offs
            pack_totals = np.add.reduceat(token_lengths, boundaries[:-1].astype(np.intp))
            if self.short_pack_strategy == "error":
                bad = np.where(pack_totals < self.pack_max_length)[0]
                if len(bad) > 0:
                    idx = int(bad[0])
                    raise ValueError(
                        f"Pack {idx}: total tokens {int(pack_totals[idx])} < pack_max_length {self.pack_max_length}."
                    )
            if self.long_pack_strategy == "error":
                bad = np.where(pack_totals > self.pack_max_length)[0]
                if len(bad) > 0:
                    idx = int(bad[0])
                    raise ValueError(
                        f"Pack {idx}: total tokens {int(pack_totals[idx])} > pack_max_length {self.pack_max_length}."
                    )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(len(self._boundaries) - 1)

    @property
    def longest(self) -> np.ndarray:
        """Per-pack max token length among sub-samples (same meaning as
        ``HardPackDataset.longest``).

        A new int64 vector of shape ``(num_packs,)`` on each access (not cached) so ``_samples`` /
        ``_boundaries`` stay mmap-backed.
        """
        return get_longest(self._boundaries, self._samples)

    def __getitem__(self, i: int) -> list[DataItem]:
        start = int(self._boundaries[i])
        end = int(self._boundaries[i + 1])
        rows = self._samples[start:end]

        items: list[DataItem] = []
        running_tokens = 0

        for row in rows:
            path_id = int(row[0])
            s_idx = int(row[1])
            char_start = int(row[2])
            char_end = int(row[3])
            tok_off = int(row[4])
            tok_end = int(row[5])

            ds_path = str(self._paths[path_id])
            ds_idx = self._path_to_ds_idx[ds_path]
            item: DataItem = cast(DataItem, self.datasets[ds_idx][s_idx])

            if char_start == -1:
                if "char_start" in item:
                    raise ValueError(
                        f"Pack {i}, sample_idx {s_idx}: pack config expects plain DataItem "
                        f"(char_start==-1) but dataset returned LongTextDataItem."
                    )
                if self.long_pack_strategy == "truncate":
                    remaining = self.pack_max_length - running_tokens
                    tok_end = min(tok_end, tok_off + remaining)
                item = {
                    "input_ids": item["input_ids"][tok_off:tok_end],
                    "labels": item["labels"][tok_off:tok_end],
                    "num_tokens": tok_end - tok_off,
                }
            else:
                long_item = cast(LongTextDataItem, item)
                if (
                    "char_start" not in item
                    or long_item["char_start"] != char_start
                    or long_item["char_end"] != char_end
                    or long_item["token_start_offset"] != tok_off
                ):
                    raise ValueError(
                        f"Pack {i}, sample_idx {s_idx}: LongTextDataItem fields mismatch. "
                        f"Expected char_start={char_start}, char_end={char_end}, "
                        f"token_start_offset={tok_off}. "
                        f"Got char_start={item.get('char_start')}, "
                        f"char_end={item.get('char_end')}, "
                        f"token_start_offset={item.get('token_start_offset')}."
                    )

            running_tokens += item["num_tokens"]
            items.append(item)

            if self.long_pack_strategy == "truncate" and running_tokens >= self.pack_max_length:
                break

        if self.short_pack_strategy == "padding":
            pad_len = self.pack_max_length - running_tokens
            if pad_len > 0:
                pad_item: DataItem = {
                    "input_ids": [0] * pad_len,
                    "labels": [-100] * pad_len,
                    "num_tokens": pad_len,
                }
                items.append(pad_item)

        return items

    # ------------------------------------------------------------------
    # State dict (no-op: pack config is fully determined by the NPY files)
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass
