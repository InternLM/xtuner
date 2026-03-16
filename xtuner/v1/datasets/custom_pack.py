"""CustomPackDataset: loads user-provided pack configurations.

Pack config file formats
------------------------
JSONL (one pack per line):
    {"samples": [[dataset_path, sample_idx, char_start, char_end, token_start_offset], ...]}
    For plain DataItem: char_start == char_end == -1, token_start_offset == 0

Parquet (single .parquet file, efficient for large configs):
    boundaries: np.ndarray  shape (num_packs+1,)  -- CSR boundaries
    samples:    list[list[int]]  -- [[path_id, sample_idx, char_start, char_end, token_start_offset], ...]
    paths:      list[str]        -- path_id -> dataset_path mapping
"""

import json
from typing import Literal, cast

import numpy as np
import torch.utils.data as tud

from xtuner.v1.utils import get_logger

from .data_item import DataItem
from .jsonl import JsonlDataset, load_mixed_dict_from_parquet


logger = get_logger()

# (dataset_path, sample_idx, char_start, char_end, token_start_offset)
_PackSlice = tuple[str, int, int, int, int]


def _load_pack_config_jsonl(path: str) -> list[list[_PackSlice]]:
    """Load pack config from a JSONL file.

    Returns:
        list[list[_PackSlice]]: A list of packs, each pack is a list of slices
            (dataset_path, sample_idx, char_start, char_end, token_start_offset).
    """
    packs: list[list[_PackSlice]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e}") from e
            if "samples" not in obj:
                raise ValueError(f"Missing 'samples' key on line {lineno} of {path}.")
            samples = obj["samples"]
            if not isinstance(samples, list) or len(samples) == 0:
                raise ValueError(f"'samples' must be a non-empty list on line {lineno} of {path}.")
            pack: list[_PackSlice] = []
            for s in samples:
                if not (isinstance(s, (list, tuple)) and len(s) == 5):
                    raise ValueError(
                        f"Each slice must be [dataset_path, sample_idx, char_start, char_end, token_start_offset] "
                        f"on line {lineno} of {path}. Got: {s}"
                    )
                pack.append((str(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4])))
            packs.append(pack)
    return packs


def _load_pack_config_parquet(path: str) -> list[list[_PackSlice]]:
    """Load pack config from a Parquet file.

    Returns:
        list[list[_PackSlice]]: A list of packs, each pack is a list of slices
            (dataset_path, sample_idx, char_start, char_end, token_start_offset).
    """
    data = load_mixed_dict_from_parquet(path)
    boundaries = np.asarray(data["boundaries"], dtype=np.int64)
    samples: list[list[int]] = cast(list[list[int]], data["samples"])
    paths: list[str] = cast(list[str], data["paths"])

    packs: list[list[_PackSlice]] = []
    for i in range(len(boundaries) - 1):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        pack: list[_PackSlice] = []
        for s in samples[start:end]:
            path_id, s_idx, c_start, c_end, tok_off = int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4])
            pack.append((paths[path_id], s_idx, c_start, c_end, tok_off))
        packs.append(pack)
    return packs


def _load_pack_config(path: str) -> list[list[_PackSlice]]:
    """Dispatch to the correct loader based on file extension."""
    if path.endswith(".jsonl"):
        return _load_pack_config_jsonl(path)
    elif path.endswith(".parquet"):
        return _load_pack_config_parquet(path)
    raise ValueError(f"Unsupported pack config format: {path}. Expected .jsonl or .parquet.")


class CustomPackDataset(tud.Dataset):
    """Dataset that reads pack groupings from a user-supplied config file.

    The interface of ``__getitem__`` is identical to ``HardPackDataset``:
    it returns a ``list[DataItem]`` (one item per source sample slice in the
    pack).

    Parameters
    ----------
    datasets:
        List of :class:`JsonlDataset` instances returned by ``build_datasets()``.
    pack_config_path:
        Path to the pack configuration file (JSONL or Parquet).
    pack_max_length:
        Expected total token count per pack.
    short_pack_strategy:
        What to do when a pack has fewer tokens than ``pack_max_length``.
        ``"error"`` raises; ``"skip"`` drops the pack; ``"padding"`` pads to
        length (labels for pad positions are ``-100``).
    long_pack_strategy:
        What to do when a pack has more tokens than ``pack_max_length``.
        ``"error"`` raises; ``"skip"`` drops the pack; ``"truncate"``
        truncates the last slice so the total equals ``pack_max_length``.
    """

    def __init__(
        self,
        datasets: list[JsonlDataset],
        pack_config_path: str,
        pack_max_length: int,
        short_pack_strategy: Literal["error", "skip", "padding"] = "error",
        long_pack_strategy: Literal["error", "skip", "truncate"] = "error",
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.pack_max_length = pack_max_length
        self.short_pack_strategy = short_pack_strategy
        self.long_pack_strategy = long_pack_strategy

        # ------------------------------------------------------------------
        # 1. Load pack config file
        # ------------------------------------------------------------------
        raw_packs = _load_pack_config(pack_config_path)
        logger.info(f"CustomPackDataset: loaded {len(raw_packs)} raw packs from {pack_config_path}.")

        # ------------------------------------------------------------------
        # 2. Pre-compute per-dataset token counts indexed by sampled position
        # ------------------------------------------------------------------
        self._sample_num_tokens: list[np.ndarray] = []
        for ds in datasets:
            sampled_arr = np.array(ds.sampled, dtype=np.int64)
            assert ds.num_tokens is not None
            self._sample_num_tokens.append(ds.num_tokens[sampled_arr])

        # ------------------------------------------------------------------
        # 3. Validate packs and build self.pack_infos
        # ------------------------------------------------------------------
        valid_packs: list[list[tuple[int, int, int, int]]] = []
        num_skipped = 0
        used_pairs: set[tuple[int, int]] = set()

        for pack_idx, raw_pack in enumerate(raw_packs):
            slices, ok = self._validate_pack(pack_idx, raw_pack)
            if slices is None:
                if ok == "skip":
                    num_skipped += 1
                    continue
            else:
                valid_packs.append(slices)
                for ds_id, s_idx, _, _ in slices:
                    used_pairs.add((ds_id, s_idx))

        self.pack_infos: list[list[tuple[int, int, int, int]]] = valid_packs

        # ------------------------------------------------------------------
        # 4. Log summary
        # ------------------------------------------------------------------
        total_samples = sum(len(ds) for ds in datasets)
        used_samples = len(used_pairs)
        pct = 100.0 * used_samples / total_samples if total_samples > 0 else 0.0
        logger.info(
            f"CustomPackDataset: loaded {len(valid_packs)} packs ({num_skipped} skipped).\n"
            f"Total sample coverage: {used_samples}/{total_samples} samples ({pct:.1f}%) "
            f"across all datasets."
        )
        for ds_id, ds in enumerate(datasets):
            ds_total = len(ds)
            ds_used = sum(1 for (d, _) in used_pairs if d == ds_id)
            ds_pct = 100.0 * ds_used / ds_total if ds_total > 0 else 0.0
            ds_name = getattr(ds, "name", str(ds_id))
            logger.info(f"  dataset[{ds_id}] ({ds_name}): {ds_used}/{ds_total} samples ({ds_pct:.1f}%)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_pack(
        self, pack_idx: int, raw_pack: list[_PackSlice]
    ) -> tuple[list[tuple[int, int, int, int]] | None, str]:
        """Validate one raw pack.

        Returns
        -------
        (slices, "ok")    – valid pack; slices may be modified by truncation.
        (None,  "skip")   – pack should be skipped.
        Never returns (None, "error") – errors are raised directly.
        """
        # TODO: Feature 2 – rewrite to use new path->index mapping and remove skip
        slices: list[tuple[int, int, int, int]] = []

        for entry in raw_pack:
            # entry = (dataset_path, s_idx, char_start, char_end, token_start_offset)
            # For now map dataset_path to ds_id by position (legacy compat stub)
            # This will be replaced entirely in Feature 2.
            dataset_path, s_idx, t_start, t_end, _tok_off = entry
            ds_id = next(
                (i for i, ds in enumerate(self.datasets) if ds.path == dataset_path),
                -1,
            )

            if ds_id < 0 or ds_id >= len(self.datasets):
                raise ValueError(f"Pack {pack_idx}: dataset_id {ds_id} is out of range [0, {len(self.datasets)}).")
            ds_num_tokens = self._sample_num_tokens[ds_id]
            n_samples = len(ds_num_tokens)
            if s_idx < 0 or s_idx >= n_samples:
                raise ValueError(
                    f"Pack {pack_idx}: sample_idx {s_idx} is out of range [0, {n_samples}) for dataset {ds_id}."
                )
            tok_len = int(ds_num_tokens[s_idx])
            resolved_end = tok_len if t_end == 0 else t_end
            if t_start < 0 or resolved_end > tok_len or t_start >= resolved_end:
                raise ValueError(
                    f"Pack {pack_idx}: invalid token range [{t_start}, {t_end}) "
                    f"for sample ({ds_id}, {s_idx}) with {tok_len} tokens."
                )
            slices.append((ds_id, s_idx, t_start, resolved_end))

        total_tokens = sum(end - start for _, _, start, end in slices)

        if total_tokens < self.pack_max_length:
            if self.short_pack_strategy == "error":
                raise ValueError(
                    f"Pack {pack_idx}: total tokens {total_tokens} < pack_max_length {self.pack_max_length}."
                )
            elif self.short_pack_strategy == "skip":
                logger.warning(
                    f"CustomPackDataset: skipping pack {pack_idx} "
                    f"(total tokens {total_tokens} < pack_max_length {self.pack_max_length})."
                )
                return None, "skip"

        elif total_tokens > self.pack_max_length:
            if self.long_pack_strategy == "error":
                raise ValueError(
                    f"Pack {pack_idx}: total tokens {total_tokens} > pack_max_length {self.pack_max_length}."
                )
            elif self.long_pack_strategy == "skip":
                logger.warning(
                    f"CustomPackDataset: skipping pack {pack_idx} "
                    f"(total tokens {total_tokens} > pack_max_length {self.pack_max_length})."
                )
                return None, "skip"
            else:  # "truncate"
                excess = total_tokens - self.pack_max_length
                ds_id, s_idx, t_start, t_end = slices[-1]
                new_end = t_end - excess
                if new_end <= t_start:
                    raise ValueError(
                        f"Pack {pack_idx}: truncation would make the last slice empty "
                        f"(slice [{t_start}, {new_end})). Reduce pack size or adjust pack config."
                    )
                slices[-1] = (ds_id, s_idx, t_start, new_end)

        return slices, "ok"

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pack_infos)

    def __getitem__(self, i: int) -> list[DataItem]:
        pack = self.pack_infos[i]
        items: list[DataItem] = []

        for ds_id, s_idx, t_start, t_end in pack:
            # TODO: Feature 3 – replace token slicing with consistency check
            raw_item: DataItem = cast(DataItem, self.datasets[ds_id][s_idx])
            sliced: DataItem = {
                "input_ids": raw_item["input_ids"][t_start:t_end],
                "labels": raw_item["labels"][t_start:t_end],
                "num_tokens": t_end - t_start,
            }
            items.append(sliced)

        if self.short_pack_strategy == "padding":
            total_tokens = sum(item["num_tokens"] for item in items)
            pad_len = self.pack_max_length - total_tokens
            if pad_len > 0:
                pad_item: DataItem = {
                    "input_ids": [0] * pad_len,
                    "labels": [-100] * pad_len,
                    "num_tokens": pad_len,
                }
                items.append(pad_item)

        return items

    # ------------------------------------------------------------------
    # State dict (no-op: pack_infos are deterministic from the config file)
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass
