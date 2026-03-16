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


# (ds_idx, sample_idx, char_start, char_end, token_start_offset, max_tokens)
_ValidatedSlice = tuple[int, int, int, int, int, int]


class CustomPackDataset(tud.Dataset):
    """Dataset that reads pack groupings from a user-supplied config file.

    The interface of ``__getitem__`` is identical to ``HardPackDataset``:
    it returns a ``list[DataItem]`` (one item per source sample slice in the
    pack).

    Parameters:
        datasets (list[JsonlDataset]): List of source datasets.
        pack_config_path (str): Path to the pack configuration file (.jsonl or .parquet).
        pack_max_length (int): Expected total token count per pack.
        short_pack_strategy (str): What to do when a pack has fewer tokens than
            ``pack_max_length``. ``"error"`` raises; ``"padding"`` pads to length
            (labels for pad positions are -100).
        long_pack_strategy (str): What to do when a pack has more tokens than
            ``pack_max_length``. ``"error"`` raises; ``"truncate"`` truncates the
            last slice so the total equals ``pack_max_length``.
    """

    def __init__(
        self,
        datasets: list[JsonlDataset],
        pack_config_path: str,
        pack_max_length: int,
        short_pack_strategy: Literal["error", "padding"] = "error",
        long_pack_strategy: Literal["error", "truncate"] = "error",
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.pack_max_length = pack_max_length
        self.short_pack_strategy = short_pack_strategy
        self.long_pack_strategy = long_pack_strategy

        self._path_to_ds_idx: dict[str, int] = {ds.path: idx for idx, ds in enumerate(datasets)}

        raw_packs = _load_pack_config(pack_config_path)
        logger.info(f"CustomPackDataset: loaded {len(raw_packs)} raw packs from {pack_config_path}.")

        self.pack_infos: list[list[_ValidatedSlice]] = [
            self._validate_pack(pack_idx, raw_pack) for pack_idx, raw_pack in enumerate(raw_packs)
        ]
        logger.info(f"CustomPackDataset: {len(self.pack_infos)} valid packs loaded.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_pack(self, pack_idx: int, raw_pack: list[_PackSlice]) -> list[_ValidatedSlice]:
        """Validate one raw pack and return validated slices.

        Args:
            pack_idx (int): Index of the pack (for error messages).
            raw_pack (list[_PackSlice]): Raw slices from the config file.

        Returns:
            list[_ValidatedSlice]: Validated slices as
                (ds_idx, s_idx, char_start, char_end, token_start_offset, max_tokens).
        """
        slices: list[_ValidatedSlice] = []
        total_tokens = 0

        for dataset_path, s_idx, char_start, char_end, token_start_offset in raw_pack:
            if dataset_path not in self._path_to_ds_idx:
                raise ValueError(f"Pack {pack_idx}: dataset_path '{dataset_path}' not found in datasets list.")
            ds_idx = self._path_to_ds_idx[dataset_path]
            ds = self.datasets[ds_idx]

            n_samples = len(ds)
            if s_idx < 0 or s_idx >= n_samples:
                raise ValueError(
                    f"Pack {pack_idx}: sample_idx {s_idx} out of range [0, {n_samples}) for dataset '{dataset_path}'."
                )

            if char_start == -1 and char_end == -1:
                pass  # plain DataItem: no char range validation
            elif char_start < 0 or char_end <= char_start:
                raise ValueError(
                    f"Pack {pack_idx}: invalid char range [{char_start}, {char_end}) "
                    f"for dataset '{dataset_path}', sample_idx {s_idx}."
                )

            if token_start_offset < 0:
                raise ValueError(f"Pack {pack_idx}: token_start_offset {token_start_offset} must be >= 0.")

            assert ds.num_tokens is not None
            n_tokens = int(ds.num_tokens[s_idx])
            total_tokens += n_tokens
            slices.append((ds_idx, s_idx, char_start, char_end, token_start_offset, n_tokens))

        if total_tokens < self.pack_max_length:
            if self.short_pack_strategy == "error":
                raise ValueError(
                    f"Pack {pack_idx}: total tokens {total_tokens} < pack_max_length {self.pack_max_length}."
                )
            # "padding": kept as-is; __getitem__ appends pad tokens.
        elif total_tokens > self.pack_max_length:
            if self.long_pack_strategy == "error":
                raise ValueError(
                    f"Pack {pack_idx}: total tokens {total_tokens} > pack_max_length {self.pack_max_length}."
                )
            # "truncate": reduce the last slice's max_tokens
            excess = total_tokens - self.pack_max_length
            ds_idx, s_idx, char_start, char_end, tok_off, max_tok = slices[-1]
            new_max_tok = max_tok - excess
            if new_max_tok <= 0:
                raise ValueError(
                    f"Pack {pack_idx}: truncation would make the last slice empty. "
                    "Reduce pack size or adjust pack config."
                )
            slices[-1] = (ds_idx, s_idx, char_start, char_end, tok_off, new_max_tok)

        return slices

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pack_infos)

    def __getitem__(self, i: int) -> list[DataItem]:
        pack = self.pack_infos[i]
        items: list[DataItem] = []

        for ds_idx, s_idx, _char_start, _char_end, _tok_off, max_tokens in pack:
            # TODO: Feature 3 – replace token slicing with DataItem/LongTextDataItem consistency check
            raw_item: DataItem = cast(DataItem, self.datasets[ds_idx][s_idx])
            sliced: DataItem = {
                "input_ids": raw_item["input_ids"][:max_tokens],
                "labels": raw_item["labels"][:max_tokens],
                "num_tokens": min(max_tokens, raw_item["num_tokens"]),
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
