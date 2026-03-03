# Copyright (c) OpenMMLab. All rights reserved.
import atexit
import functools
import hashlib
import os
import random
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Sequence, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import xxhash
from PIL import Image
from xtuner.v1.utils.cache import CacheDict, CacheObj

from .data_item import CacheItem


T = TypeVar("T")

# Shared directory for mmap temp files; all ranks on the same node will use
# the same directory so content-addressed files are reused across processes.
_MMAP_DIR: Path | None = None

# Paths of .npy files created by this process during the current run.
# Tracked so that the atexit handler can clean them up on exit, preventing
# /dev/shm from accumulating files across multiple experiments.
# Only local rank 0 writes files, so only its process will have a non-empty set.
_CREATED_MMAP_FILES: set[Path] = set()


def _cleanup_mmap_files() -> None:
    for path in _CREATED_MMAP_FILES:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_cleanup_mmap_files)


def _get_mmap_dir() -> Path:
    global _MMAP_DIR
    if _MMAP_DIR is None:
        # Prefer /dev/shm (tmpfs, RAM-backed) for lowest-latency mmap;
        # fall back to the system tmp dir.
        for candidate in ("/dev/shm",):
            p = Path(candidate)
            if p.is_dir() and os.access(p, os.W_OK):
                _MMAP_DIR = p / f"xtuner_mmap_{os.getuid()}"
                _MMAP_DIR.mkdir(exist_ok=True)
                return _MMAP_DIR
        _MMAP_DIR = Path(tempfile.mkdtemp(prefix="xtuner_mmap_"))
    return _MMAP_DIR


def ndarray_to_mmap(arr: np.ndarray, group: "dist.ProcessGroup | None" = None) -> np.ndarray:
    """Save *arr* to a shared ``.npy`` file and return a read-only mmap view.

    Only rank 0 writes the file; all other ranks wait at a ``dist.barrier``
    and then open the same file as mmap.  After the caller drops the
    reference to *arr*, the OS reclaims its physical pages and all ranks
    share the single on-disk copy through the page cache.

    Args:
        arr (np.ndarray): The array to convert.
        group (dist.ProcessGroup | None): Process group to use for the barrier.
            Pass a group with a generous timeout when the write may be slow
            (e.g. large arrays on shared storage) to avoid triggering the
            default NCCL watchdog.

    Returns:
        np.ndarray: A read-only memory-mapped view of the same data.
    """
    arr = np.ascontiguousarray(arr)
    content_hash = xxhash.xxh128(arr.data).hexdigest()
    mmap_dir = _get_mmap_dir()
    shape_str = "x".join(str(d) for d in arr.shape)
    npy_path = mmap_dir / f"{content_hash}_{arr.dtype}_{shape_str}.npy"

    if is_local_rank0() and not npy_path.exists():
        # Write to a temp file then atomically rename so other ranks never
        # observe a partially-written file.
        fd, tmp_path = tempfile.mkstemp(suffix=".npy", dir=mmap_dir)
        os.close(fd)
        np.save(tmp_path, arr)
        os.rename(tmp_path, str(npy_path))
        _CREATED_MMAP_FILES.add(npy_path)

    if dist.is_initialized():
        dist.barrier(group=group)

    return np.load(npy_path, mmap_mode="r")


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class CachableTokenizeFunction(ABC, Generic[T]):
    def __init__(self, tokenizer, *args, llm_pack_weight: float = 1.0, visual_pack_weight: float = 0.0, **kwargs):
        self.tokenizer = tokenizer
        self.llm_pack_weight = llm_pack_weight
        self.visual_pack_weight = visual_pack_weight
        self.state = "runtime"

    @abstractmethod
    def __call__(self, item: Any, **kwargs) -> T | CacheItem:
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> str:
        raise NotImplementedError

    def set_state(self, state: Literal["cache", "runtime"]):
        self.state = state

    def proxy_attention_flops(
        self, num_tokens: int, num_img_tokens: list[int], flash_attn_block_size: int = 128
    ) -> float:
        llm_num_patch = (round(num_tokens / flash_attn_block_size)) ** 2
        img_num_patch = sum((n / flash_attn_block_size) ** 2 for n in num_img_tokens)
        return self.llm_pack_weight * llm_num_patch + self.visual_pack_weight * img_num_patch


def calculate_file_sha256(path):
    with open(path, "rb") as f:
        file_hash = hashlib.sha256()
        file_hash.update(f.read())
    return file_hash.hexdigest()


def tokenizer_hash(tokenizer: "PreTrainedTokenizer"):
    with tempfile.TemporaryDirectory() as tokenizer_tempdir:
        tokenizer.save_pretrained(tokenizer_tempdir)
        tokenizer_files = sorted(Path(tokenizer_tempdir).iterdir())
        file_hash = hashlib.sha256()
        for file in tokenizer_files:
            with open(file, "rb") as f:
                file_hash.update(f.read())
        return file_hash.hexdigest()


def tokenizer_xxhash(tokenizer: "PreTrainedTokenizer"):
    with tempfile.TemporaryDirectory() as tokenizer_tempdir:
        tokenizer.save_pretrained(tokenizer_tempdir)
        tokenizer_files = sorted(Path(tokenizer_tempdir).iterdir())
        h = xxhash.xxh64()
        for file in tokenizer_files:
            with open(file, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
        return h.hexdigest()


def calculate_xxhash(data_bytes: bytes | memoryview):
    h = xxhash.xxh64()
    for start in range(0, len(data_bytes), 65536):
        end = min(start + 65536, len(data_bytes))
        h.update(data_bytes[start:end])
    return h.hexdigest()


_EXIF_ORIENT = 274  # exif 'Orientation' tag


def apply_exif_orientation(image):
    """Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def replace_image_context_and_collect_media_data(
    prompt: str | list[dict[str, Any]], media_root: str, replace_image_ctx: bool
) -> tuple:
    """Collect image data from the prompt and extra_info.

    Args:
        prompt (str): The input prompt containing image placeholders.
        media_root (str): The root directory of the media files.
        replace_image_ctx (bool): Whether to replace the image context in the prompt.

    Returns:
        List[dict]: A list of image data dictionaries.
    """
    if not isinstance(prompt, list):
        return [], []

    image_paths = []
    video_paths = []
    for msg in prompt:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                for c in content:
                    if c["type"] == "image_url":
                        image_paths.append(os.path.join(media_root, c["image_url"]["url"]))
                    elif c["type"] == "video_url":
                        video_paths.append(os.path.join(media_root, c["video_url"]["url"]))
                    elif c["type"] == "text":
                        _c = c["text"]
                        if replace_image_ctx:
                            c["text"] = _c.replace("<IMG_CONTEXT>", "")

    return image_paths, video_paths


def dict_to_sorted_string(input_dict: dict[str, Any]) -> str:
    """Convert a potentially nested dictionary into a sorted string
    representation."""

    def process_value(value):
        if isinstance(value, dict):
            return dict_to_sorted_string(value)
        elif isinstance(value, list):
            return [process_value(v) for v in value]
        elif isinstance(value, tuple):
            return tuple(process_value(v) for v in value)
        return value

    sorted_items = sorted((k, process_value(v)) for k, v in input_dict.items())
    return str(sorted_items)


def generate_random_int_from_dict(input_dict: dict[str, Any], min_num: int, max_num: int):
    """Generate a deterministic random integer based on a nested dictionary
    (using stable hashing)"""
    dict_string = dict_to_sorted_string(input_dict)
    input_bytes = dict_string.encode("utf-8")

    hash_hex = hashlib.md5(input_bytes).hexdigest()
    hash_int = int(hash_hex, 16)

    rng = np.random.default_rng(hash_int)
    return rng.integers(min_num, max_num + 1)


def concat_cumulative_sizes_from_lengths(lengths: Sequence[int]) -> np.ndarray:
    """Return cumulative lengths for :class:`torch.utils.data.ConcatDataset`
    (same as ``ConcatDataset.cumsum``)."""

    r: list[int] = []
    s = 0
    for e in lengths:
        s += int(e)
        r.append(s)
    return np.asarray(r, dtype=np.int64)


def get_dataset_id_and_sample_idx_from_idx(idx: int, cumulative_sizes: np.ndarray) -> tuple[int, int]:
    """Map a flat index into concatenated datasets to ``(sub_dataset_id,
    local_sample_idx)``.

    ``cumulative_sizes`` must be the same 1-D int64 array produced by
    :func:`concat_cumulative_sizes_from_lengths` / :meth:`torch.utils.data.ConcatDataset.cumsum`
    for the same sub-dataset lengths (in order).
    """

    idx_i = int(idx)
    total = int(cumulative_sizes[-1])
    if idx_i < 0:
        if -idx_i > total:
            raise ValueError("absolute value of index should not exceed dataset length")
        idx_i = total + idx_i
    sub_id = int(np.searchsorted(cumulative_sizes, idx_i, side="right"))
    if sub_id == 0:
        sample_idx = idx_i
    else:
        sample_idx = idx_i - int(cumulative_sizes[sub_id - 1])
    return sub_id, sample_idx


def _vectorized_flat_to_subdataset_ids(
    flat_idx: np.ndarray, cumulative_sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batched version of :func:`get_dataset_id_and_sample_idx_from_idx`."""

    ccs = np.asarray(cumulative_sizes, dtype=np.int64).reshape(-1)
    idx_i = flat_idx.astype(np.int64, copy=False)
    total = int(ccs[-1])
    if np.any(idx_i < 0):
        idx_i = idx_i.copy()
        neg = idx_i < 0
        neg_vals = -idx_i[neg]
        if np.any(neg_vals > total):
            raise ValueError("absolute value of index should not exceed dataset length")
        idx_i[neg] = total + idx_i[neg]
    sub_id = np.searchsorted(ccs, idx_i, side="right").astype(np.int64, copy=False)
    prev = np.zeros_like(idx_i, dtype=np.int64)
    m = sub_id > 0
    prev[m] = ccs[sub_id[m] - 1]
    sample_idx = idx_i - prev
    return sub_id, sample_idx


def get_longest(boundaries: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Per-pack max sub-sample token length from CSR ``boundaries`` and
    ``samples`` rows.

    Args:
        boundaries: int64, shape ``(num_packs + 1,)`` — CSR pack boundaries.
        samples: int64, shape ``(total_slices, 6)`` — token span is columns 4 and 5.

    Returns:
        New int64 vector of shape ``(num_packs,)`` (empty if ``num_packs == 0``).
    """
    n = int(len(boundaries) - 1)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    b = boundaries.astype(np.int64, copy=False)
    counts = b[1:] - b[:-1]
    tok_lens = samples[:, 5] - samples[:, 4]
    pack_idx = np.repeat(np.arange(n, dtype=np.int64), counts)
    out = np.zeros(n, dtype=np.int64)
    np.maximum.at(out, pack_idx, tok_lens)
    return out


def _pack_config_from_pack_infos_loop(
    pack_infos: dict[str, np.ndarray],
    path_id: int,
    num_tokens: np.ndarray,
    *,
    paths: list[str],
    concat_cumulative_sizes: np.ndarray | None,
) -> dict[str, Any]:
    """Original Python-loop implementation (reference for ``mode='loop'``)."""

    npack = int(pack_infos["dataset_id"].shape[0])
    rows: list[list[int]] = []
    boundaries: list[int] = [0]
    cu, ix = pack_infos["indices_cu_len"], pack_infos["indices"]
    starts, ends = pack_infos["start_offset"], pack_infos["end_offset"]
    for item in range(npack):
        i0 = 0 if item == 0 else int(cu[item - 1])
        i1 = int(cu[item])
        indices = ix[i0:i1]
        s_off, e_off = int(starts[item]), int(ends[item])
        for i, idx in enumerate(indices):
            idx_i = int(idx)
            L = int(num_tokens[idx_i])
            st = 0 if i else s_off
            ed = L if i < len(indices) - 1 else e_off
            if concat_cumulative_sizes is not None:
                row_path_id, sample_idx = get_dataset_id_and_sample_idx_from_idx(idx_i, concat_cumulative_sizes)
            else:
                row_path_id, sample_idx = path_id, idx_i
            rows.append([row_path_id, sample_idx, -1, -1, st, ed])
        boundaries.append(len(rows))
    boundaries_arr = np.asarray(boundaries, dtype=np.int64)
    samples = np.asarray(rows, dtype=np.int64).reshape(-1, 6) if rows else np.empty((0, 6), dtype=np.int64)
    return {
        "boundaries": boundaries_arr,
        "samples": samples,
        "longest": pack_infos["longest"],
        "paths": list(paths),
    }


def _pack_config_from_pack_infos_numpy(
    pack_infos: dict[str, np.ndarray],
    path_id: int,
    num_tokens: np.ndarray,
    *,
    paths: list[str],
    concat_cumulative_sizes: np.ndarray | None,
) -> dict[str, Any]:
    """Vectorized NumPy implementation (``mode='numpy'``)."""

    cu = np.asarray(pack_infos["indices_cu_len"], dtype=np.int64).reshape(-1)
    ix = np.asarray(pack_infos["indices"], dtype=np.int64).reshape(-1)
    n_total = int(ix.shape[0])
    npack = int(cu.shape[0])
    longest = pack_infos["longest"]

    j = np.arange(n_total, dtype=np.int64)
    pack_id = np.digitize(j, cu, right=False).astype(np.int64, copy=False)
    pack_starts = np.empty(npack, dtype=np.int64)
    pack_starts[0] = 0
    if npack > 1:
        pack_starts[1:] = cu[:-1]
    pos_in_pack = j - pack_starts[pack_id]
    pack_len = (cu - pack_starts).astype(np.int64, copy=False)

    starts_arr = np.asarray(pack_infos["start_offset"], dtype=np.int64).reshape(-1)
    ends_arr = np.asarray(pack_infos["end_offset"], dtype=np.int64).reshape(-1)
    st = np.where(pos_in_pack == 0, starts_arr[pack_id], np.int64(0))
    L_all = np.take(np.asarray(num_tokens, dtype=np.int64).reshape(-1), ix, mode="raise")
    is_last = pos_in_pack == (pack_len[pack_id] - 1)
    ed = np.where(is_last, ends_arr[pack_id], L_all)

    samples = np.empty((n_total, 6), dtype=np.int64)
    if concat_cumulative_sizes is not None:
        ccs = np.asarray(concat_cumulative_sizes, dtype=np.int64).reshape(-1)
        samples[:, 0], samples[:, 1] = _vectorized_flat_to_subdataset_ids(ix, ccs)
    else:
        samples[:, 0] = np.int64(path_id)
        samples[:, 1] = ix
    samples[:, 2] = -1
    samples[:, 3] = -1
    samples[:, 4] = st
    samples[:, 5] = ed

    boundaries_arr = np.empty(npack + 1, dtype=np.int64)
    boundaries_arr[0] = 0
    boundaries_arr[1:] = cu

    return {
        "boundaries": boundaries_arr,
        "samples": samples,
        "longest": longest,
        "paths": list(paths),
    }


def get_pack_config_from_pack_infos_by_hard_split(
    pack_infos: dict[str, np.ndarray],
    path_id: int,
    num_tokens: np.ndarray,
    *,
    paths: list[str],
    concat_cumulative_sizes: np.ndarray | None = None,
    mode: Literal["loop", "numpy"] = "numpy",
) -> dict[str, Any]:
    """Same keys as ``get_pack_config_by_simple_hard_split``; built from
    ``get_pack_infos_by_hard_split`` output.

    When ``concat_cumulative_sizes`` is provided, each ``indices`` entry is a flat ``ConcatDataset`` index and is
    converted via :func:`get_dataset_id_and_sample_idx_from_idx` to ``(path_id, sample_idx)`` for the preset NPY.
    Otherwise ``path_id`` is fixed and ``idx`` is written as ``sample_idx`` (single-JsonlDataset case).

    ``paths`` is echoed into the returned dict (for writing ``paths.json`` next to preset NPY files).

    Args:
        mode: ``loop`` — original nested-loop implementation; ``numpy`` — vectorized (default).
    """
    if concat_cumulative_sizes is None:
        assert len(paths) == 1
    else:
        assert len(paths) == len(concat_cumulative_sizes)

    if mode not in ("loop", "numpy"):
        raise ValueError(f"unknown mode: {mode!r}, expected 'loop' or 'numpy'")

    cu = np.asarray(pack_infos["indices_cu_len"], dtype=np.int64).reshape(-1)
    ix = np.asarray(pack_infos["indices"], dtype=np.int64).reshape(-1)
    n_total = int(ix.shape[0])
    longest = pack_infos["longest"]

    if n_total == 0:
        return {
            "boundaries": np.array([0], dtype=np.int64),
            "samples": np.empty((0, 6), dtype=np.int64),
            "longest": longest,
            "paths": list(paths),
        }

    if int(cu[-1]) != n_total:
        raise ValueError(f"len(indices)={n_total} must equal indices_cu_len[-1]={int(cu[-1])}")

    if mode == "loop":
        return _pack_config_from_pack_infos_loop(
            pack_infos, path_id, num_tokens, paths=paths, concat_cumulative_sizes=concat_cumulative_sizes
        )
    return _pack_config_from_pack_infos_numpy(
        pack_infos, path_id, num_tokens, paths=paths, concat_cumulative_sizes=concat_cumulative_sizes
    )


def get_sampler_config(
    order_path: Path,
    *,
    mode: Literal["sequential", "length_grouped"],
    num_packs: int,
    longest: np.ndarray | list | tuple | None = None,
    global_batch_size: int = 2,
    world_size: int = 1,
    seed: int = 0,
    epoch: int = 0,
) -> np.ndarray:
    """Build pack index order and persist to ``order_path`` (``.npy``).

    Args:
        order_path: Output path for a 1-D int64 array of pack indices (same format as training).
        mode:
            ``sequential`` — ``0..num_packs-1`` (legacy / PresetSampler baseline).
            ``length_grouped`` — same megabatch + sort + group shuffle logic as
            :class:`LengthGroupedSampler` via :func:`get_length_grouped_indices`.
        num_packs: Number of valid packs (length of the order permutation).
        longest: Per-pack max token length; required when ``mode=='length_grouped'``.
        global_batch_size: Same as dataloader ``global_batch_size`` (drives megabatch sizing).
        world_size: Same as distributed world size (``group_size`` in length-grouped logic).
        seed / epoch: Match :class:`LengthGroupedSampler` (`seed + epoch` for both RNGs).

    Returns:
        The saved order array.
    """
    from .sampler import LengthGroupedSampler, get_length_grouped_indices  # local import: avoid cycles with packing

    if mode == "sequential":
        order = np.arange(num_packs, dtype=np.int64)
    elif mode == "length_grouped":
        if longest is None:
            raise ValueError("longest is required when mode is 'length_grouped'")
        longest_arr = np.asarray(longest)
        if longest_arr.shape[0] != num_packs:
            raise ValueError(f"len(longest)={longest_arr.shape[0]} must equal num_packs={num_packs}")
        mega_batch_mult = min(
            num_packs // (global_batch_size * LengthGroupedSampler.GROUP_BATCH_FACTOR),
            LengthGroupedSampler.MAX_GROUP_BATCH_SIZE,
        )
        if mega_batch_mult == 0:
            mega_batch_mult = 1
        group_batch_size = mega_batch_mult * global_batch_size
        group_size = world_size
        torch_generator = torch.Generator()
        torch_generator.manual_seed(seed + epoch)
        random_generator = random.Random()
        random_generator.seed(seed + epoch)
        order_list = get_length_grouped_indices(
            max_lengths=longest_arr,
            group_batch_size=group_batch_size,
            group_size=group_size,
            torch_generator=torch_generator,
            random_generator=random_generator,
        )
        order = np.asarray(order_list, dtype=np.int64)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    np.save(order_path, order)
    return order
