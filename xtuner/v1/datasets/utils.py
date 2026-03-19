# Copyright (c) OpenMMLab. All rights reserved.
import atexit
import functools
import hashlib
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import numpy as np
import torch.distributed as dist
import xxhash
from PIL import Image

from xtuner.v1.utils import is_local_rank0

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


def with_proxy_attention_flops(call_fn=None, *, flash_attn_block_size: int = 128):
    """Decorator: automatically compute and fill the `proxy_attn_flops` field for CacheItem."""

    def decorator(call_fn):
        @functools.wraps(call_fn)
        def wrapper(self, *args, **kwargs) -> CacheItem | T:
            ret = call_fn(self, *args, **kwargs)
            if self.state == "cache":
                if "num_img_tokens" not in ret:
                    num_img_tokens = [0]
                else:
                    num_img_tokens = ret["num_img_tokens"]
                num_tokens = ret["num_tokens"]
                if isinstance(num_tokens, list):
                    # Chunked mode (LongTextPretrainTokenizeFunction): compute per-chunk flops
                    ret["proxy_attn_flops"] = [
                        self.proxy_attention_flops(nt, num_img_tokens, flash_attn_block_size) for nt in num_tokens
                    ]
                else:
                    ret["proxy_attn_flops"] = self.proxy_attention_flops(
                        num_tokens, num_img_tokens, flash_attn_block_size
                    )
            return ret

        return wrapper

    if call_fn is not None:
        return decorator(call_fn)
    return decorator


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
