# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import numpy as np
import xxhash
from PIL import Image
from typing_extensions import TypedDict

from .data_item import CacheItem


T = TypeVar("T")

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class CacheObj(TypedDict, total=False):
    num_tokens: int


class CachableTokenizeFunction(ABC, Generic[T]):
    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        self.state = "runtime"

    @abstractmethod
    def __call__(self, item: Any, **kwargs) -> T | CacheItem:
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> str:
        raise NotImplementedError

    def set_state(self, state: Literal["cache", "runtime"]):
        self.state = state


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
