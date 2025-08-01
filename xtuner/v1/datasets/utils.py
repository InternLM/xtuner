# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import xxhash

from ..datasets.data_item import DataItem


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class CacheObj(TypedDict, total=False):
    num_tokens: int


class CachableTokenizeFunction(ABC):
    def __init__(self, *args, **kwargs):
        self.state = "runtime"

    @abstractmethod
    def __call__(self, item: Any, **kwargs) -> DataItem:
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
