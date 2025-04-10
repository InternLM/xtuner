# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict

from transformers import PreTrainedTokenizer

from xtuner._lite import get_logger


class CacheObj(TypedDict, total=False):
    num_tokens: int


class CachableTokenizeFunction(ABC):
    @abstractmethod
    def __call__(self, item: Any) -> CacheObj:
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> str:
        raise NotImplementedError


def calculate_file_sha256(path):
    hash_method = os.environ.get("HASH_METHOD", "sha256")
    assert (
        hash_method in hashlib.algorithms_guaranteed
    ), f"hash method {hash_method} not supported"
    if hash_method != "sha256":
        get_logger().warning(f"Non-default hash method {hash_method} is used")
    hash_method = getattr(hashlib, hash_method)
    with open(path, "rb") as f:
        file_hash = hash_method()
        file_hash.update(f.read())
    return file_hash.hexdigest()


def tokenizer_hash(tokenizer: PreTrainedTokenizer):
    with tempfile.TemporaryDirectory() as tokenizer_tempdir:
        tokenizer.save_pretrained(tokenizer_tempdir)
        tokenizer_files = sorted(Path(tokenizer_tempdir).iterdir())
        file_hash = hashlib.sha256()
        for file in tokenizer_files:
            with open(file, "rb") as f:
                file_hash.update(f.read())
        return file_hash.hexdigest()
