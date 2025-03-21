# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict

from transformers import PreTrainedTokenizer


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
    with open(path, "rb") as f:
        file_hash = hashlib.sha256()
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
