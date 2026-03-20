import copy
import hashlib
import inspect
from typing import Annotated, Union

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.datasets.data_item import CacheItem, DataItem

from ..utils import CachableTokenizeFunction, tokenizer_xxhash


class PretrainTokenizeFunction(CachableTokenizeFunction[DataItem]):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        add_eos_token: bool = True,
        add_bos_token: bool = False,
        tokenizer_hash: str | None = None,
        hash: str | None = None,
    ):
        self.add_eos_token = add_eos_token
        self.add_bos_token = add_bos_token
        self._tokenizer_hash = tokenizer_hash
        self._hash = hash
        super().__init__(tokenizer)

    def __call__(self, item: dict, **kwargs) -> DataItem | CacheItem:
        if "messages" in item:
            messages = item["messages"]
            assert messages[0]["role"] == "pretrain", "messages[0] must be pretrain role"
            text = messages[0]["content"]
        else:
            text = item["content"]

        if self.add_bos_token:
            assert self.tokenizer.bos_token is not None, "tokenizer has no bos_token"
            text = self.tokenizer.bos_token + text
        if self.add_eos_token:
            assert self.tokenizer.eos_token is not None, "tokenizer has no eos_token"
            text = text + self.tokenizer.eos_token

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(input_ids)

        if num_tokens == 0:
            labels = []
        else:
            labels = copy.deepcopy(input_ids)
            labels[0] = -100

        return {"input_ids": input_ids, "labels": labels, "num_tokens": num_tokens}

    def hash(self) -> str:
        if self._hash is None:
            # truncate to 16 characters prevent too long cache directory
            if self._tokenizer_hash is None:
                _tokenizer_hash = tokenizer_xxhash(self.tokenizer)[:16]
            else:
                _tokenizer_hash = self._tokenizer_hash

            _source_hash = (
                hashlib.sha256(inspect.getsource(self.__class__.__call__).encode()).hexdigest()[:16]
                + hashlib.sha256(inspect.getsource(self.__class__.__init__).encode()).hexdigest()[:16]
            )

            self._hash = f"{_tokenizer_hash}_{_source_hash}_{self.add_bos_token}_{self.add_eos_token}"
        else:
            assert isinstance(self._hash, str), (
                "hash is not a valid string, it means `PretrainTokenizeFunction._hash` is modified by user."
            )

        return self._hash


class PretrainTokenizeFunctionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    add_eos_token: Annotated[bool, Parameter(group="tokenize_fn")] = True
    add_bos_token: Annotated[bool, Parameter(group="tokenize_fn")] = False
    hash: Annotated[str | None, Parameter(group="tokenize_fn")] = None

    def build(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        tokenizer_hash: str | None = None,
        anno_name: str | None = None,
        **kwargs,
    ) -> PretrainTokenizeFunction:
        return PretrainTokenizeFunction(
            tokenizer=tokenizer,
            add_eos_token=self.add_eos_token,
            add_bos_token=self.add_bos_token,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
        )
