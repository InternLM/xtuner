# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import inspect
from typing import Annotated

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP
from xtuner.v1.datasets.data_item import CacheItem, DataItem
from xtuner.v1.utils import get_logger

from ..utils import CachableTokenizeFunction, tokenizer_xxhash


logger = get_logger()


class OpenaiTokenizeFunction(CachableTokenizeFunction[DataItem]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: str,
        tokenizer_hash: str | None = None,
        max_length: int | None = None,
        hash: str | None = None,
    ):
        assert chat_template in CHAT_TEMPLATE_MAP, (
            f"chat_template {chat_template} not found in {CHAT_TEMPLATE_MAP.keys()}"
        )
        self.chat_template = CHAT_TEMPLATE_MAP[chat_template]
        self._hash = hash
        self._tokenizer_hash = tokenizer_hash
        self.max_length = max_length
        super().__init__(tokenizer)

    def __call__(self, item: dict | list, **kwargs) -> DataItem | CacheItem:
        if isinstance(item, dict) and "messages" in item:
            item = item["messages"]
        tools = None
        if isinstance(item, dict) and "tools" in item:
            tools = item["tools"]
        messages = ChatMessages(messages=item, tools=tools)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)

        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        if self.max_length is not None and len(input_ids) > self.max_length:
            logger.info(
                f"WARNING: input_ids length {len(input_ids)} exceeds model_max_length {self.max_length}. truncated!"
            )
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        return DataItem(
            input_ids=input_ids,
            labels=labels,
            num_tokens=len(input_ids),
        )

    def hash(self) -> str:
        if self._hash is None:
            # truncate to 16 characters prevent too long cache directory
            if self._tokenizer_hash is None:
                _tokenizer_hash = tokenizer_xxhash(self.tokenizer)[:16]
            else:
                _tokenizer_hash = self._tokenizer_hash
            _template_hash = hashlib.sha256(repr(self.chat_template).encode()).hexdigest()[:16]
            _source_hash = (
                # hashlib.sha256(inspect.getsource(ftdp_tokenize).encode()).hexdigest()[:16]
                hashlib.sha256(inspect.getsource(self.__class__.__call__).encode()).hexdigest()[:16]
                + hashlib.sha256(inspect.getsource(self.__class__.__init__).encode()).hexdigest()[:16]
            )

            self._hash = f"{_tokenizer_hash}_{_template_hash}_{_source_hash}"
        else:
            assert isinstance(self._hash, str), (
                "hash is not a valid string, it means `FtdpTokenizeFunction._hash` is modified by user."
            )

        return self._hash


# TODO: Maybe rename
class OpenaiTokenizeFunctionConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="forbid")
    chat_template: Annotated[str, Parameter(group="tokenize_fn")]
    max_length: int | None = None
    hash: Annotated[str | None, Parameter(group="tokenize_fn")] = None

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs
    ) -> OpenaiTokenizeFunction:
        return OpenaiTokenizeFunction(
            tokenizer,
            chat_template=self.chat_template,
            hash=self.hash,
            tokenizer_hash=tokenizer_hash,
            max_length=self.max_length,
        )
