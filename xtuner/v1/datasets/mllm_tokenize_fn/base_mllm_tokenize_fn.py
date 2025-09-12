# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Any
import xxhash
from PIL import Image
from pydantic import BaseModel, ConfigDict
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import ChatTemplate
from xtuner.v1.utils import get_logger
from ..utils import CachableTokenizeFunction, tokenizer_xxhash
from ..data_item import BaseMLLMDataItem

logger = get_logger()


def collect_image_video_paths(messages):
    image_paths = []
    video_paths = []
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                for c in content:
                    if c["type"] == "image_url":
                        image_paths.append(c["image_url"])
                    if c["type"] == "video_url":
                        video_paths.append(c["video_url"])
    return image_paths, video_paths


def load_image(image_path):
    # Load the image using tcs_loader if available, otherwise use PIL
    return Image.open(image_path).convert("RGB")


def get_image_path(image_path, media_root):
    if image_path.startswith("s3://"):  # for ceph
        image_path = media_root + image_path
    else:  # for local image
        image_path = os.path.join(media_root, image_path)
    return image_path


class BaseMLLMTokenizeFunction(CachableTokenizeFunction[BaseMLLMDataItem]):
    def __init__(self, tokenizer,
                 chat_template: ChatTemplate | None = None,
                 max_length: int | None = None,
                 tokenizer_hash: str | None = None,
                 hash: str | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length
        self._tokenizer_hash = tokenizer_hash
        self._hash = hash
        self._hash_str = ''

    def calc_num_tokens_multi_modal_get_item(self, item: Any) -> BaseMLLMDataItem:
        raise NotImplementedError

    def multi_modal_get_item(self, item: Any, media_root: str = "") -> BaseMLLMDataItem:
        raise NotImplementedError

    def calc_num_tokens_video_get_item(self, item: Any) -> BaseMLLMDataItem:
        raise NotImplementedError

    def video_get_item(self, item: Any, media_root: str = "") -> BaseMLLMDataItem:
        raise NotImplementedError

    def calc_num_tokens_pure_text_get_item(self, data_item):
        if isinstance(data_item, dict) and "messages" in data_item:
            message = data_item["messages"]
        else:
            message = data_item
        messages = ChatMessages(messages=message)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        input_ids, _ = self._truncated_input_and_labels(input_ids, labels)
        return {"num_tokens": len(input_ids)}

    def pure_text_get_item(self, item: Any) -> BaseMLLMDataItem:
        raise NotImplementedError

    def _truncated_input_and_labels(self, input_ids, labels):
        if self.max_length is not None and len(input_ids) > self.max_length:
            logger.info(
                f"WARNING: input_ids length {len(input_ids)} exceeds model_max_length {self.max_length}. truncated!"
            )
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
        return input_ids, labels

    def __call__(self, item: Any, media_root: str = "") -> BaseMLLMDataItem:  # type: ignore[override]
        if isinstance(item, dict) and "messages" in item:
            message = item["messages"]
        else:
            message = item
        self._image_path, self._video_path = collect_image_video_paths(message)
        if len(self._image_path) > 0:
            if self.state == "cache":
                ret = self.calc_num_tokens_multi_modal_get_item(item)
            else:
                ret = self.multi_modal_get_item(item, media_root)
        elif len(self._video_path) > 0:
            if self.state == "cache":
                ret = self.calc_num_tokens_video_get_item(item)
            else:
                ret = self.video_get_item(item, media_root)
        else:
            if self.state == "cache":
                ret = self.calc_num_tokens_pure_text_get_item(item)
            else:
                ret = self.pure_text_get_item(item)
        return ret

    def hash(self) -> str:
        if self._hash is None:
            # truncate to 16 characters prevent too long cache directory
            if self._tokenizer_hash is None:
                _tokenizer_hash = tokenizer_xxhash(self.tokenizer)[:16]
            else:
                _tokenizer_hash = self._tokenizer_hash
            _init_hash = xxhash.xxh64(self._hash_str.encode()).hexdigest()[:16]
            self._hash = f"{_tokenizer_hash}_{_init_hash}"
        else:
            assert isinstance(self._hash, str), (
                "hash is not a valid string, it means `InternS1TokenizeFunction._hash` is modified by user."
            )

        return self._hash


class BaseMLLMTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    system_message: str | None = None
    max_length: int | None = None
    hash: str | None = None

    def build(
            self, tokenizer, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs
    ) -> BaseMLLMTokenizeFunction:
        return BaseMLLMTokenizeFunction(
            tokenizer,
            max_length=self.max_length,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
        )
