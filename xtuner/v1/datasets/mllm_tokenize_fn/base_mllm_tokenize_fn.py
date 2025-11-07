# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Any, Literal, TypeVar

import torch
import xxhash
from PIL import Image
from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import ChatTemplate, HybridChatTemplate
from xtuner.v1.utils import get_logger

from ..data_item import BaseMLLMDataItem, CacheItem
from ..utils import CachableTokenizeFunction, tokenizer_xxhash


logger = get_logger()

IMAGE_TOKEN_ALIAS = "XTUNER-ALIAS-ALIAS-XTUNER-2025"


def collect_image_video_paths_and_extra(messages: list[dict]):
    image_paths = []
    image_wh_list = []
    video_paths = []
    for msg in messages:
        if msg["role"] == "user" or msg["role"] == "pretrain":
            content = msg["content"]
            if isinstance(content, list):
                for c in content:
                    if c["type"] == "image_url":
                        image_paths.append(c["image_url"]["url"])
                        if "image_wh" in c["image_url"]:
                            image_wh = c["image_url"]["image_wh"]
                            if isinstance(image_wh[0], (list, tuple)):
                                assert len(image_wh) == 1, (
                                    f"Only one image size is supported for each image. but got {image_wh}"
                                )
                                image_wh = image_wh[0]
                            image_wh_list.append(image_wh)
                            assert len(image_wh) == 2, f"image_wh should be [width, height], but got {image_wh}"
                    if c["type"] == "video_url":
                        video_paths.append(c["video_url"]["url"])
    if len(image_wh_list) > 0:
        assert len(image_wh_list) == len(image_paths), "If image_wh is provided, it should match the number of images."
    return image_paths, video_paths, {"image_wh": image_wh_list}


def replace_image_token(
    messages: ChatMessages,
    chat_template: HybridChatTemplate,
    num_image_token_list: list[int],
    add_vision_id: bool = False,
):
    current_image_idx = 0
    for msg in messages.messages:
        if msg.role == "pretrain":
            assert len(messages.messages) == 1, "pretrain message should only have one message"
        if msg.role == "user" or msg.role == "pretrain":
            content = msg.content
            if isinstance(content, list):
                for c in content:
                    if c.type == "text":
                        text = c.text
                        text = text.replace("<IMG_CONTEXT>", IMAGE_TOKEN_ALIAS)
                        image_cnt = text.count(IMAGE_TOKEN_ALIAS)
                        for i in range(image_cnt):
                            image_tokens = f"{chat_template.image_start_token}{chat_template.image_context_token * num_image_token_list[current_image_idx]}{chat_template.image_end_token}"  # type: ignore
                            if add_vision_id and image_cnt > 1:
                                # add vision id for each image when there are multiple images
                                image_tokens = f"Picture {i + 1}: " + image_tokens
                            text = text.replace(IMAGE_TOKEN_ALIAS, image_tokens, 1)
                            current_image_idx += 1
                        c.text = text
    # if current_image_idx < num_image, it means <image> placeholder is less than num_image
    assert current_image_idx == len(num_image_token_list), (
        f"ERROR: current_image_idx: {current_image_idx} != num_image: {len(num_image_token_list)}"
    )


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def get_image_path(image_path: str, media_root: str):
    if image_path.startswith("s3://"):  # for ceph
        image_path = media_root + image_path
    else:  # for local image
        image_path = os.path.join(media_root, image_path)
    return image_path


T = TypeVar("T", bound=BaseMLLMDataItem)


class BaseMLLMTokenizeFunction(CachableTokenizeFunction[T]):
    def __init__(
        self,
        tokenizer,
        chat_template: ChatTemplate,
        max_length: int | None = None,
        tokenizer_hash: str | None = None,
        hash: str | None = None,
    ):
        self.max_length = max_length
        self._tokenizer_hash = tokenizer_hash
        self._hash = hash
        self._hash_str = ""
        self.chat_template = chat_template

        self._image_path: list[str] = []
        self._video_path: list[str] = []
        self._image_wh_list: list[list] = []
        super().__init__(tokenizer)

    def calc_num_tokens_multi_modal_get_item(self, data_item: dict) -> CacheItem:
        raise NotImplementedError

    def multi_modal_get_item(self, data_item: dict, media_root: str = "") -> BaseMLLMDataItem:
        raise NotImplementedError

    def calc_num_tokens_video_get_item(self, data_item: dict) -> CacheItem:
        raise NotImplementedError

    def video_get_item(self, data_item: dict, media_root: str = "") -> BaseMLLMDataItem:
        raise NotImplementedError

    def calc_num_tokens_pure_text_get_item(self, data_item) -> CacheItem:
        messages = ChatMessages(messages=data_item["messages"])
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        input_ids, _ = self._truncated_input_and_labels(input_ids, labels)
        return {"num_tokens": len(input_ids)}

    def pure_text_get_item(self, data_item: Any) -> BaseMLLMDataItem:
        raise NotImplementedError

    def _truncated_input_and_labels(self, input_ids, labels: torch.Tensor | None = None):
        if self.max_length is not None and len(input_ids) > self.max_length:
            logger.info(
                f"WARNING: input_ids length {len(input_ids)} exceeds model_max_length {self.max_length}. truncated!"
            )
            input_ids = input_ids[: self.max_length]
            if labels is not None:
                labels = labels[: self.max_length]
        return input_ids, labels

    def __call__(self, item: dict, media_root: str = "", **kwargs) -> T | CacheItem:  # type: ignore[override]
        self._image_path, self._video_path, extra_info = collect_image_video_paths_and_extra(item["messages"])
        self._image_wh_list = extra_info["image_wh"]
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
    model_config = ConfigDict(
        title="Base dataset config for xtuner",
        extra="forbid",
        protected_namespaces=(),
    )
    system_message: str | None = None
    max_length: int | None = None
    hash: str | None = None
    debug: bool = False
    oss_time_log_thr: int = 10  # 10s
    add_eos_token: bool = True  # for mllm pretrain
    add_bos_token: bool = False  # for mllm pretrain

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs
    ) -> BaseMLLMTokenizeFunction:
        raise NotImplementedError("The 'build' method must be implemented.")


class OSSLoaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["petrel"] = "petrel"
    backend_kwargs: dict = {}
