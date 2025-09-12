# Copyright (c) OpenMMLab. All rights reserved.

import hashlib
import os
from typing import Any

import numpy as np
import torch
import xxhash
from PIL import Image
from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer, AutoProcessor
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP
from xtuner.v1.utils import get_logger
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from .base_mllm_tokenize_fn import BaseMLLMTokenizeFunction,BaseMLLMTokenizeFnConfig
from ..utils import CachableTokenizeFunction
from ..vlm_utils import TCSLoader, apply_exif_orientation
from . import ALIAS

logger = get_logger()


def smart_get_thw(image_size, image_processor):
    orig_width, orig_height = image_size

    resized_height, resized_width = smart_resize(
        orig_height,
        orig_width,
        factor=image_processor.patch_size * image_processor.merge_size,
        min_pixels=image_processor.min_pixels,
        max_pixels=image_processor.max_pixels,
    )
    grid_t = 1  # 单图
    grid_h, grid_w = resized_height // image_processor.patch_size, resized_width // image_processor.patch_size
    return [grid_t, grid_h, grid_w]


class Qwen3VLTokenizeFunction(BaseMLLMTokenizeFunction):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        processor_path: str,
        anno_name: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        video_max_total_pixels: int = 1664 * 28 * 28,
        video_min_total_pixels: int = 256 * 28 * 28,
        system_message: str | None = None,
        max_length: int | None = None,
        tokenizer_hash: str | None = None,
        hash: str | None = None
    ):

        self.image_processor = AutoProcessor.from_pretrained(processor_path).image_processor
        self.video_max_total_pixels = video_max_total_pixels
        self.video_min_total_pixels = video_min_total_pixels
        # default min_pixels 3136=56x56=28x28x2x2=56x56 pix 一张图片输出给 llm 会占 4 个 token
        # default max_pixels 12845056=28x28x128x128=3584x3584 一张图片输出给 llm 会占 16384 个 token
        if min_pixels is not None:
            self.image_processor.min_pixels = min_pixels
        if max_pixels is not None:
            self.image_processor.max_pixels = max_pixels
        self.image_processor.size["longest_edge"] = self.image_processor.max_pixels
        self.image_processor.size["shortest_edge"] = self.image_processor.min_pixels
        self.merge_length = self.image_processor.merge_size ** 2

        self.data_name = os.path.basename(anno_name)
        logger.info(
            f'[{self.data_name}] min_pixels: {self.image_processor.min_pixels}, max_pixels: {self.image_processor.max_pixels},'
            f'video_min_total_pixels: {self.video_min_total_pixels}, video_max_total_pixels: {self.video_max_total_pixels}')

        self.chat_template = CHAT_TEMPLATE_MAP['qwen3-vl']
        if system_message is not None:
            self.chat_template.default_system = system_message

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_context_token)

        self._image_path: list[str] = []
        self._video_path: list[str] = []
        # 必须要最后调用
        super().__init__(tokenizer, self.chat_template, max_length, tokenizer_hash, hash)

    def calc_num_tokens_multi_modal_get_item(self, data_item: dict) -> dict:
        try:
            assert "image_wh" in data_item, "image must have `hw` attribute when packing data"
            image_size = data_item["image_wh"]  # eta: [[100,120]] or [[100,120],[200,240]]
            if not isinstance(image_size[0], list):
                image_size = [image_size]
            for size in image_size:
                if size[0] == 0 or size[1] == 0:
                    # Image is corrupted, flag=0, and this data will be removed later
                    return {"num_tokens": 0}
        except Exception as e:
            print(f"ERROR of image_wh: {e}, data_name: {self.data_name}")
            return {"num_tokens": 0}

        media_grid_thw = []
        for size in image_size:
            media_grid_thw.append(smart_get_thw(size, self.image_processor))
        media_grid_thw = torch.tensor(media_grid_thw, dtype=torch.int).reshape(-1, 3)
        sum_media_grid_thw = media_grid_thw.prod(dim=1) // self.merge_length


class Qwen3VLTokenizeFnConfig(BaseMLLMTokenizeFnConfig):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    processor_path: str
    min_pixels: int | None = None
    max_pixels: int | None = None
    video_min_total_pixels: int = 256 * 28 * 28
    video_max_total_pixels: int = 1664 * 28 * 28

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs
    ) -> Qwen3VLTokenizeFunction:
        return Qwen3VLTokenizeFunction(
            tokenizer,
            self.processor_path,
            anno_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            video_min_total_pixels=self.video_min_total_pixels,
            video_max_total_pixels=self.video_max_total_pixels,
            max_length=self.max_length,
            system_message=self.system_message,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
        )
