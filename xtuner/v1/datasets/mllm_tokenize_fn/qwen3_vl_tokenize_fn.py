# Copyright (c) OpenMMLab. All rights reserved.

import copy
import os
from collections.abc import Sequence

import torch
from pydantic import ConfigDict

from transformers import AutoProcessor, PreTrainedTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP
from xtuner.v1.utils import get_logger

from ..data_item import CacheItem, QwenVL3DataItem
from ..utils import apply_exif_orientation
from .base_mllm_tokenize_fn import BaseMLLMTokenizeFnConfig, BaseMLLMTokenizeFunction, load_image, replace_image_token
from .qwenvl_rope2d import get_rope_index_3


logger = get_logger()


def smart_get_thw(image_size, image_processor):
    orig_width, orig_height = image_size

    resized_height, resized_width = smart_resize(
        orig_height,
        orig_width,
        factor=image_processor.patch_size * image_processor.merge_size,
        min_pixels=image_processor.size["shortest_edge"],
        max_pixels=image_processor.size["longest_edge"],
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
        min_pixels: int | None = None,  # Max image pixels (H*W) for image
        max_pixels: int | None = None,  # Min image pixels (H*W) for image
        video_min_frames: int = 4,  # Min frames per video
        video_max_frames: int = 8,  # Max frames per video
        base_interval: int = 2,  # Sampling time interval (seconds) between frames
        video_max_total_pixels: int = 1664 * 28 * 28,  # Max pixels within a frame
        video_min_total_pixels: int = 256 * 28 * 28,  # Min pixels within a frame
        system_message: str | None = None,
        add_vision_id: bool = True,
        max_length: int | None = None,
        tokenizer_hash: str | None = None,
        hash: str | None = None,
    ):
        self.image_processor = AutoProcessor.from_pretrained(processor_path).image_processor
        self.video_max_total_pixels = video_max_total_pixels
        self.video_min_total_pixels = video_min_total_pixels
        # default min_pixels 4096=4x32x32=4x16x16x2x2 pix 一张图片 patch size=16x16，然后 merge size=2x2, 最终输出给 llm 占 4 个 token
        # default max_pixels 16777216=16384x32x32 pix 一张图片输出给 llm 会占 16384 个 token
        if min_pixels is not None:
            self.image_processor.size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            self.image_processor.size["longest_edge"] = max_pixels
        self.merge_length = self.image_processor.merge_size**2
        self.add_vision_id = add_vision_id

        self.data_name = os.path.basename(anno_name)
        logger.info(
            f"[{self.data_name}] min_pixels: {self.image_processor.size['shortest_edge']}, "
            f"max_pixels: {self.image_processor.size['longest_edge']},"
            f"video_min_total_pixels: {self.video_min_total_pixels}, "
            f"video_max_total_pixels: {self.video_max_total_pixels}"
        )

        self.chat_template = CHAT_TEMPLATE_MAP["qwen3-vl"]
        if system_message is not None:
            self.chat_template.default_system = system_message

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_context_token)

        # Note: 比较重要，防止改了参数但是没有重新 cache
        self._hash_str = (
            f"{self.image_processor.size['shortest_edge']}_{self.image_processor.size['longest_edge']}_{self.video_min_total_pixels}"
            f"_{self.video_max_total_pixels}_{self.add_vision_id}_{system_message}_{max_length}"
        )

        # 必须要最后调用
        super().__init__(tokenizer, self.chat_template, max_length, tokenizer_hash, hash)

    def _truncated_data_item(
        self, input_ids: list[int], labels: list[int] | None = None, position_ids: torch.Tensor | None = None
    ):
        # 如果 input_ids 超过单条最大长度会被截断，那么 position_ids 也要被截断
        if position_ids is not None:
            assert position_ids.size(-1) == len(input_ids), (
                f"position_ids.shape {position_ids.shape} != len(input_ids) {input_ids}. "
            )
        if self.max_length is not None and len(input_ids) > self.max_length:
            logger.info(
                f"WARNING: input_ids length {len(input_ids)} exceeds model_max_length {self.max_length}. truncated!"
            )
            input_ids = input_ids[: self.max_length]
            if labels is not None:
                labels = labels[: self.max_length]
            if position_ids is not None:
                position_ids = position_ids[..., : self.max_length]
        return input_ids, labels, position_ids

    def process_image_unified(self, image_file: str, media_root: str = ""):
        processor = copy.deepcopy(self.image_processor)
        image = load_image(os.path.join(media_root, image_file))
        image = apply_exif_orientation(image)

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def pure_text_get_item(self, data_item: dict) -> QwenVL3DataItem:
        messages = ChatMessages(messages=data_item["messages"])
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels: list[int] = tokenized["labels"]

        position_ids: torch.Tensor

        position_ids = get_rope_index_3(
            torch.tensor(input_ids).unsqueeze(0),
            spatial_merge_size=self.image_processor.merge_size,
        )

        input_ids, labels, position_ids = self._truncated_data_item(input_ids, labels, position_ids)

        ret = QwenVL3DataItem(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            num_tokens=len(input_ids),
            num_img_tokens=[0],
            num_imgs=[0],
            num_patches=[0],
        )
        return ret

    def calc_num_tokens_multi_modal_get_item(self, data_item: dict) -> CacheItem:
        try:
            assert len(self._image_wh_list) >= 1, "image must have `hw` attribute when packing data"
            for size in self._image_wh_list:
                if size[0] == 0 or size[1] == 0:
                    # Image is corrupted, flag=0, and this data will be removed later
                    return {"num_tokens": 0}  # type: ignore
        except Exception as e:
            print(f"ERROR of image_wh: {e}, data_name: {self.data_name}")
            return {"num_tokens": 0}  # type: ignore

        media_grid_thw = []
        for size in self._image_wh_list:
            media_grid_thw.append(smart_get_thw(size, self.image_processor))
        media_grid_thw = torch.tensor(media_grid_thw, dtype=torch.int).reshape(-1, 3)  # type: ignore
        sum_media_grid_thw = media_grid_thw.prod(dim=1) // self.merge_length  # type: ignore

        messages = ChatMessages(messages=data_item["messages"])
        replace_image_token(messages, self.chat_template, sum_media_grid_thw, add_vision_id=self.add_vision_id)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]

        input_ids, _, _ = self._truncated_data_item(input_ids)

        # 如果图片被截断，则该数据丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.image_token_id).sum()
        num_image_tokens_2 = sum_media_grid_thw.sum()
        if num_image_tokens_1 != num_image_tokens_2:
            logger.warning(
                f"num_image_tokens_1.shape {num_image_tokens_1} != num_image_tokens_2.shape {num_image_tokens_2}, "
                f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
            )
            return {"num_tokens": 0}

        return {"num_tokens": len(input_ids)}

    def multi_modal_get_item(self, data_item: dict, media_root: str = "") -> QwenVL3DataItem:
        results = [self.process_image_unified(file, media_root) for file in self._image_path]
        image, grid_thw = zip(*results)

        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [merged_thw.prod() // self.merge_length for merged_thw in grid_thw_merged]  # type: ignore
        messages = ChatMessages(messages=data_item["messages"])
        replace_image_token(messages, self.chat_template, grid_thw_merged, add_vision_id=self.add_vision_id)  # type: ignore
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]

        position_ids = get_rope_index_3(
            torch.tensor(input_ids).unsqueeze(0),
            spatial_merge_size=self.image_processor.merge_size,
            image_grid_thw=torch.stack(grid_thw, dim=0),
        )

        input_ids, labels, position_ids = self._truncated_data_item(input_ids, labels, position_ids)

        # 如果图片被截断，则该数据要丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.image_token_id).sum()
        num_image_tokens_2 = torch.stack(grid_thw_merged, dim=0).sum()
        # assert 会被捕获，该数据会丢弃
        assert num_image_tokens_1.shape == num_image_tokens_2.shape, (
            f"num_image_tokens_1.shape {num_image_tokens_1.shape} != num_image_tokens_2.shape {num_image_tokens_2.shape}, "
            f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
        )

        num_img_tokens = sum(grid_thw_merged[i].item() + 2 for i in range(len(grid_thw_merged)))

        ret = QwenVL3DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=torch.cat(image, dim=0),  # (n,d)
            image_grid_thw=torch.cat([_thw.unsqueeze(0) for _thw in grid_thw], dim=0),  # b,3
            position_ids=position_ids,
            num_tokens=len(input_ids),
            num_img_tokens=[num_img_tokens],
            num_imgs=[len(self._image_path)],
            num_patches=[0],
        )
        return ret


class Qwen3VLTokenizeFnConfig(BaseMLLMTokenizeFnConfig):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="forbid")
    processor_path: str
    min_pixels: int | None = None
    max_pixels: int | None = None
    video_min_total_pixels: int = 256 * 28 * 28
    video_max_total_pixels: int = 1664 * 28 * 28
    # When handling multiple images, it's helpful to add labels to the images and videos for better reference.
    add_vision_id: bool = True

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
            add_vision_id=self.add_vision_id,
            max_length=self.max_length,
            system_message=self.system_message,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
        )
