# Copyright (c) OpenMMLab. All rights reserved.

import hashlib
import os
import time
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP, HybridChatTemplate
from xtuner.v1.model import InternS1BaseConfig, InternVLBaseConfig
from xtuner.v1.utils import get_logger

from ..data_item import CacheItem, InternS1DataItem
from ..utils import apply_exif_orientation
from .base_mllm_tokenize_fn import (
    IMAGE_TOKEN_ALIAS,
    BaseMLLMTokenizeFnConfig,
    BaseMLLMTokenizeFunction,
    OSSLoaderConfig,
    get_image_path,
    load_image,
    replace_image_token,
)
from .intern_s1_vl_process import build_transform, dynamic_num_patch, dynamic_preprocess
from .intern_s1_vl_utils import InternS1VLOSSLoader, pil_loader, read_interns1_vl_video


logger = get_logger()


def dict_to_sorted_string(input_dict):
    """Convert a potentially nested dictionary into a sorted string
    representation."""

    def process_value(value):
        if isinstance(value, dict):
            return dict_to_sorted_string(value)
        elif isinstance(value, list):
            return [process_value(v) for v in value]
        return value

    sorted_items = sorted((k, process_value(v)) for k, v in input_dict.items())
    return str(sorted_items)


def generate_random_int_from_dict(input_dict, min_num, max_num):
    """Generate a deterministic random integer based on a nested dictionary
    (using stable hashing)"""
    dict_string = dict_to_sorted_string(input_dict)
    input_bytes = dict_string.encode("utf-8")

    hash_hex = hashlib.md5(input_bytes).hexdigest()
    hash_int = int(hash_hex, 16)

    rng = np.random.default_rng(hash_int)
    return rng.integers(min_num, max_num + 1)


def replace_video_token(messages: ChatMessages, chat_template: HybridChatTemplate, num_image_token_list: list[int]):
    current_image_idx = 0
    n_frames = len(num_image_token_list)
    for msg in messages.messages:
        if msg.role == "pretrain":
            assert len(messages.messages) == 1, "pretrain message should only have one message"
        if msg.role == "user" or msg.role == "pretrain":
            content = msg.content
            if isinstance(content, list):
                for c in content:
                    if c.type == "text":
                        text = c.text
                        # assert "<VIDEO_CONTEXT>" in text
                        text = text.replace("<VIDEO_CONTEXT>", IMAGE_TOKEN_ALIAS)
                        video_cnt = text.count(IMAGE_TOKEN_ALIAS)
                        assert video_cnt == 1, "Only one <VIDEO_CONTEXT> is supported for video."
                        for _ in range(video_cnt):
                            special_tokens = "\n".join(
                                [f"Frame-{frame_idx + 1}: {IMAGE_TOKEN_ALIAS}" for frame_idx in range(n_frames)]
                            )
                            text = text.replace(IMAGE_TOKEN_ALIAS, special_tokens)
                            image_tokens = f"{chat_template.image_start_token}{chat_template.video_context_token * num_image_token_list[current_image_idx]}{chat_template.image_end_token}"  # type: ignore
                            text = text.replace(IMAGE_TOKEN_ALIAS, image_tokens)
                            current_image_idx += n_frames
                        c.text = text
    # if current_image_idx < num_image, it means <image> placeholder is less than num_image
    assert current_image_idx == len(num_image_token_list), (
        f"ERROR: current_image_idx: {current_image_idx} != num_image: {len(num_image_token_list)}"
    )


class InternS1VLTokenizeFunction(BaseMLLMTokenizeFunction[InternS1DataItem]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_cfg,
        anno_name: str,
        max_dynamic_patch: int | None = None,
        min_dynamic_patch: int | None = None,
        min_num_frames: int = 4,
        max_num_frames: int = 24,
        data_augment: bool = False,
        system_message: str | None = None,
        oss_loader_cfg: OSSLoaderConfig | None = None,
        tokenizer_hash: str | None = None,
        max_length: int | None = None,
        hash: str | None = None,
        only_prompt: bool = False,
        template_name: Literal["intern-s1", "internvl-3.5"] = "intern-s1",
        debug: bool = False,
        oss_time_log_thr: int = 10,  # 10s
        add_eos_token: bool = True,  # for mllm pretrain
        add_bos_token: bool = False,  # for mllm pretrain
    ):
        assert isinstance(model_cfg, (InternS1BaseConfig, InternVLBaseConfig))

        self.oss_loader = None
        self.debug = debug
        self.oss_time_log_thr = oss_time_log_thr
        if oss_loader_cfg is not None:
            self.oss_loader = InternS1VLOSSLoader(
                backend=oss_loader_cfg.backend,
                debug=self.debug,
                oss_time_log_thr=self.oss_time_log_thr,
                **oss_loader_cfg.backend_kwargs,
            )

        self.only_prompt = only_prompt

        self.image_size = model_cfg.vision_config.image_size[0]
        self.patch_size = model_cfg.vision_config.patch_size[0]
        if max_dynamic_patch is not None:
            max_num = max_dynamic_patch
        else:
            max_num = model_cfg.max_dynamic_patch
        if min_dynamic_patch is not None:
            min_num = min_dynamic_patch
        else:
            min_num = model_cfg.min_dynamic_patch
        self.max_dynamic_patch = max_num
        self.min_dynamic_patch = min_num
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames

        self.dynamic_image_size = model_cfg.dynamic_image_size
        self.use_thumbnail = model_cfg.use_thumbnail
        self.data_name = os.path.basename(anno_name)
        self.data_augment = data_augment
        logger.info(
            f"[{self.data_name}] Using dynamic image size: {self.dynamic_image_size} and "
            f"max_dynamic_patch: {max_num} and min_dynamic_patch: {min_num} and "
            f"use_thumbnail: {self.use_thumbnail} data_aug: {self.data_augment} for training."
        )
        self.downsample_ratio = model_cfg.downsample_ratio
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (self.downsample_ratio**2))
        self.system_message = system_message

        # Note: 比较重要，防止改了参数但是没有重新 cache
        self._hash_str = (
            f"{self.downsample_ratio}_{self.num_image_token}_{self.system_message}_{self.use_thumbnail}"
            f"_{self.dynamic_image_size}_{self.max_num_frames}_{self.min_num_frames}"
            f"_{self.min_dynamic_patch}_{self.max_dynamic_patch}_{max_length}"
        )

        self.chat_template = CHAT_TEMPLATE_MAP[template_name]
        if system_message is not None:
            self.chat_template.default_system = system_message

        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_start_token)
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_context_token)
        self.video_context_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.video_context_token)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_end_token)

        self.add_eos_token = add_eos_token
        self.add_bos_token = add_bos_token
        self.bos_token_id = None
        if self.add_bos_token and tokenizer.bos_token is None:
            logger.warning("tokenizer has no bos_token, set add_bos_token=False")
            self.add_bos_token = False
        if self.add_bos_token:
            self.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # 必须要最后调用
        super().__init__(tokenizer, self.chat_template, max_length, tokenizer_hash, hash)

    def _get_transform(self):
        transform = build_transform(
            is_train=self.data_augment, input_size=self.image_size, pad2square=False, normalize_type="imagenet"
        )
        return transform

    def pure_text_get_item(self, data_item: dict) -> InternS1DataItem:
        messages = ChatMessages(messages=data_item["messages"])

        is_pretrain = False
        if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
            is_pretrain = True
        assert is_pretrain is False, "Text pretrain data should not be processed by this function"

        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        ret = InternS1DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=torch.randn(1, 3, self.image_size, self.image_size),
            image_flags=torch.tensor([0] * 1, dtype=torch.long),
            num_tokens=len(input_ids),
            num_img_tokens=[0],
            num_imgs=[0],
            num_patches=[1],
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

        num_tiles = []
        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            for size in self._image_wh_list:
                num_patches = dynamic_num_patch(
                    size,
                    min_num=self.min_dynamic_patch,
                    max_num=max(1, self.max_dynamic_patch // len(self._image_path)),
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                num_tiles.append(num_patches)
        else:  # Otherwise, use the original image as a single patch
            num_tiles = [1] * len(self._image_wh_list)

        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]

        messages = ChatMessages(messages=data_item["messages"])

        try:
            replace_image_token(messages, self.chat_template, num_image_tokens)
            tokenized = messages.tokenize(self.tokenizer, self.chat_template)
            input_ids = tokenized["input_ids"]

            is_pretrain = False
            if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
                is_pretrain = True
            if is_pretrain:
                if self.add_bos_token:
                    input_ids = [self.bos_token_id] + input_ids
                if self.add_eos_token:
                    input_ids = input_ids + [self.eos_token_id]

            input_ids, _ = self._truncated_input_and_labels(input_ids)
            assert (torch.tensor(input_ids) == self.img_context_token_id).sum() == sum(num_image_tokens), (
                "ERROR: image tokens are truncated"
            )
            return {"num_tokens": len(input_ids)}
        except Exception as e:
            print(
                f"ERROR of Preprocess function: {e}, data_name: {self.data_name}, "
                # f"conversations: {data_item['conversations']}"
            )
            return {"num_tokens": 0}

    def multi_modal_get_item(self, data_item: dict, media_root: str = "") -> InternS1DataItem:
        num_tiles = []
        images = []
        oss_image_times = 0.0
        for i, image_path_ in enumerate(self._image_path):
            image_path_ = get_image_path(image_path_, media_root)
            if self.oss_loader is not None and "s3://" in image_path_:
                oss_start_time = time.time()
                img_value_str = self.oss_loader.client.get(image_path_)
                oss_image_times += time.time() - oss_start_time
                image = pil_loader(img_value_str)
            else:
                assert "s3://" not in image_path_, "Please use oss_loader_cfg to load image from s3."
                image = load_image(image_path_)
            image = apply_exif_orientation(image)

            if len(self._image_wh_list) >= 1:
                image_size = self._image_wh_list[i]
                if tuple(image_size) != image.size:
                    logger.warning(f"Image size mismatch: {image_size} vs {image.size} for image {image_path_}")
                    raise RuntimeError("Image size mismatch, please check the image file or the annotation file.")

            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=max(1, self.max_dynamic_patch // len(self._image_path)),
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)

        transform = self._get_transform()
        pixel_values_list = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values_list)
        num_patches = pixel_values.size(0)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        messages = ChatMessages(messages=data_item["messages"])
        replace_image_token(messages, self.chat_template, num_image_tokens)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]

        is_pretrain = False
        if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
            is_pretrain = True
        if is_pretrain:
            if self.add_bos_token:
                input_ids = [self.bos_token_id] + input_ids
                labels = [self.bos_token_id] + labels
            if self.add_eos_token:
                input_ids = input_ids + [self.eos_token_id]
                labels = labels + [self.eos_token_id]
            np_labels = np.array(labels)
            np_labels[np_labels == self.img_start_token_id] = -100
            np_labels[np_labels == self.img_context_token_id] = -100
            np_labels[np_labels == self.img_end_token_id] = -100
            labels = np_labels.tolist()

        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        assert (torch.tensor(input_ids) == self.img_context_token_id).sum() == sum(num_image_tokens), (
            "ERROR: image tokens are truncated"
        )

        if self.debug and oss_image_times > self.oss_time_log_thr:
            logger.info(f"[Warning] OSS read {len(self._image_path)} image cost {oss_image_times} seconds")

        ret = InternS1DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_tokens=len(input_ids),
            num_img_tokens=num_image_tokens,
            num_imgs=[len(self._image_path)],
            num_patches=[num_patches],
        )
        return ret

    def calc_num_tokens_video_get_item(self, data_item) -> CacheItem:
        # TODO: 目前只支持一个视频
        # 根据 data_item 生成一个确定性的随机整数
        random_frame_num = generate_random_int_from_dict(data_item, self.min_num_frames, self.max_num_frames)
        # 根据采样的帧数（min_num_frames, max_num_frames+1），计算token数量，实际可能采样不到这么多帧（比如视频一共只有10帧），算出来num_tokens可能会偏大
        n_frames = random_frame_num
        num_tiles = [1] * n_frames
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        messages = ChatMessages(messages=data_item["messages"])

        try:
            replace_video_token(messages, self.chat_template, num_image_tokens)
            tokenized = messages.tokenize(self.tokenizer, self.chat_template)

            input_ids = tokenized["input_ids"]

            is_pretrain = False
            if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
                is_pretrain = True
            if is_pretrain:
                if self.add_bos_token:
                    input_ids = [self.bos_token_id] + input_ids
                if self.add_eos_token:
                    input_ids = input_ids + [self.eos_token_id]

            input_ids, _ = self._truncated_input_and_labels(input_ids)
            assert (torch.tensor(input_ids) == self.video_context_token_id).sum() == sum(num_image_tokens), (
                "ERROR: video tokens are truncated"
            )
            return {"num_tokens": len(input_ids)}
        except Exception as e:
            print(
                f"ERROR of Preprocess function: {e}, data_name: {self.data_name}, "
                # f"conversations: {data_item['conversations']}"
            )
            return {"num_tokens": 0}

    def video_get_item(self, data_item: dict, media_root: str = "") -> InternS1DataItem:
        assert len(self._video_path) == 1, "Only one video is supported for now."
        video_path = os.path.join(media_root, self._video_path[0])

        # 根据 data_item 生成一个确定性的随机整数
        random_frame_num = generate_random_int_from_dict(data_item, self.min_num_frames, self.max_num_frames)

        if self.oss_loader is not None:
            image_list = self.oss_loader(
                video_path,
                image_type="video",
                max_num_frames=self.max_num_frames,
                min_num_frames=self.min_num_frames,
                sample="rand",
                clip=data_item.get("clip", None),
                random_frame_num=random_frame_num,
            )
        else:
            image_list = read_interns1_vl_video(
                video_path,
                max_num_frames=self.max_num_frames,
                min_num_frames=self.min_num_frames,
                sample="rand",
                clip=data_item.get("clip", None),
                random_frame_num=random_frame_num,
            )

        transform = self._get_transform()
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)  # type: ignore
        num_patches = pixel_values.size(0)  # type: ignore
        num_image_tokens = [self.num_image_token] * num_patches

        messages = ChatMessages(messages=data_item["messages"])
        replace_video_token(messages, self.chat_template, num_image_tokens)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]

        is_pretrain = False
        if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
            is_pretrain = True
        if is_pretrain:
            if self.add_bos_token:
                input_ids = [self.bos_token_id] + input_ids
                labels = [self.bos_token_id] + labels
            if self.add_eos_token:
                input_ids = input_ids + [self.eos_token_id]
                labels = labels + [self.eos_token_id]
            np_labels = np.array(labels)
            np_labels[np_labels == self.img_start_token_id] = -100
            np_labels[np_labels == self.video_context_token_id] = -100
            np_labels[np_labels == self.img_end_token_id] = -100
            labels = np_labels.tolist()

        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        assert (torch.tensor(input_ids) == self.video_context_token_id).sum() == sum(num_image_tokens), (
            "ERROR: video tokens are truncated"
        )

        ret = InternS1DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,  # type: ignore
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_tokens=len(input_ids),
            num_img_tokens=num_image_tokens,
            num_imgs=[len(image_list)],
            num_patches=[num_patches],
        )
        return ret

    def __call__(self, item: dict, media_root: str = "", **kwargs) -> InternS1DataItem | CacheItem:
        return super().__call__(item, media_root)


class InternS1VLTokenizeFnConfig(BaseMLLMTokenizeFnConfig):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="forbid")
    model_cfg: (
        BaseModel  # TODO: (huanghaian)  Using model config protocol rather than directly using InternS1BaseConfig
    )
    max_dynamic_patch: int | None = None
    min_dynamic_patch: int | None = None
    min_num_frames: int = 4
    max_num_frames: int = 24
    data_augment: bool = False
    oss_loader_cfg: OSSLoaderConfig | None = None
    template_name: Literal["intern-s1", "internvl-3.5"] = "intern-s1"

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs
    ) -> InternS1VLTokenizeFunction:
        return InternS1VLTokenizeFunction(
            tokenizer,
            self.model_cfg,
            anno_name,
            max_length=self.max_length,
            tokenizer_hash=tokenizer_hash,
            max_dynamic_patch=self.max_dynamic_patch,
            min_dynamic_patch=self.min_dynamic_patch,
            data_augment=self.data_augment,
            system_message=self.system_message,
            min_num_frames=self.min_num_frames,
            max_num_frames=self.max_num_frames,
            oss_loader_cfg=self.oss_loader_cfg,
            template_name=self.template_name,
            hash=self.hash,
            debug=self.debug,
            oss_time_log_thr=self.oss_time_log_thr,
            add_eos_token=self.add_eos_token,  # for mllm pretrain
            add_bos_token=self.add_bos_token,  # for mllm pretrain
        )
