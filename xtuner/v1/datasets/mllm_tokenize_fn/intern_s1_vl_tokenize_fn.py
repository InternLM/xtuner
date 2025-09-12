# Copyright (c) OpenMMLab. All rights reserved.

import hashlib
import os
from typing import Any

import numpy as np
import torch
import xxhash
from PIL import Image
from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP
from xtuner.v1.datasets.data_item import InternS1DataItem
from xtuner.v1.utils import get_logger

from ..utils import CachableTokenizeFunction, tokenizer_xxhash
from ..vlm_utils import TCSLoader, apply_exif_orientation
from .intern_s1_vl_process import build_transform, dynamic_num_patch, dynamic_preprocess
from .video_utils import read_frames_decord


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


ALIAS = "XTUNER-ALIAS-ALIAS-XTUNER-2025"


class InternS1VLTokenizeFunction(CachableTokenizeFunction[InternS1DataItem]):
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
        tcs_loader: TCSLoader | None = None,
        tokenizer_hash: str | None = None,
        max_length: int | None = None,
        hash: str | None = None,
        only_prompt: bool = False,
    ):
        super().__init__()
        from xtuner.v1.model import InternS1BaseConfig, InternVLBaseConfig

        assert isinstance(model_cfg, (InternS1BaseConfig, InternVLBaseConfig))
        chat_template = "intern-s1"

        self._hash = hash
        self._tokenizer_hash = tokenizer_hash
        self.tcs_loader = tcs_loader
        self.tokenizer = tokenizer
        self.only_prompt = only_prompt
        self.max_length = max_length

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
        self.data_name = anno_name
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
            f"_{self.min_dynamic_patch}_{self.max_dynamic_patch}"
        )

        self.chat_template = CHAT_TEMPLATE_MAP[chat_template]
        if system_message is not None:
            self.chat_template.default_system = system_message

        self._image_path: list[str] = []
        self._video_path: list[str] = []

    def _truncated_input_and_labels(self, input_ids, labels):
        if self.max_length is not None and len(input_ids) > self.max_length:
            logger.info(
                f"WARNING: input_ids length {len(input_ids)} exceeds model_max_length {self.max_length}. truncated!"
            )
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
        return input_ids, labels

    def _collect_image_video_paths(self, messages):
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

    def _load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and "s3://" in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert("RGB")

    def _get_image_path(self, image_path, media_root):
        if image_path.startswith("s3://"):  # for ceph
            image_path = media_root + image_path
        else:  # for local image
            image_path = os.path.join(media_root, image_path)
        return image_path

    def _get_transform(self):
        transform = build_transform(
            is_train=self.data_augment, input_size=self.image_size, pad2square=False, normalize_type="imagenet"
        )
        return transform

    def replace_image_token(self, messages, num_image_token_list):
        current_image_idx = 0
        for msg in messages.messages:
            if msg.role == "user":
                content = msg.content
                if isinstance(content, list):
                    for c in content:
                        if c.type == "text":
                            text = c.text
                            assert "<IMG_CONTEXT>" in text
                            text = text.replace("<IMG_CONTEXT>", ALIAS)
                            image_cnt = text.count(ALIAS)
                            for _ in range(image_cnt):
                                image_tokens = f"{self.chat_template.image_start_token}{self.chat_template.image_context_token * num_image_token_list[current_image_idx]}{self.chat_template.image_end_token}"
                                text = text.replace(ALIAS, image_tokens, 1)
                                current_image_idx += 1
                            c.text = text
            # if current_image_idx < num_image, it means <image> placeholder is less than num_image
            assert current_image_idx == len(num_image_token_list), (
                f"ERROR: current_image_idx: {current_image_idx} != num_image: {len(num_image_token_list)}"
            )

    def replace_video_token(self, messages, num_image_token_list):
        current_image_idx = 0
        n_frames = len(num_image_token_list)
        for msg in messages.messages:
            if msg.role == "user":
                content = msg.content
                if isinstance(content, list):
                    for c in content:
                        if c.type == "text":
                            text = c.text
                            assert "<IMG_CONTEXT>" in text
                            text = text.replace("<IMG_CONTEXT>", ALIAS)
                            image_cnt = text.count(ALIAS)
                            assert image_cnt == 1, "Only one <IMG_CONTEXT> is supported for video."
                            for _ in range(image_cnt):
                                special_tokens = "\n".join(
                                    [f"Frame-{frame_idx + 1}: {ALIAS}" for frame_idx in range(n_frames)]
                                )
                                text = text.replace(ALIAS, special_tokens)
                                image_tokens = f"{self.chat_template.image_start_token}{self.chat_template.image_context_token * num_image_token_list[current_image_idx]}{self.chat_template.image_end_token}"
                                text = text.replace(ALIAS, image_tokens)
                                current_image_idx += n_frames
                            c.text = text
            # if current_image_idx < num_image, it means <image> placeholder is less than num_image
            assert current_image_idx == len(num_image_token_list), (
                f"ERROR: current_image_idx: {current_image_idx} != num_image: {len(num_image_token_list)}"
            )

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

    def pure_text_get_item(self, data_item) -> InternS1DataItem:
        # Build transformation function
        transform = self._get_transform()

        # Create a blank white image
        image = Image.new("RGB", (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=1,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values_list = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values_list)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."

        if isinstance(data_item, dict) and "messages" in data_item:
            message = data_item["messages"]
        else:
            message = data_item
        messages = ChatMessages(messages=message)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)

        ret = InternS1DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            num_tokens=len(input_ids),
            num_img_tokens=[0],
            num_imgs=[0],
            num_patches=[num_patches],
        )
        return ret

    def calc_num_tokens_multi_modal_get_item(self, data_item):
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

        num_tiles = []
        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            for size in image_size:
                num_patches = dynamic_num_patch(
                    size,
                    min_num=self.min_dynamic_patch,
                    max_num=max(1, self.max_dynamic_patch // len(self._image_path)),
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                num_tiles.append(num_patches)
        else:  # Otherwise, use the original image as a single patch
            num_tiles = [1] * len(image_size)

        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]

        if isinstance(data_item, dict) and "messages" in data_item:
            message = data_item["messages"]
        else:
            message = data_item
        messages = ChatMessages(messages=message)

        try:
            self.replace_image_token(messages, num_image_tokens)
            tokenized = messages.tokenize(self.tokenizer, self.chat_template)
            input_ids = tokenized["input_ids"]
            labels = tokenized["labels"]
            input_ids, _ = self._truncated_input_and_labels(input_ids, labels)
            return {"num_tokens": len(input_ids)}
        except Exception as e:
            print(
                f"ERROR of Preprocess function: {e}, data_name: {self.data_name}, "
                # f"conversations: {data_item['conversations']}"
            )
            return {"num_tokens": 0}

    def multi_modal_get_item(self, data_item, media_root) -> InternS1DataItem:
        image_sizes = data_item.get("image_wh", None)
        if image_sizes:
            if not isinstance(image_sizes[0], list):
                image_sizes = [image_sizes]
            if len(image_sizes) != len(self._image_path):
                logger.warning(f"Image sizes {image_sizes} do not match image paths {self._image_path}")
                raise RuntimeError("Image sizes do not match image paths, please check the annotation file.")

        num_tiles = []
        images = []
        for i, image_path_ in enumerate(self._image_path):
            image_path_ = self._get_image_path(image_path_, media_root)
            image = self._load_image(image_path_)
            image = apply_exif_orientation(image)

            if image_sizes is not None:
                image_size = image_sizes[i]
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
        if isinstance(data_item, dict) and "messages" in data_item:
            message = data_item["messages"]
        else:
            message = data_item
        messages = ChatMessages(messages=message)
        self.replace_image_token(messages, num_image_tokens)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
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

    def calc_num_tokens_video_get_item(self, data_item):
        # TODO: 目前只支持一个视频
        # 根据 data_item 生成一个确定性的随机整数
        random_frame_num = generate_random_int_from_dict(data_item, self.min_num_frames, self.max_num_frames)
        # 根据采样的帧数（min_num_frames, max_num_frames+1），计算token数量，实际可能采样不到这么多帧（比如视频一共只有10帧），算出来num_tokens可能会偏大
        n_frames = random_frame_num
        num_tiles = [1] * n_frames
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        if isinstance(data_item, dict) and "messages" in data_item:
            message = data_item["messages"]
        else:
            message = data_item
        messages = ChatMessages(messages=message)

        try:
            self.replace_video_token(messages, num_image_tokens)
            tokenized = messages.tokenize(self.tokenizer, self.chat_template)
            input_ids = tokenized["input_ids"]
            labels = tokenized["labels"]
            input_ids, _ = self._truncated_input_and_labels(input_ids, labels)
            return {"num_tokens": len(input_ids)}
        except Exception as e:
            print(
                f"ERROR of Preprocess function: {e}, data_name: {self.data_name}, "
                # f"conversations: {data_item['conversations']}"
            )
            return {"num_tokens": 0}

    def video_get_item(self, data_item, media_root) -> InternS1DataItem:
        assert len(self._video_path) == 1, "Only one video is supported for now."
        video_path = os.path.join(media_root, self._video_path[0])

        # 根据 data_item 生成一个确定性的随机整数
        random_frame_num = generate_random_int_from_dict(data_item, self.min_num_frames, self.max_num_frames)

        assert video_path.endswith((".mp4", ".avi", ".mov", ".webm", ".mkv", ".ts", ".rmvb", ".flv"))
        image_list = read_frames_decord(
            video_path,
            num_frames=self.max_num_frames,
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

        if isinstance(data_item, dict) and "messages" in data_item:
            message = data_item["messages"]
        else:
            message = data_item
        messages = ChatMessages(messages=message)
        self.replace_video_token(messages, num_image_tokens)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        ret = InternS1DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,  # type: ignore
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_tokens=len(input_ids),
            num_img_tokens=num_image_tokens,
            num_imgs=[len(self._image_path)],
            num_patches=[num_patches],
        )
        return ret

    def __call__(self, item: Any, media_root: str = "") -> InternS1DataItem:  # type: ignore[override]
        if isinstance(item, dict) and "messages" in item:
            message = item["messages"]
        else:
            message = item
        self._image_path, self._video_path = self._collect_image_video_paths(message)
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


class InternS1VLTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    model_cfg: (
        BaseModel  # TODO: (huanghaian)  Using model config protocol rather than directly using InternS1BaseConfig
    )
    max_length: int | None = None
    max_dynamic_patch: int | None = None
    min_dynamic_patch: int | None = None
    min_num_frames: int = 4
    max_num_frames: int = 24
    data_augment: bool = False
    system_message: str | None = None
    hash: str | None = None

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
            hash=self.hash,
        )
