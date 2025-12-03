# Copyright (c) OpenMMLab. All rights reserved.

import copy
import math
import os
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import torch
from packaging import version
from pydantic import ConfigDict

import transformers
from transformers import AutoProcessor, PreTrainedTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP, HybridChatTemplate
from xtuner.v1.utils import get_logger

from ..data_item import CacheItem, QwenVL3DataItem
from ..utils import apply_exif_orientation, generate_random_int_from_dict
from .base_mllm_tokenize_fn import (
    IMAGE_TOKEN_ALIAS,
    BaseMLLMTokenizeFnConfig,
    BaseMLLMTokenizeFunction,
    OSSLoaderConfig,
    get_image_path,
    load_image,
    replace_image_token,
)
from .qwen3_vl_utils import Qwen3VLOSSLoader, read_qwen3_vl_video
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


def video_smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def sample_frames(
    origin_total_num_frames: int,
    origin_fps: Union[int, float],
    num_frames: Optional[int] = None,
    fps: Union[int, float] = 2,
    min_frames: int = 4,
    max_frames: int = 16,
):
    total_num_frames = origin_total_num_frames

    # 如果没有给定 num_frames，则根据 fps 计算 num_frames，然后均匀采样
    # 如果给了 num_frames，则均匀采样 num_frames 个帧
    if num_frames is None:
        num_frames = int(total_num_frames / origin_fps * fps)
        num_frames = min(max(num_frames, min_frames), max_frames, total_num_frames)
    num_frames = max(num_frames, min_frames)  # 额外保证

    indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)

    return indices


def calculate_timestamps(
    indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2, timestamps: list[float] | None = None
):
    if not isinstance(indices, list):
        indices = indices.tolist()
    if len(indices) % merge_size != 0:
        indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        if timestamps is not None:
            timestamps.extend(timestamps[-1] for _ in range(merge_size - len(timestamps) % merge_size))

    if timestamps is None:
        timestamps = [idx / video_fps for idx in indices]
    else:
        assert len(timestamps) == len(indices), "timestamps should have the same length as indices"
    # @JJJYmmm frames are merged by self.merge_size, \
    # so we need to average the timestamps between the first/last frame within the temporal patch
    timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)]
    return indices, timestamps


def replace_video_token(
    messages: ChatMessages,
    chat_template: HybridChatTemplate,
    num_image_token_list: list[list[int]],
    timestamps_list: list[list[float]],
    add_vision_id: bool = False,
):
    current_image_idx = 0
    n_video = len(num_image_token_list)
    n_image = sum([len(num_image_token_list[i]) for i in range(n_video)])
    if len(timestamps_list) > 0:
        assert len(timestamps_list) == len(num_image_token_list), (
            "timestamps should have the same length as num_image_token_list"
        )

    for msg in messages.messages:
        if msg.role == "pretrain":
            assert len(messages.messages) == 1, "pretrain message should only have one message"
        if msg.role == "user" or msg.role == "pretrain":
            content = msg.content
            if isinstance(content, list):
                for c in content:
                    if c.type == "text":
                        text = c.text
                        video_cnt = text.count("<VIDEO_CONTEXT>")

                        # 如果存在 conversation_timestamps，则拼接到每个 <VIDEO_CONTEXT> 后面
                        if c.conversation_timestamps is not None:
                            conversation_timestamps = c.conversation_timestamps
                            if not isinstance(conversation_timestamps[0], list):
                                conversation_timestamps = [conversation_timestamps]  # type: ignore
                            assert len(conversation_timestamps) == video_cnt
                            text = text.replace("<VIDEO_CONTEXT>", IMAGE_TOKEN_ALIAS)
                            for i in range(video_cnt):
                                start_time = conversation_timestamps[i][0]  # type: ignore
                                end_time = conversation_timestamps[i][1]  # type: ignore
                                timestamps = f"<{start_time:.1f}-{end_time:.1f} seconds>"
                                text = text.replace(IMAGE_TOKEN_ALIAS, f"<VIDEO_CONTEXT>{timestamps}", 1)

                        if add_vision_id and video_cnt > 1:
                            # 标记每个视频
                            text = text.replace("<VIDEO_CONTEXT>", IMAGE_TOKEN_ALIAS)
                            for i in range(video_cnt):
                                video_index = f"Video {i + 1}: "
                                text = text.replace(IMAGE_TOKEN_ALIAS, f"{video_index}<VIDEO_CONTEXT>", 1)

                        text = text.replace("<VIDEO_CONTEXT>", IMAGE_TOKEN_ALIAS)
                        video_cnt = text.count(IMAGE_TOKEN_ALIAS)
                        for i in range(video_cnt):
                            n_frames = len(num_image_token_list[i])

                            # 在一个 video 中，每一帧的 image_token 应该是完全一样，因此直接 num_image_token_list[i][0] 就行
                            image_tokens = f"{chat_template.image_start_token}{chat_template.video_context_token * num_image_token_list[i][0]}{chat_template.image_end_token}"  # type: ignore
                            video_placeholder = ""
                            for frame_idx in range(n_frames):
                                if len(timestamps_list) > 0:
                                    if timestamps_list[i] is not None:
                                        curr_time = timestamps_list[i][frame_idx]
                                        video_placeholder += f"<{curr_time:.1f} seconds>{image_tokens}"
                                else:
                                    video_placeholder += f"{image_tokens}"
                            text = text.replace(IMAGE_TOKEN_ALIAS, video_placeholder, 1)
                            current_image_idx += len(num_image_token_list[i])
                        c.text = text
    assert current_image_idx == n_image, f"VIDEO ERROR: total_image_idx: {current_image_idx} != {n_image}"


class Qwen3VLTokenizeFunction(BaseMLLMTokenizeFunction):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        processor_path: str,
        anno_name: str,
        min_pixels: int | None = None,  # Max image pixels (H*W) for image
        max_pixels: int | None = None,  # Min image pixels (H*W) for image
        video_min_frames: int | None = None,  # Min frames per video
        video_max_frames: int | None = None,  # Max frames per video
        # Max frames per video for random sampling when origin_video_length is None
        # 当用户没有提供 origin_video_length 和 origin_fps 时候，不能采用 video_max_frames 值
        # 因为默认的 video_max_frames 太大了，会导致采样的视频帧太多，导致显存不足
        rand_video_max_frames: int = 24,
        fps: int | None = None,  # Sampling time interval (seconds) between frames
        video_max_total_pixels: int | None = None,  # Max pixels within a frame
        video_min_total_pixels: int | None = None,  # Min pixels within a frame
        system_message: str | None = None,
        add_vision_id: bool = True,
        max_length: int | None = None,
        oss_loader_cfg: OSSLoaderConfig | None = None,
        debug: bool = False,
        oss_time_log_thr: int = 10,  # 10s
        tokenizer_hash: str | None = None,
        hash: str | None = None,
        add_eos_token: bool = True,  # for mllm pretrain
        add_bos_token: bool = False,  # for mllm pretrain
    ):
        self.oss_loader = None
        self.debug = debug
        self.oss_time_log_thr = oss_time_log_thr
        if oss_loader_cfg is not None:
            self.oss_loader = Qwen3VLOSSLoader(
                backend=oss_loader_cfg.backend,
                debug=self.debug,
                oss_time_log_thr=self.oss_time_log_thr,
                **oss_loader_cfg.backend_kwargs,
            )
        version_str = transformers.__version__
        if version.parse(version_str) < version.parse("4.57.0"):
            raise ValueError(f"请升级 transformers 到 4.57.0 及其以上版本，当前版本为 {version_str}")

        _processor = AutoProcessor.from_pretrained(processor_path)
        self.image_processor = _processor.image_processor
        self.video_processor = _processor.video_processor
        # default min_pixels 4096=4x32x32=4x16x16x2x2 pix 一张图片 patch size=16x16，然后 merge size=2x2, 最终输出给 llm 占 4 个 token
        # default max_pixels 16777216=16384x32x32 pix 一张图片输出给 llm 会占 16384 个 token
        if min_pixels is not None:
            self.image_processor.size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            self.image_processor.size["longest_edge"] = max_pixels
        # default video_min_total_pixels 4096=4x32x32 整个视频输出给 llm 会占 4 个 token
        # default video_max_total_pixels 25165824=20480x32x32 整个视频输出给 llm 会占 20480 个 token
        if video_min_total_pixels is not None:
            self.video_processor.size["shortest_edge"] = video_min_total_pixels
        if video_max_total_pixels is not None:
            self.video_processor.size["longest_edge"] = video_max_total_pixels
        if video_min_frames is not None:  # default 4
            self.video_processor.min_frames = video_min_frames
        if video_max_frames is not None:  # default 768
            self.video_processor.max_frames = video_max_frames
        if fps is not None:  # default 2
            self.video_processor.fps = fps

        self.merge_length = self.image_processor.merge_size**2
        self.add_vision_id = add_vision_id
        self.rand_video_max_frames = rand_video_max_frames
        assert self.video_processor.min_frames <= rand_video_max_frames <= self.video_processor.max_frames, (
            f"rand_video_max_frames: {rand_video_max_frames} must be less than {self.video_processor.min_frames} or "
            f"equal to video_max_frames: {self.video_processor.max_frames}"
        )
        self.data_name = os.path.basename(anno_name)
        logger.info(
            f"[{self.data_name}] min_pixels: {self.image_processor.size['shortest_edge']}, "
            f"max_pixels: {self.image_processor.size['longest_edge']}, "
            f"video_min_total_pixels: {self.video_processor.size['shortest_edge']}, "
            f"video_max_total_pixels: {self.video_processor.size['longest_edge']}, "
            f"video_min_frames: {self.video_processor.min_frames}, "
            f"video_max_frames: {self.video_processor.max_frames}, fps: {self.video_processor.fps}, "
            f"rand_video_max_frames: {self.rand_video_max_frames}"
        )

        self.chat_template = CHAT_TEMPLATE_MAP["qwen3-vl"]
        if system_message is not None:
            self.chat_template.default_system = system_message

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_context_token)
        self.video_context_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.video_context_token)
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_start_token)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_end_token)

        # Note: 比较重要，防止改了参数但是没有重新 cache
        self._hash_str = (
            f"{self.image_processor.size['shortest_edge']}_{self.image_processor.size['longest_edge']}_"
            f"{self.video_processor.size['shortest_edge']}"
            f"_{self.video_processor.size['longest_edge']}_{self.video_processor.min_frames}_"
            f"{self.video_processor.max_frames}_{self.video_processor.fps}_"
            f"{self.add_vision_id}_{system_message}_{max_length}_{self.rand_video_max_frames}"
        )

        self.size = SimpleNamespace(**self.video_processor.size)

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
        super().__init__(tokenizer, self.chat_template, max_length, tokenizer_hash, hash, data_name=self.data_name)

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
        image_path_ = get_image_path(image_file, media_root)
        if self.oss_loader is not None and "s3://" in image_path_:
            image = self.oss_loader.client.get(image_path_)
        else:
            assert "s3://" not in image_path_, "Please use oss_loader_cfg to load image from s3."
            image = load_image(image_path_)
        image = apply_exif_orientation(image)
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def pure_text_get_item(self, data_item: dict) -> QwenVL3DataItem:
        messages = ChatMessages(messages=data_item["messages"])

        is_pretrain = False
        if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
            is_pretrain = True
        assert is_pretrain is False, "Text pretrain data should not be processed by this function"

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

        # 图片宽高比可能不符合 qwen3vl 要求
        try:
            media_grid_thw = []
            for size in self._image_wh_list:
                media_grid_thw.append(smart_get_thw(size, self.image_processor))
            media_grid_thw = torch.tensor(media_grid_thw, dtype=torch.int).reshape(-1, 3)  # type: ignore
            sum_media_grid_thw = media_grid_thw.prod(dim=1) // self.merge_length  # type: ignore
        except ValueError as e:
            print(f"ERROR of {self._image_wh_list}: {e}, data_name: {self.data_name}")
            return {"num_tokens": 0}  # type: ignore

        messages = ChatMessages(messages=data_item["messages"])
        replace_image_token(messages, self.chat_template, sum_media_grid_thw, add_vision_id=self.add_vision_id)
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

        input_ids, _, _ = self._truncated_data_item(input_ids)

        # 如果图片被截断，则该数据丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.img_context_token_id).sum()
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

        position_ids = get_rope_index_3(
            torch.tensor(input_ids).unsqueeze(0),
            spatial_merge_size=self.image_processor.merge_size,
            image_grid_thw=torch.stack(grid_thw, dim=0),
        )

        input_ids, labels, position_ids = self._truncated_data_item(input_ids, labels, position_ids)

        # 如果图片被截断，则该数据要丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.img_context_token_id).sum()
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
        )
        return ret

    def calc_frame_info(self, data_item):
        """视频处理逻辑比较复杂，需要特意说明.

        1. 对于 video 数据，建议用户提供 origin_video_length 和 origin_fps 这两个字段。如果不存在，则会基于预设的 rand_video_max_frames 参数
        随机采样，并且最终数据不会带任何时间戳。不过每个 video 的 image_wh 在 qwen3vl 中是必备的

        2. 如果仅仅存在上述两个字段，那么默认会基于这 2 个字段和用户指定的 fps 来采样视频帧，并且会在每个 <VIDEO_CONTEXT> 前面加上每一帧的时间戳

        3. 如果除了上述两个字段还存在 processed_video_length 和 processed_fps 这两个字段，那么
            a. 如果 processed_fps 可以整除用户指定的 fps(可以降采样)，则重新计算新的 fps=processed_fps//fps，然后基于 fps 来采样视频帧，并且会在每个 <VIDEO_CONTEXT> 前面加上每一帧的时间戳
            b. 如果不满足上述情况，则忽略用户传入的 fps 参数：
                a. 如果处理后的视频长度不超过 rand_video_max_frames，则直接全部使用，并基于这些信息算出每一帧的时间戳，追加到 <VIDEO_CONTEXT> 前面
                b. 如果处理后的视频长度超过 rand_video_max_frames，则会均匀采样随机帧数，并基于这些信息算出每一帧的时间戳，追加到 <VIDEO_CONTEXT> 前面

        4. 如果处理上述字段还额外存在 frames_timestamp，则不需要自己算每一帧的时间戳，则是直接用这个信息重复 3 的计算过程

        5. 除了上述外，如果还额外存在 conversation_timestamps，则将这个内容直接追加到每个 <VIDEO_CONTEXT> 后面
        """
        num_frames_indices_list = []
        origin_fps_list = []
        timestamps_list = []
        if len(self._video_extra_info_list) > 0:
            for video_extra_info in self._video_extra_info_list:
                # 这两个对象必然要存在
                origin_fps = video_extra_info["origin_fps"]
                origin_fps_list.append(origin_fps)
                origin_video_length = video_extra_info["origin_video_length"]

                processed_video_length = video_extra_info.get("processed_video_length")
                processed_fps = video_extra_info.get("processed_fps")
                # 两个要不都存在，要么都不存在
                assert (processed_video_length is None) == (processed_fps is None), (
                    f"processed_video_length and processed_fps must both exist or both not exist, "
                    f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
                )
                frames_timestamp = video_extra_info.get("frames_timestamp")

                if processed_video_length is None:
                    # 如果仅仅存在 origin_video_length 和 origin_fps, 基于这 2 个字段和用户指定的 fps 来采样视频帧,并计算时间戳
                    indices = sample_frames(
                        origin_total_num_frames=origin_video_length,
                        origin_fps=origin_fps,
                        fps=self.video_processor.fps,
                        min_frames=self.video_processor.min_frames,
                        max_frames=self.video_processor.max_frames,
                    )
                    indices, timestamps = calculate_timestamps(
                        indices, origin_fps, merge_size=self.video_processor.merge_size
                    )
                else:
                    assert processed_fps is not None
                    if frames_timestamp is not None:
                        assert len(frames_timestamp) == processed_video_length, (
                            f"frames_timestamp must have the same length as processed_video_length, "
                            f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
                        )

                    if processed_fps % self.video_processor.fps == 0:
                        # 如果 processed_fps 是 fps 的整数倍, 则允许再次进行降采样
                        indices = sample_frames(
                            origin_total_num_frames=processed_video_length,
                            origin_fps=processed_fps,
                            fps=self.video_processor.fps,
                            min_frames=self.video_processor.min_frames,
                            max_frames=self.video_processor.max_frames,
                        )
                        if frames_timestamp is not None:
                            frames_timestamp = [frames_timestamp[i] for i in indices]
                        indices, timestamps = calculate_timestamps(
                            indices,
                            processed_fps,
                            merge_size=self.video_processor.merge_size,
                            timestamps=frames_timestamp,
                        )
                    else:
                        if processed_video_length > self.rand_video_max_frames:
                            # 均匀采样 rand_video_max_frames 帧
                            indices = sample_frames(
                                origin_total_num_frames=processed_video_length,
                                origin_fps=processed_fps,
                                num_frames=self.rand_video_max_frames,
                            )
                            if frames_timestamp is not None:
                                frames_timestamp = [frames_timestamp[i] for i in indices]
                            indices, timestamps = calculate_timestamps(
                                indices,
                                processed_fps,
                                merge_size=self.video_processor.merge_size,
                                timestamps=frames_timestamp,
                            )
                        else:
                            # 如果处理后的视频长度不超过 rand_video_max_frames，则直接全部使用，并基于这些信息算出每一帧的时间戳，追加到 <VIDEO_CONTEXT> 前面
                            indices = list(range(processed_video_length))
                            indices, timestamps = calculate_timestamps(
                                indices,
                                processed_fps,
                                merge_size=self.video_processor.merge_size,
                                timestamps=frames_timestamp,
                            )
                timestamps_list.append(timestamps)
                num_frames_indices_list.append(indices)

        if len(num_frames_indices_list) == 0:
            # 如果 self._video_extra_info_list 啥都没有,则完全随机采样
            # 根据采样的帧数（min_num_frames, max_num_frames+1），计算token数量，实际可能采样不到这么多帧（比如视频一共只有10帧），算出来num_tokens可能会偏大
            for video_path in self._video_path:
                num_frames = generate_random_int_from_dict(
                    {"data_item": data_item, "video_path": video_path},
                    self.video_processor.min_frames,
                    self.rand_video_max_frames,
                )
                # 提前确保一定可以被 merge_size 整除
                if num_frames % self.video_processor.merge_size != 0:
                    num_frames += self.video_processor.merge_size - num_frames % self.video_processor.merge_size
                num_frames_indices_list.append(int(num_frames))  # 特殊情况
        if len(timestamps_list) > 0:
            assert len(num_frames_indices_list) == len(timestamps_list), (
                "num_frames_list and timestamps_list should have the same length"
            )
            for num_frames_indices, timestamps in zip(num_frames_indices_list, timestamps_list):
                assert len(num_frames_indices) == len(timestamps) * 2

        if len(origin_fps_list) > 0:
            assert len(origin_fps_list) == len(num_frames_indices_list), (
                "origin_fps_list and num_frames_indices_list should have the same length"
            )
        for num_frames_indices in num_frames_indices_list:
            if isinstance(num_frames_indices, list):
                assert len(num_frames_indices) % self.video_processor.merge_size == 0, (
                    "num_frames must be divisible by merge_size"
                )
            else:
                assert isinstance(num_frames_indices, int), (
                    f"num_frames_indices must be int {type(num_frames_indices)}"
                )
                assert num_frames_indices % self.video_processor.merge_size == 0, (
                    "num_frames must be divisible by merge_size"
                )
        return num_frames_indices_list, origin_fps_list, timestamps_list

    def calc_num_tokens_video_get_item(self, data_item: dict) -> CacheItem:
        assert len(self._video_wh_list) >= 1, "video wh list must be non-empty"
        frames_indices_list, _, timestamps_list = self.calc_frame_info(data_item)
        num_image_token_list = []
        total_sum_media_grid_thw = 0
        for i, wh in enumerate(self._video_wh_list):
            height, width = wh
            if isinstance(frames_indices_list[i], int):
                num_frames = frames_indices_list[i]
            else:
                num_frames = len(frames_indices_list[i])
            # 图片宽高比可能不符合 qwen3vl 要求
            try:
                resized_height, resized_width = video_smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=self.video_processor.temporal_patch_size,
                    factor=self.video_processor.patch_size * self.video_processor.merge_size,
                    min_pixels=self.size.shortest_edge,
                    max_pixels=self.size.longest_edge,
                )
            except ValueError as e:
                print(f"ERROR of {self._video_wh_list}: {e}, data_name: {self.data_name}")
                return {"num_tokens": 0}  # type: ignore

            assert num_frames % self.video_processor.merge_size == 0, "num_frames must be divisible by merge_size"

            grid_t = num_frames // self.video_processor.temporal_patch_size
            grid_h, grid_w = (
                resized_height // self.video_processor.patch_size,
                resized_width // self.video_processor.patch_size,
            )
            sum_media_grid_thw = grid_t * grid_h * grid_w // self.merge_length
            frame_seqlen = grid_h * grid_w // self.merge_length
            num_image_token_list.append([frame_seqlen] * grid_t)
            total_sum_media_grid_thw += sum_media_grid_thw

        messages = ChatMessages(messages=data_item["messages"])
        replace_video_token(
            messages,
            self.chat_template,
            num_image_token_list,
            timestamps_list=timestamps_list,
            add_vision_id=self.add_vision_id,
        )
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

        input_ids, _, _ = self._truncated_data_item(input_ids)

        # 如果图片被截断，则该数据丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.video_context_token_id).sum()
        num_image_tokens_2 = total_sum_media_grid_thw
        if num_image_tokens_1 != num_image_tokens_2:
            logger.warning(
                f"num_video_tokens_1 {num_image_tokens_1} != num_video_tokens_2 {num_image_tokens_2}, "
                f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
            )
            return {"num_tokens": 0}

        return {"num_tokens": len(input_ids)}

    def video_get_item(self, data_item: dict, media_root: str = "") -> QwenVL3DataItem:
        num_image_tokens_list = []
        pixel_values_list = []
        num_imgs_list = []
        total_sum_media_grid_thw = 0
        grid_thw_list = []

        frames_indices_list, origin_fps_list, timestamps_list = self.calc_frame_info(data_item)

        for i, video_path in enumerate(self._video_path):
            frames_indices = frames_indices_list[i]
            timestamps = None
            if len(timestamps_list) > 0:
                timestamps = timestamps_list[i]

            video_path = os.path.join(media_root, video_path)
            if len(self._video_extra_info_list)>0:
                video_extra_dict = self._video_extra_info_list[i]
            else:
                video_extra_dict = None

            if self.oss_loader is not None:
                image_list, frame_indices, timestamps = self.oss_loader(
                    video_path, image_type="video", frames_indices=frames_indices, timestamps=timestamps,
                    video_extra_dict=video_extra_dict
                )
            else:
                image_list, frame_indices, timestamps = read_qwen3_vl_video(
                    video_path, frames_indices=frames_indices, timestamps=timestamps,
                    video_extra_dict=video_extra_dict
                )

            assert len(image_list) % self.video_processor.merge_size == 0, "num_frames must be divisible by merge_size"
            assert len(frame_indices) % self.video_processor.merge_size == 0, (
                "num_frames must be divisible by merge_size"
            )
            # timestamps 可能本身不是 0，但是因为数据错误，返回可能变成 0
            if len(timestamps_list) > 0:
                if timestamps is not None:
                    assert len(timestamps) * 2 == len(image_list) == len(frame_indices)
                assert len(frame_indices) % self.video_processor.merge_size == 0, (
                    "num_frames must be divisible by merge_size"
                )
                timestamps_list[i] = timestamps  # 记得要重新更新

            video_data = torch.stack(image_list)  # num_patch,3,h,w
            num_frames = len(image_list)

            # 前面保证了一定可以被 merge_size 整除，因此内部一定不会额外 padding
            video_result = self.video_processor._preprocess(
                [video_data],
                size=self.size,
                image_mean=tuple(self.video_processor.image_mean),
                image_std=tuple(self.video_processor.image_std),
                patch_size=self.video_processor.patch_size,
                temporal_patch_size=self.video_processor.temporal_patch_size,
                merge_size=self.video_processor.merge_size,
                return_tensors="pt",
            )
            image_tensor = video_result["pixel_values_videos"]
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0]
            pixel_values_list.append(image_tensor)

            grid_thw = video_result["video_grid_thw"]  # (1,3)
            grid_thw_list.append(grid_thw)

            sum_media_grid_thw = grid_thw.prod() // self.merge_length
            frame_seqlen = grid_thw[0][1:].prod() // self.merge_length

            # 作为验证
            height, width = self._video_wh_list[i]
            resized_height, resized_width = video_smart_resize(
                num_frames=num_frames,
                height=height,
                width=width,
                temporal_factor=self.video_processor.temporal_patch_size,
                factor=self.video_processor.patch_size * self.video_processor.merge_size,
                min_pixels=self.video_processor.size["shortest_edge"],
                max_pixels=self.video_processor.size["longest_edge"],
            )
            grid_t = num_frames // self.video_processor.temporal_patch_size
            grid_h, grid_w = (
                resized_height // self.video_processor.patch_size,
                resized_width // self.video_processor.patch_size,
            )
            sum_media_grid_thw_check = grid_t * grid_h * grid_w // self.merge_length
            assert sum_media_grid_thw == sum_media_grid_thw_check, (
                f"sum_media_grid_thw {sum_media_grid_thw} != sum_media_grid_thw_check {sum_media_grid_thw_check}, "
                f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
            )
            num_image_tokens_list.append([frame_seqlen] * grid_thw[0][0])
            num_imgs_list.append(num_frames)
            total_sum_media_grid_thw += sum_media_grid_thw

        messages = ChatMessages(messages=data_item["messages"])
        replace_video_token(
            messages,
            self.chat_template,
            num_image_tokens_list,
            timestamps_list=timestamps_list,
            add_vision_id=self.add_vision_id,
        )
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

        position_ids = get_rope_index_3(
            torch.tensor(input_ids).unsqueeze(0),
            spatial_merge_size=self.image_processor.merge_size,
            video_grid_thw=torch.cat(grid_thw_list),
        )

        input_ids, labels, position_ids = self._truncated_data_item(input_ids, labels, position_ids)

        # 如果图片被截断，则该数据要丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.video_context_token_id).sum()
        num_image_tokens_2 = total_sum_media_grid_thw
        # assert 会被捕获，该数据会丢弃
        assert num_image_tokens_1 == num_image_tokens_2, (
            f"num_video_tokens_1 {num_image_tokens_1} != num_video_tokens_2 {num_image_tokens_2}, "
            f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
        )

        pixel_values = torch.cat(pixel_values_list, dim=0)

        ret = QwenVL3DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,  # (n, d)
            image_grid_thw=torch.cat(grid_thw_list),  # b, 3
            position_ids=position_ids,
            num_tokens=len(input_ids),
            num_img_tokens=[total_sum_media_grid_thw],
            num_imgs=num_imgs_list,
        )
        return ret


class Qwen3VLTokenizeFnConfig(BaseMLLMTokenizeFnConfig):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="forbid")
    processor_path: str
    min_pixels: int | None = None
    max_pixels: int | None = None
    oss_loader_cfg: OSSLoaderConfig | None = None

    video_min_total_pixels: int | None = None
    video_max_total_pixels: int | None = None
    video_min_frames: int | None = None
    video_max_frames: int | None = None
    fps: int | None = None
    rand_video_max_frames: int = 24

    # When handling multiple images or multiple videos,
    # it's helpful to add labels to the images and videos for better reference.
    # 注意这个逻辑和 hf 官方不是完全一致。 hf 官方只要开启这个 flag 就一定追加，不管是单个图片还是单个视频
    # xtuner 中做了优化，开启该 flag 且存在多图或者多视频才会追加
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
            oss_loader_cfg=self.oss_loader_cfg,
            video_min_total_pixels=self.video_min_total_pixels,
            video_max_total_pixels=self.video_max_total_pixels,
            video_min_frames=self.video_min_frames,
            video_max_frames=self.video_max_frames,
            rand_video_max_frames=self.rand_video_max_frames,
            fps=self.fps,
            add_vision_id=self.add_vision_id,
            max_length=self.max_length,
            system_message=self.system_message,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
            debug=self.debug,
            oss_time_log_thr=self.oss_time_log_thr,
            add_eos_token=self.add_eos_token,  # for mllm pretrain
            add_bos_token=self.add_bos_token,  # for mllm pretrain
        )
