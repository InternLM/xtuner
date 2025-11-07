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

    indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)

    return indices


def calculate_timestamps(indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
    if not isinstance(indices, list):
        indices = indices.tolist()
    if len(indices) % merge_size != 0:
        indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
    timestamps = [idx / video_fps for idx in indices]
    # @JJJYmmm frames are merged by self.merge_size, \
    # so we need to average the timestamps between the first/last frame within the temporal patch
    timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)]
    return timestamps


def replace_video_token(
    messages: ChatMessages,
    chat_template: HybridChatTemplate,
    num_image_token_list: list[int],
    timestamps: list[float] | None = None,
):
    current_image_idx = 0
    n_frames = len(num_image_token_list)
    if timestamps is not None:
        assert len(timestamps) == n_frames, "timestamps should have the same length as num_image_token_list"
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
                            video_placeholder = ""
                            for frame_idx in range(n_frames):
                                if timestamps is not None:
                                    curr_time = timestamps[frame_idx]
                                    video_placeholder += f"<{curr_time:.1f} seconds>"
                                video_placeholder += IMAGE_TOKEN_ALIAS
                            text = text.replace(IMAGE_TOKEN_ALIAS, video_placeholder)
                            image_tokens = f"{chat_template.image_start_token}{chat_template.video_context_token * num_image_token_list[current_image_idx]}{chat_template.image_end_token}"  # type: ignore
                            text = text.replace(IMAGE_TOKEN_ALIAS, image_tokens)
                            current_image_idx += n_frames
                        c.text = text
    # if current_image_idx < num_image, it means <image> placeholder is less than num_image
    assert current_image_idx == len(num_image_token_list), (
        f"ERROR: current_image_idx: {current_image_idx} != num_image: {len(num_image_token_list)}"
    )


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

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_context_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.video_context_token)

        # Note: 比较重要，防止改了参数但是没有重新 cache
        self._hash_str = (
            f"{self.image_processor.size['shortest_edge']}_{self.image_processor.size['longest_edge']}_"
            f"{self.video_processor.size['shortest_edge']}"
            f"_{self.video_processor.size['longest_edge']}_{self.video_processor.min_frames}_"
            f"{self.video_processor.max_frames}_{self.video_processor.fps}_"
            f"{self.add_vision_id}_{system_message}_{max_length}_{self.rand_video_max_frames}"
        )

        self.size = SimpleNamespace(**self.video_processor.size)
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

    def _calc_frame_info(self, data_item):
        # TODO: 目前只支持一个视频
        assert len(self._video_path) == 1, "Only one video is supported for now."
        assert len(self._video_wh_list) == 1, "Only one video is supported for now."
        num_frames = None
        origin_fps = None
        origin_video_length = None
        timestamps = None
        if len(self._video_extra_info_list) > 0:
            assert len(self._video_extra_info_list) == 1, "Only one video is supported for now."
            origin_fps = self._video_extra_info_list[0].get("origin_fps")
            origin_video_length = self._video_extra_info_list[0].get("origin_video_length")
            # 两个要不都存在，要么都不存在
            assert (origin_fps is None) == (origin_video_length is None), (
                f"origin_fps and origin_video_length must both exist or both not exist, "
                f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
            )
            if origin_fps is not None and origin_video_length is not None:
                indices = sample_frames(
                    origin_total_num_frames=origin_video_length,
                    origin_fps=origin_fps,
                    fps=self.video_processor.fps,
                    min_frames=self.video_processor.min_frames,
                    max_frames=self.video_processor.max_frames,
                )
                timestamps = calculate_timestamps(indices, origin_fps, merge_size=self.video_processor.merge_size)
                num_frames = len(indices)

        if num_frames is None:
            # 根据采样的帧数（min_num_frames, max_num_frames+1），计算token数量，实际可能采样不到这么多帧（比如视频一共只有10帧），算出来num_tokens可能会偏大
            num_frames = generate_random_int_from_dict(
                data_item, self.video_processor.min_frames, self.rand_video_max_frames
            )
        return num_frames, origin_fps, origin_video_length, timestamps

    def calc_num_tokens_video_get_item(self, data_item: dict) -> CacheItem:
        num_frames, _, _, timestamps = self._calc_frame_info(data_item)

        height, width = self._video_wh_list[0]
        resized_height, resized_width = video_smart_resize(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=self.video_processor.temporal_patch_size,
            factor=self.video_processor.patch_size * self.video_processor.merge_size,
            min_pixels=self.size.shortest_edge,
            max_pixels=self.size.longest_edge,
        )

        # 如果 num_frames 不是 temporal_patch_size 的整数倍，需要确保可以整除
        if num_frames % self.video_processor.temporal_patch_size != 0:
            # 这个代码实际上只有在 temporal_patch_size=2 才是对的，不修复的原因是： hf 里面是这么写的，没法随便改，否则会出现 cache 问题
            # 但是幸好，temporal_patch_size 就是等于 2
            num_frames += self.video_processor.temporal_patch_size - 1

        grid_t = num_frames // self.video_processor.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.video_processor.patch_size,
            resized_width // self.video_processor.patch_size,
        )
        sum_media_grid_thw = grid_t * grid_h * grid_w // self.merge_length
        frame_seqlen = grid_h * grid_w // self.merge_length

        messages = ChatMessages(messages=data_item["messages"])
        replace_video_token(messages, self.chat_template, [frame_seqlen] * grid_t, timestamps=timestamps)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]

        input_ids, _, _ = self._truncated_data_item(input_ids)

        # 如果图片被截断，则该数据丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.video_token_id).sum()
        num_image_tokens_2 = sum_media_grid_thw
        if num_image_tokens_1 != num_image_tokens_2:
            logger.warning(
                f"num_video_tokens_1 {num_image_tokens_1} != num_video_tokens_2 {num_image_tokens_2}, "
                f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
            )
            return {"num_tokens": 0}

        return {"num_tokens": len(input_ids)}

    def video_get_item(self, data_item: dict, media_root: str = "") -> QwenVL3DataItem:
        num_frames, origin_fps, origin_video_length, _ = self._calc_frame_info(data_item)
        video_path = os.path.join(media_root, self._video_path[0])

        # 上面计算的 num_frames 是理论值，需要根据实际读取的数进行更新
        if self.oss_loader is not None:
            image_list, frame_indices = self.oss_loader(video_path, image_type="video", num_frames=num_frames)
        else:
            image_list, frame_indices = read_qwen3_vl_video(video_path, num_frames=num_frames)

        video_data = torch.stack(image_list)  # num_patch,3,h,w
        num_frames = len(image_list)
        timestamps = None
        if origin_fps is not None and origin_video_length is not None:
            # 如果 timestamps 无法被整除，则会追加, 官方代码写的有不少问题
            timestamps = calculate_timestamps(frame_indices, origin_fps, merge_size=self.video_processor.merge_size)

        # 如果视频长度无法被整除，也会追加
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
        grid_thw = video_result["video_grid_thw"]
        sum_media_grid_thw = grid_thw.prod() // self.merge_length
        frame_seqlen = grid_thw[0][1:].prod() // self.merge_length

        # 作为验证
        height, width = self._video_wh_list[0]
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

        messages = ChatMessages(messages=data_item["messages"])
        replace_video_token(messages, self.chat_template, [frame_seqlen] * grid_thw[0][0], timestamps=timestamps)
        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]

        position_ids = get_rope_index_3(
            torch.tensor(input_ids).unsqueeze(0),
            spatial_merge_size=self.image_processor.merge_size,
            video_grid_thw=grid_thw,
        )

        input_ids, labels, position_ids = self._truncated_data_item(input_ids, labels, position_ids)

        # 如果图片被截断，则该数据要丢弃
        num_image_tokens_1 = (torch.tensor(input_ids) == self.video_token_id).sum()
        num_image_tokens_2 = sum_media_grid_thw
        # assert 会被捕获，该数据会丢弃
        assert num_image_tokens_1.shape == num_image_tokens_2.shape, (
            f"num_video_tokens_1 {num_image_tokens_1} != num_video_tokens_2 {num_image_tokens_2}, "
            f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
        )

        num_img_tokens = sum_media_grid_thw + num_frames * 2

        ret = QwenVL3DataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=image_tensor,  # (n,d)
            image_grid_thw=torch.cat([_thw.unsqueeze(0) for _thw in grid_thw], dim=0),  # b,3
            position_ids=position_ids,
            num_tokens=len(input_ids),
            num_img_tokens=[num_img_tokens],
            num_imgs=[num_frames],
            num_patches=[0],
        )
        return ret


class Qwen3VLTokenizeFnConfig(BaseMLLMTokenizeFnConfig):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="forbid")
    processor_path: str
    min_pixels: int | None = None
    max_pixels: int | None = None

    video_min_total_pixels: int | None = None
    video_max_total_pixels: int | None = None
    video_min_frames: int | None = None
    video_max_frames: int | None = None
    fps: int | None = None
    rand_video_max_frames: int = 24

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
            video_min_frames=self.video_min_frames,
            video_max_frames=self.video_max_frames,
            rand_video_max_frames=self.rand_video_max_frames,
            fps=self.fps,
            add_vision_id=self.add_vision_id,
            max_length=self.max_length,
            system_message=self.system_message,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
        )
