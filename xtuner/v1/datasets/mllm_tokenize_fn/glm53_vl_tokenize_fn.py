# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash VL TokenizeFn: cache/runtime dual path, see
doc/xtuner_glm5p3flash_design.md F1.

Placeholder expansion and ``mm_token_type_ids`` are produced by calling the real HF
``Glm5NextProcessor``'s own public ``replace_image_token``/``replace_video_token``/
``create_mm_token_type_ids`` methods in BOTH the cache and runtime paths (design doc §8.2:
"cache 与 runtime 共用同一份构造代码"). This is the strongest form of that guarantee: cache and
runtime literally call the same HF code, differing only in whether ``image_grid_thw`` /
``video_metadata`` come from geometry-only prediction (cache, no media decode) or the real
processor output (runtime).

``transformers`` 5.17.0 does not expose ``Glm5NextVideoProcessor.get_number_of_video_patches``
(only the image side has a convenience wrapper; verified against the installed package -- the
design doc's claim that both exist does not hold for video). Cache-path video grid prediction
therefore composes the two public building blocks that ARE exposed --
``Glm5NextVideoProcessor.sample_frames`` (frame selection from metadata only) and the
module-level ``smart_resize`` function (spatial resize target) -- the same two calls the real
preprocessing path makes internally. Verified bitwise-equal to the real processor's
``video_grid_thw`` output for a synthetic 4-frame video in this repo's development scratch.
"""

import os
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from pydantic import ConfigDict

from transformers import AutoProcessor
from transformers.models.glm5_next.image_processing_glm5_next import smart_resize as image_smart_resize
from transformers.models.glm5_next.video_processing_glm5_next import smart_resize as video_smart_resize
from transformers.video_utils import VideoMetadata
from xtuner.v1.data_proto.messages.glm53_chat import _tokenize_with_loss_mask, render_glm53_chat
from xtuner.v1.utils import get_logger

from ..data_item import CacheItem, Glm53VLDataItem
from ..utils import apply_exif_orientation
from .base_mllm_tokenize_fn import (
    BaseMLLMTokenizeFnConfig,
    BaseMLLMTokenizeFunction,
    collect_image_video_paths_and_extra,
    get_image_path,
    load_image,
)


logger = get_logger()

# Shared across datasets that use the same processor_path, mirroring qwen3_vl_tokenize_fn.py.
_PROCESSOR_CACHE: dict[str, Any] = {}

# `replace_image_token`/`replace_video_token` (HF) only replace the single INNER placeholder
# token; the `<|begin_of_image|>`/`<|end_of_image|>` (or `..._video...`) wrapper that
# render_glm53_chat/the jinja template emits around it stays untouched in the final sequence.
# Verified against a real processor(text=..., images=...) call: expanding the whole 3-token
# wrapper (as an earlier version of this function did) drops the begin/end tokens and undercounts
# num_tokens by 2 per media item.
_IMAGE_MARKER = "<|image|>"
_VIDEO_MARKER = "<|video|>"


def _read_video_frames(video_path: str, frame_indices: list[int]) -> list[Image.Image]:
    """Read specific frames from a video directory (numbered image files) or a
    real video file.

    Real video decoding uses ``decord`` directly rather than the qwen3_vl helpers: those assume
    qwen's merge-paired timestamp convention (``len(frames_indices) == len(timestamps) * 2``),
    which doesn't match GLM-5.3-Flash's own timestamp derivation (``VideoMetadata.timestamps``,
    computed from ``fps`` + ``frames_indices`` alone).
    """
    if os.path.isdir(video_path):
        files = sorted(os.listdir(video_path))
        return [Image.open(os.path.join(video_path, files[i])).convert("RGB") for i in frame_indices]

    from decord import VideoReader

    reader = VideoReader(video_path)
    frames = reader.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


def _expand_marker(text: str, loss_mask: list[bool], marker: str, expansions: list[str]) -> tuple[str, list[bool]]:
    """Replace each (in-order) occurrence of ``marker`` with its expansion;
    expansions carry no loss."""
    text_parts: list[str] = []
    mask_parts: list[list[bool]] = []
    for expansion in expansions:
        idx = text.index(marker)
        text_parts.append(text[:idx])
        mask_parts.append(loss_mask[:idx])
        text_parts.append(expansion)
        mask_parts.append([False] * len(expansion))
        text = text[idx + len(marker) :]
        loss_mask = loss_mask[idx + len(marker) :]
    text_parts.append(text)
    mask_parts.append(loss_mask)
    return "".join(text_parts), [flag for chunk in mask_parts for flag in chunk]


class Glm53VLTokenizeFunction(BaseMLLMTokenizeFunction):
    def __init__(
        self,
        tokenizer,
        processor_path: str,
        anno_name: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        system_message: str | None = None,
        max_length: int | None = None,
        llm_pack_weight: float = 1.0,
        visual_pack_weight: float = 1.0,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        reasoning_effort: Literal["low", "high", "max"] = "max",
        tokenizer_hash: str | None = None,
        hash: str | None = None,
        debug: bool = False,
        oss_time_log_thr: int = 10,
        trim_memory_interval: int = 1,
    ):
        if processor_path not in _PROCESSOR_CACHE:
            _PROCESSOR_CACHE[processor_path] = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        self.processor = _PROCESSOR_CACHE[processor_path]
        self.image_processor = self.processor.image_processor
        self.video_processor = self.processor.video_processor

        if min_pixels is not None:
            self.image_processor.min_image_tokens = min_pixels
        if max_pixels is not None:
            self.image_processor.max_image_tokens = max_pixels
        if fps is not None:
            self.video_processor.fps = fps
        if max_frames is not None:
            self.video_processor.max_frames = max_frames

        assert self.image_processor.merge_size == self.video_processor.merge_size, (
            "image/video merge_size must match: GLM-5.3-Flash placeholders share image_token_id "
            "and num_img_tokens semantics assume one merge_unit for both."
        )
        self.merge_unit = self.image_processor.merge_size**2

        # Consumed by BaseMLLMTokenizeFunction's OSS media loader, same as qwen3_vl's.
        self.debug = debug
        self.oss_time_log_thr = oss_time_log_thr
        self.add_generation_prompt = add_generation_prompt
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.system_message = system_message
        self.data_name = os.path.basename(anno_name)

        _hash_str = (
            f"{self.image_processor.min_image_tokens}_{self.image_processor.max_image_tokens}_"
            f"{self.video_processor.fps}_{self.video_processor.max_frames}_{self.merge_unit}_"
            f"processor_path:{processor_path}_llm_pack_weight:{llm_pack_weight}_"
            f"visual_pack_weight:{visual_pack_weight}"
        )

        super().__init__(
            tokenizer,
            None,  # type: ignore[arg-type]  # GLM-5.3-Flash doesn't use CHAT_TEMPLATE_MAP; see module docstring.
            max_length,
            tokenizer_hash,
            hash,
            hash_str=_hash_str,
            data_name=self.data_name,
            llm_pack_weight=llm_pack_weight,
            visual_pack_weight=visual_pack_weight,
            trim_memory_interval=trim_memory_interval,
        )

    def __call__(self, item: dict, media_root: str = "", **kwargs) -> Any:
        image_paths, video_paths, _ = collect_image_video_paths_and_extra(item["messages"])
        if image_paths and video_paths:
            raise NotImplementedError(
                "GLM-5.3-Flash TokenizeFn only supports image-only or video-only samples in this "
                "version; mixed image+video in a single sample needs grid/placeholder/"
                "mm_token_type_ids/num_img_tokens global-order support that isn't implemented yet "
                "(design doc §8.2)."
            )
        return super().__call__(item, media_root=media_root, **kwargs)

    # ---- shared render/expand/tokenize, called by both cache and runtime paths ----

    def _render_base(self, data_item: dict) -> tuple[str, list[bool]]:
        messages = list(data_item["messages"])
        if self.system_message is not None and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.system_message}, *messages]
        return render_glm53_chat(
            messages,
            tools=data_item.get("tools"),
            add_generation_prompt=self.add_generation_prompt,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        )

    def _finalize(self, text: str, loss_mask: list[bool]) -> tuple[list[int], list[int], list[int]]:
        input_ids, labels = _tokenize_with_loss_mask(self.tokenizer, text, loss_mask)
        mm_token_type_ids = self.processor.create_mm_token_type_ids([input_ids])[0]
        return input_ids, labels, mm_token_type_ids

    # ---- image: geometry prediction (cache) ----

    def _predict_image_grid(self, width: int, height: int) -> tuple[int, int, int]:
        resized_h, resized_w = image_smart_resize(
            num_frames=self.image_processor.temporal_patch_size,
            height=height,
            width=width,
            factor=self.image_processor.patch_size * self.image_processor.merge_size,
            temporal_factor=self.image_processor.temporal_patch_size,
            min_pixels=self.image_processor.min_image_tokens,
            max_pixels=self.image_processor.max_image_tokens,
        )
        grid_h = resized_h // self.image_processor.patch_size
        grid_w = resized_w // self.image_processor.patch_size
        return 1, grid_h, grid_w

    def calc_num_tokens_multi_modal_get_item(self, data_item: dict) -> CacheItem:
        grids = [self._predict_image_grid(w, h) for w, h in self._image_wh_list]
        text, loss_mask = self._render_base(data_item)
        image_inputs = {"image_grid_thw": torch.tensor(grids)}
        expansions = [self.processor.replace_image_token(image_inputs, i) for i in range(len(grids))]
        text, loss_mask = _expand_marker(text, loss_mask, _IMAGE_MARKER, expansions)
        input_ids, labels = _tokenize_with_loss_mask(self.tokenizer, text, loss_mask)
        input_ids, _ = self._truncated_input_and_labels(input_ids, labels)
        num_img_tokens = [grid_h * grid_w for _, grid_h, grid_w in grids]
        return {"num_tokens": len(input_ids), "num_img_tokens": num_img_tokens}

    def multi_modal_get_item(self, data_item: dict, media_root: str = "") -> Glm53VLDataItem:
        images = []
        for image_file in self._image_path:
            image = load_image(get_image_path(image_file, media_root))
            images.append(apply_exif_orientation(image))

        processed = self.image_processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        image_grid_thw = processed["image_grid_thw"]

        text, loss_mask = self._render_base(data_item)
        image_inputs = {"image_grid_thw": image_grid_thw}
        expansions = [self.processor.replace_image_token(image_inputs, i) for i in range(image_grid_thw.shape[0])]
        text, loss_mask = _expand_marker(text, loss_mask, _IMAGE_MARKER, expansions)
        input_ids, labels, mm_token_type_ids = self._finalize(text, loss_mask)

        num_img_tokens = (image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        # Truncation must never cut a visual span in half; assert-and-drop mirrors the existing
        # qwen3_vl_tokenize_fn.py convention (an AssertionError here is caught upstream and the
        # sample is discarded, not silently trained on a corrupted span).
        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        mm_token_type_ids = mm_token_type_ids[: len(input_ids)]
        placeholder_count = input_ids.count(self.processor.image_token_id)
        assert placeholder_count == sum(num_img_tokens) // self.merge_unit, (
            f"GLM-5.3-Flash image placeholder count {placeholder_count} != expected "
            f"{sum(num_img_tokens) // self.merge_unit}, data_name: {self.data_name}, "
            f"data_id: {data_item.get('id', '')}. Discard this data."
        )

        return Glm53VLDataItem(
            input_ids=input_ids,
            labels=labels,
            num_tokens=len(input_ids),
            num_imgs=[len(images)],
            num_img_tokens=num_img_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=torch.tensor(mm_token_type_ids, dtype=torch.long).unsqueeze(0),
        )

    # ---- video: geometry prediction (cache) ----

    def _predict_video_grid(
        self, num_frames: int, fps: float, height: int, width: int
    ) -> tuple[list[int], tuple[int, int, int]]:
        metadata = VideoMetadata(total_num_frames=num_frames, fps=fps, duration=num_frames / fps)
        sampled = self.video_processor.sample_frames(metadata, fps=self.video_processor.fps)
        resized_h, resized_w = video_smart_resize(
            num_frames=len(sampled),
            height=height,
            width=width,
            factor=self.video_processor.patch_size * self.video_processor.merge_size,
            temporal_factor=self.video_processor.temporal_patch_size,
            min_pixels=self.video_processor.min_image_tokens,
            max_pixels=self.video_processor.max_image_tokens,
        )
        grid_h = resized_h // self.video_processor.patch_size
        grid_w = resized_w // self.video_processor.patch_size
        padded_n = len(sampled) + (-len(sampled) % self.video_processor.temporal_patch_size)
        grid_t = padded_n // self.video_processor.temporal_patch_size
        return sampled.tolist(), (grid_t, grid_h, grid_w)

    def calc_num_tokens_video_get_item(self, data_item: dict) -> CacheItem:
        assert len(self._video_extra_info_list) == len(self._video_path), (
            "GLM-5.3-Flash video cache prediction requires origin_video_length/origin_fps "
            f"metadata for every video, data_name: {self.data_name}."
        )
        grids: list[list[int]] = []
        metadata_list = []
        for (width, height), extra in zip(self._video_wh_list, self._video_extra_info_list):
            num_frames, fps = extra["origin_video_length"], extra["origin_fps"]
            sampled, grid = self._predict_video_grid(num_frames, fps, height, width)
            metadata_list.append(VideoMetadata(total_num_frames=num_frames, fps=fps, frames_indices=sampled))
            grids.append(list(grid))

        text, loss_mask = self._render_base(data_item)
        video_inputs = {"video_grid_thw": torch.tensor(grids), "video_metadata": metadata_list}
        expansions = [self.processor.replace_video_token(video_inputs, i) for i in range(len(grids))]
        text, loss_mask = _expand_marker(text, loss_mask, _VIDEO_MARKER, expansions)
        input_ids, labels = _tokenize_with_loss_mask(self.tokenizer, text, loss_mask)
        input_ids, _ = self._truncated_input_and_labels(input_ids, labels)

        num_img_tokens: list[int] = []
        for grid_t, grid_h, grid_w in grids:
            num_img_tokens.extend([grid_h * grid_w] * grid_t)
        return {"num_tokens": len(input_ids), "num_img_tokens": num_img_tokens}

    def video_get_item(self, data_item: dict, media_root: str = "") -> Glm53VLDataItem:
        assert len(self._video_extra_info_list) == len(self._video_path), (
            "GLM-5.3-Flash video runtime requires origin_video_length/origin_fps metadata for "
            f"every video, data_name: {self.data_name}."
        )
        pixel_values_list = []
        grid_list = []
        metadata_list = []
        for video_file, (width, height), extra in zip(
            self._video_path, self._video_wh_list, self._video_extra_info_list
        ):
            video_path = os.path.join(media_root, video_file)
            num_frames, fps = extra["origin_video_length"], extra["origin_fps"]
            sampled, _ = self._predict_video_grid(num_frames, fps, height, width)

            frames = _read_video_frames(video_path, sampled)
            frame_tensor = torch.stack(
                [torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in frames]
            )  # [T, C, H, W]
            metadata = VideoMetadata(total_num_frames=num_frames, fps=fps, frames_indices=sampled)
            out = self.video_processor(
                videos=[frame_tensor], video_metadata=[metadata], do_sample_frames=False, return_tensors="pt"
            )
            pixel_values_list.append(out["pixel_values_videos"])
            grid_list.append(out["video_grid_thw"][0])
            metadata_list.append(metadata)

        pixel_values_videos = torch.cat(pixel_values_list, dim=0)
        video_grid_thw = torch.stack(grid_list)

        text, loss_mask = self._render_base(data_item)
        video_inputs = {"video_grid_thw": video_grid_thw, "video_metadata": metadata_list}
        expansions = [self.processor.replace_video_token(video_inputs, i) for i in range(video_grid_thw.shape[0])]
        text, loss_mask = _expand_marker(text, loss_mask, _VIDEO_MARKER, expansions)
        input_ids, labels, mm_token_type_ids = self._finalize(text, loss_mask)

        num_img_tokens: list[int] = []
        for grid_t, grid_h, grid_w in video_grid_thw.tolist():
            num_img_tokens.extend([grid_h * grid_w] * grid_t)

        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        mm_token_type_ids = mm_token_type_ids[: len(input_ids)]
        placeholder_count = input_ids.count(self.processor.image_token_id)
        assert placeholder_count == sum(num_img_tokens) // self.merge_unit, (
            f"GLM-5.3-Flash video placeholder count {placeholder_count} != expected "
            f"{sum(num_img_tokens) // self.merge_unit}, data_name: {self.data_name}, "
            f"data_id: {data_item.get('id', '')}. Discard this data."
        )

        return Glm53VLDataItem(
            input_ids=input_ids,
            labels=labels,
            num_tokens=len(input_ids),
            num_imgs=[0],
            num_img_tokens=num_img_tokens,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=torch.tensor(mm_token_type_ids, dtype=torch.long).unsqueeze(0),
        )

    # ---- pure text ----

    def calc_num_tokens_pure_text_get_item(self, data_item) -> CacheItem:
        text, loss_mask = self._render_base(data_item)
        input_ids, labels = _tokenize_with_loss_mask(self.tokenizer, text, loss_mask)
        input_ids, _ = self._truncated_input_and_labels(input_ids, labels)
        return {"num_tokens": len(input_ids), "num_img_tokens": [0]}

    def pure_text_get_item(self, data_item: dict) -> Glm53VLDataItem:
        text, loss_mask = self._render_base(data_item)
        input_ids, labels, mm_token_type_ids = self._finalize(text, loss_mask)
        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        mm_token_type_ids = mm_token_type_ids[: len(input_ids)]
        return Glm53VLDataItem(
            input_ids=input_ids,
            labels=labels,
            num_tokens=len(input_ids),
            num_imgs=[0],
            num_img_tokens=[0],
            mm_token_type_ids=torch.tensor(mm_token_type_ids, dtype=torch.long).unsqueeze(0),
        )

    def _truncated_input_and_labels(self, input_ids, labels=None):
        if self.max_length is not None and len(input_ids) > self.max_length:
            logger.info(
                f"WARNING: input_ids length {len(input_ids)} exceeds model_max_length {self.max_length}. truncated!"
            )
            input_ids = input_ids[: self.max_length]
            if labels is not None:
                labels = labels[: self.max_length]
        return input_ids, labels


class Glm53VLTokenizeFnConfig(BaseMLLMTokenizeFnConfig):
    model_config = ConfigDict(title="GLM-5.3-Flash VL dataset config for xtuner", extra="forbid")
    # GLM-5.3-Flash renders through `render_glm53_chat`, not a `CHAT_TEMPLATE_MAP` entry, so
    # `build()` never reads this field and `BaseMLLMTokenizeFunction` is constructed with None.
    # The base class declares it required, which made the documented "no chat_template needed"
    # construction raise at validation time; defaulted here to the name the renderer implements.
    chat_template: str = "glm5.3"
    processor_path: str
    min_pixels: int | None = None
    max_pixels: int | None = None
    fps: float | None = None
    max_frames: int | None = None
    add_generation_prompt: bool = False
    enable_thinking: bool = True
    reasoning_effort: Literal["low", "high", "max"] = "max"
    visual_pack_weight: float = 1.0

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs
    ) -> Glm53VLTokenizeFunction:
        return Glm53VLTokenizeFunction(
            tokenizer,
            self.processor_path,
            anno_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            fps=self.fps,
            max_frames=self.max_frames,
            system_message=self.system_message,
            max_length=self.max_length,
            llm_pack_weight=self.llm_pack_weight,
            visual_pack_weight=self.visual_pack_weight,
            add_generation_prompt=self.add_generation_prompt,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
            debug=self.debug,
            oss_time_log_thr=self.oss_time_log_thr,
            trim_memory_interval=self.trim_memory_interval,
        )
