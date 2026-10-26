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

import copy
import json
import os
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from pydantic import ConfigDict, Field

from transformers import AutoProcessor
from transformers.models.glm5_next.image_processing_glm5_next import smart_resize as image_smart_resize
from transformers.models.glm5_next.video_processing_glm5_next import smart_resize as video_smart_resize
from transformers.video_utils import VideoMetadata
from xtuner.v1.data_proto.messages.glm53_chat import (
    _tokenize_with_loss_mask,
    glm53_media_messages,
    render_glm53_chat,
)
from xtuner.v1.utils import get_logger

from ..data_item import CacheItem, Glm53VLDataItem
from ..utils import apply_exif_orientation
from .base_mllm_tokenize_fn import (
    BaseMLLMTokenizeFnConfig,
    BaseMLLMTokenizeFunction,
    OSSLoaderConfig,
    collect_image_video_paths_and_extra,
    get_image_path,
    load_image,
)
from .image_data_uri import is_data_uri, load_image_data_uri
from .qwen3_vl_utils import Qwen3VLOSSLoader, pil_loader, sort_frames


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


def _video_frame_source(video_file: str, extra: dict) -> tuple[int, float]:
    """(frame count, fps) of the frames this video is actually read from.

    A frame folder -- marked by a trailing ``/``, the VLM data convention qwen3_vl's reader also
    keys on -- holds the video *after* resampling: ``processed_video_length`` frames at
    ``processed_fps``. Sampling has to index into that sequence; indexing it with the original
    video's frame count runs off the end of the folder (qwen3_vl's ``calc_frame_info`` rule 3). A
    video file is decoded from the original stream, so it keeps ``origin_video_length`` /
    ``origin_fps``.

    Decided from the path string rather than the filesystem on purpose: the cache path never
    touches the media, and the two paths must reach the same answer. When they disagreed, the
    cache predicted the original-length video (truncated to ``max_length``) while the runtime read
    overflowed the folder, and the dataset silently substituted a fake sample -- so the packer
    budgeted ~4096 tokens for a sample that trained as 33.
    """
    if video_file.endswith("/") and "processed_video_length" in extra:
        return extra["processed_video_length"], extra["processed_fps"]
    return extra["origin_video_length"], extra["origin_fps"]


def _read_video_frames(video_path: str, frame_indices: list[int]) -> list[Image.Image]:
    """Read specific frames from a video directory (numbered image files) or a
    real video file.

    Real video decoding uses ``decord`` directly rather than the qwen3_vl helpers: those assume qwen's merge-paired
    timestamp convention (``len(frames_indices) == len(timestamps) * 2``), which doesn't match GLM-5.3-Flash's own
    timestamp derivation (``VideoMetadata.timestamps``, computed from ``fps`` + ``frames_indices`` alone).
    """
    if os.path.isdir(video_path):
        files = sort_frames(os.listdir(video_path))
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
    allow_all_roles = True

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
        clear_thinking: bool = False,
        tokenizer_hash: str | None = None,
        hash: str | None = None,
        debug: bool = False,
        oss_time_log_thr: int = 10,
        trim_memory_interval: int = 1,
        oss_loader_cfg: OSSLoaderConfig | None = None,
    ):
        if processor_path not in _PROCESSOR_CACHE:
            _PROCESSOR_CACHE[processor_path] = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        self.processor = copy.deepcopy(_PROCESSOR_CACHE[processor_path])
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
        self.add_generation_prompt = add_generation_prompt
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        # 训练入口（本类与 text 侧的 Glm53ChatMessages.tokenize）暂不开放 clear_thinking=True：
        # 需要该语义的数据应由用户侧拆成多条样本（每条只有结尾轮算 loss）。config → fn → render
        # 的接线和 hash 覆盖保留，开放时无需再动 hash，当前先以断言守住取值。
        assert clear_thinking is False, (
            "Glm53VLTokenizeFunction only supports clear_thinking=False; prepare clear-thinking "
            "data by splitting turns into separate samples instead."
        )
        self.clear_thinking = clear_thinking
        self.system_message = system_message
        self.data_name = os.path.basename(anno_name)

        # Every __init__ parameter that affects cache-path output (per-sample num_tokens) must be
        # encoded here -- hash() is the only invalidation guard of the on-disk length cache, and a
        # missing entry means stale lengths survive a config change (packer budgets by old lengths
        # while runtime truncates by new ones). max_length is read from the local variable because
        # super().__init__ (which stores it as self.max_length) runs after this string is built.
        # clear_thinking 必须在 hash 里：它把历史轮的 reasoning 文本从渲染中剥掉，num_tokens
        # 随参数取值变化（实测 153 → 136 字符的对话级差异），不只是改 loss。
        _hash_str = (
            f"{self.image_processor.min_image_tokens}_{self.image_processor.max_image_tokens}_"
            f"{self.video_processor.fps}_{self.video_processor.max_frames}_{self.merge_unit}_"
            f"processor_path:{processor_path}_llm_pack_weight:{llm_pack_weight}_"
            f"visual_pack_weight:{visual_pack_weight}_max_length:{max_length}_"
            f"system_message:{system_message}_add_generation_prompt:{add_generation_prompt}_"
            f"enable_thinking:{enable_thinking}_reasoning_effort:{reasoning_effort}_"
            f"clear_thinking:{clear_thinking}"
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

    # ---- shared render/expand/tokenize, called by both cache and runtime paths ----

    def _collect_media(self, item: dict):
        # 渲染器为结构化 tool 输出（{"id": ..., "output": [...]} 条目）里嵌套的媒体同样发射
        # placeholder marker，而基类收集器只看每条消息 content 的直接列表项，嵌套媒体会出现
        # "渲染出 marker、却没有对应 payload" 的错配。glm53_media_messages 按渲染 marker 顺序
        # 把所有媒体 part（含嵌套 output 内的）拉平成合成 user 消息，保证位置式配对不错位。
        return collect_image_video_paths_and_extra(
            glm53_media_messages(item["messages"]), allow_all_roles=self.allow_all_roles
        )

    def _render_base(self, data_item: dict) -> tuple[str, list[bool]]:
        self._check_no_literal_visual_marker(data_item)
        messages = list(data_item["messages"])
        if self.system_message is not None and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.system_message}, *messages]
        return render_glm53_chat(
            messages,
            tools=data_item.get("tools"),
            add_generation_prompt=self.add_generation_prompt,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
            clear_thinking=self.clear_thinking,
        )

    def _check_no_literal_visual_marker(self, data_item: dict) -> None:
        # 字面 <|image|>/<|video|> 不只坑纯文本：_expand_marker 按首次出现位置展开，若文本（如
        # tool 输出里的代码片段）中的字面标记位于真实媒体标记之前，patch 会展开到字面字符串的
        # 位置；真实标记再被 max_length 截掉时，占位符数量恰好等于 patch 数，cache/runtime/
        # collator 三层计数校验全部放行，而媒体绑定已经错位。所以在渲染/展开之前扫描所有会被
        # renderer 输出的文本来源，把这类样本按脏数据交给统一丢弃通道；媒体结构项
        # （{"type": "image"} 等）不参与扫描——它们展开后就是合法的真实标记。
        texts: list[str] = []
        for message in data_item["messages"]:
            content = message.get("content", "")
            parts = content if isinstance(content, list) else [content]
            texts.extend(part for part in parts if isinstance(part, str))
            texts.extend(
                part.get("text", "") for part in parts if isinstance(part, Mapping) and part.get("type") == "text"
            )
            if message.get("role") == "assistant":
                reasoning_content = message.get("reasoning_content")
                if isinstance(reasoning_content, str):
                    texts.append(reasoning_content)
                for tool_call in message.get("tool_calls") or []:
                    tool_call = tool_call.get("function", tool_call)
                    texts.append(str(tool_call.get("name", "")))
                    texts.extend(
                        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                        for value in (tool_call.get("arguments") or {}).values()
                    )
            elif message.get("role") == "tool":
                for part in parts:
                    if isinstance(part, Mapping) and "output" in part:
                        output = part["output"]
                        texts.append(output if isinstance(output, str) else str(output))
        for tool in data_item.get("tools") or []:
            texts.append(json.dumps(tool.get("function", tool), ensure_ascii=False))

        for text in texts:
            for marker in (_IMAGE_MARKER, _VIDEO_MARKER):
                if marker in text:
                    raise RuntimeError(
                        f"GLM-5.3-Flash sample contains literal visual placeholder marker {marker!r} in "
                        f"message text before media expansion, data_name: {self.data_name}, "
                        f"data_id: {data_item.get('id', '')}. Discard this data."
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
        return self._cache_item(input_ids, num_img_tokens, data_item)

    def multi_modal_get_item(self, data_item: dict, media_root: str = "") -> Glm53VLDataItem:
        # Process each image before loading the next one so full-resolution PIL images
        # do not accumulate alongside the processed tensors. Preserve the media order
        # when concatenating the per-image processor outputs.
        pixel_values_list = []
        image_grid_list = []
        for image_file in self._image_path:
            # 按 Data Format Types（V0.5 Agent）§7.1-7.2：嵌入图片使用
            # data:image/<subtype>;base64,<payload>（整体称为 data URI）。
            # base64 图片直接内存解码，绕过 media_root / OSS；裸 base64 仍走引用路径，
            # 仅严格以小写 data: 开头时进入此分支；该分支内格式错误不回退到引用路径。
            if is_data_uri(image_file):
                image = load_image_data_uri(image_file)
            else:
                image_path_ = get_image_path(image_file, media_root)
                if self.oss_loader is not None and "s3://" in image_path_:
                    img_str = self.oss_loader.client.get(image_path_)
                    image = pil_loader(img_str)
                else:
                    assert "s3://" not in image_path_, "Please use oss_loader_cfg to load image from s3."
                    image = load_image(image_path_)
            image = apply_exif_orientation(image)
            processed_i = self.image_processor(images=[image], return_tensors="pt")
            pixel_values_list.append(processed_i["pixel_values"])
            image_grid_list.append(processed_i["image_grid_thw"])
            del image, processed_i

        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_grid_thw = torch.cat(image_grid_list, dim=0)
        num_images = len(pixel_values_list)
        del pixel_values_list, image_grid_list

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
            num_imgs=[num_images],
            num_img_tokens=num_img_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=torch.tensor(mm_token_type_ids, dtype=torch.long).unsqueeze(0),
        )

    def _cache_item(self, input_ids: list[int], num_img_tokens: list[int], data_item: dict) -> CacheItem:
        # Truncating text is harmless, but truncating *through* a visual span leaves fewer
        # placeholders than patches, which the runtime path then rejects and the dataset silently
        # replaces with a fake sample. Report such a sample as damaged (`num_tokens=0`) here, so
        # it is filtered out before packing instead of being packed at its truncated length and
        # training as a 33-token fake -- qwen3_vl's cache path makes the same check.
        placeholder_count = input_ids.count(self.processor.image_token_id)
        if placeholder_count != sum(num_img_tokens) // self.merge_unit:
            logger.warning(
                f"GLM-5.3-Flash sample exceeds max_length={self.max_length} inside its visual span "
                f"({placeholder_count} of {sum(num_img_tokens) // self.merge_unit} placeholders kept), "
                f"data_name: {self.data_name}, data_id: {data_item.get('id', '')}. Discard this data."
            )
            return {"num_tokens": 0, "num_img_tokens": [0]}
        return {"num_tokens": len(input_ids), "num_img_tokens": num_img_tokens}

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
        if len(self._video_extra_info_list) != len(self._video_path):
            # 单条样本缺视频元数据属脏数据；RuntimeError 由 __call__ 转成警告+丢弃（cache），
            # 不能用 assert——cache worker 进程里 AssertionError 逃逸会杀死 worker 并挂住主进程。
            raise RuntimeError(
                "GLM-5.3-Flash video cache prediction requires origin_video_length/origin_fps "
                f"metadata for every video, data_name: {self.data_name}."
            )
        grids: list[list[int]] = []
        metadata_list = []
        for video_file, (width, height), extra in zip(
            self._video_path, self._video_wh_list, self._video_extra_info_list
        ):
            num_frames, fps = _video_frame_source(video_file, extra)
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
        return self._cache_item(input_ids, num_img_tokens, data_item)

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
            num_frames, fps = _video_frame_source(video_file, extra)
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

    def _check_no_literal_visual_tokens(self, input_ids: list[int]) -> None:
        # tokenize 层的最后兜底：渲染入口的 _check_no_literal_visual_marker 已在源头（含多模态
        # 样本）拦截字面 <|image|>/<|video|>，这里兜住其余路径漏进的字面占位 token——它会被
        # create_mm_token_type_ids 标成视觉类型，混进含真实媒体的 pack 后 splice 计数错位、
        # collator 断言崩溃（线上 256 卡 Step 8 事故的根因）。在 splice 掩码能区分字面/真实
        # 占位符的正确修复落地前，这类样本按脏数据丢弃：cache 由 __call__ 的脏数据捕获转成
        # 警告 + num_tokens=0，runtime 抛出后由上游数据集换无 loss 占位样本——两种状态都不中断训练。
        n_literal = input_ids.count(self.processor.image_token_id)
        if n_literal > 0:
            raise RuntimeError(
                f"GLM-5.3-Flash pure-text sample contains {n_literal} literal visual placeholder "
                f"token(s) (e.g. '<|image|>' in text), data_name: {self.data_name}."
            )

    def calc_num_tokens_pure_text_get_item(self, data_item) -> CacheItem:
        text, loss_mask = self._render_base(data_item)
        input_ids, labels = _tokenize_with_loss_mask(self.tokenizer, text, loss_mask)
        input_ids, _ = self._truncated_input_and_labels(input_ids, labels)
        self._check_no_literal_visual_tokens(input_ids)
        return {"num_tokens": len(input_ids), "num_img_tokens": [0]}

    def pure_text_get_item(self, data_item: dict) -> Glm53VLDataItem:
        text, loss_mask = self._render_base(data_item)
        input_ids, labels, mm_token_type_ids = self._finalize(text, loss_mask)
        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        mm_token_type_ids = mm_token_type_ids[: len(input_ids)]
        self._check_no_literal_visual_tokens(input_ids)
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
    # 非正的像素上限曾在构造期放行、到每张图的 HF smart_resize 才抛 ValueError——落在
    # _DIRTY_DATA_ERRORS 里被 cache 当成单条脏样本静默丢弃，所有图片样本无声消失。配置错误
    # 必须在构造期暴露，靠这里拦住，不靠每样本的丢弃通道。
    min_pixels: int | None = Field(default=None, gt=0)
    max_pixels: int | None = Field(default=None, gt=0)
    oss_loader_cfg: OSSLoaderConfig | None = None
    fps: float | None = None
    max_frames: int | None = None
    add_generation_prompt: bool = False
    enable_thinking: bool = True
    reasoning_effort: Literal["low", "high", "max"] = "max"
    clear_thinking: bool = False
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
            oss_loader_cfg=self.oss_loader_cfg,
            fps=self.fps,
            max_frames=self.max_frames,
            system_message=self.system_message,
            max_length=self.max_length,
            llm_pack_weight=self.llm_pack_weight,
            visual_pack_weight=self.visual_pack_weight,
            add_generation_prompt=self.add_generation_prompt,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
            clear_thinking=self.clear_thinking,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
            debug=self.debug,
            oss_time_log_thr=self.oss_time_log_thr,
            trim_memory_interval=self.trim_memory_interval,
        )
