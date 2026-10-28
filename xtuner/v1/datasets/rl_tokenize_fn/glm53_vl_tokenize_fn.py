# Copyright (c) OpenMMLab. All rights reserved.
"""GLM53 RL prompts reuse the SFT processor's expanded training inputs."""

import copy
from typing import Any, cast

import torch

from transformers import PreTrainedTokenizerBase
from xtuner.v1.data_proto.rl_data import MultimodalInfo, RolloutState
from xtuner.v1.datasets.data_item import CacheItem, Glm53VLDataItem
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import get_image_path
from xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn import (
    Glm53VLTokenizeFnConfig,
    Glm53VLTokenizeFunction,
)
from xtuner.v1.datasets.mllm_tokenize_fn.image_data_uri import is_data_uri


class RLGlm53VLTokenizeFunction(Glm53VLTokenizeFunction):
    """Produce GLM53 image/text rollout state and multimodal training metadata.

    Video outputs are for training replay; built-in online rollout transports reject them explicitly.
    """

    def __init__(
        self, *args: Any, ignore_multimodal_info: bool = False, data_judger_mapping: dict | None = None, **kwargs: Any
    ) -> None:
        self.ignore_multimodal_info = ignore_multimodal_info
        self.data_judger_mapping = data_judger_mapping
        super().__init__(*args, **kwargs)

    def __call__(self, item: dict, media_root: str = "", **kwargs: Any) -> RolloutState | CacheItem:
        prompt = copy.deepcopy(item["prompt"])
        messages = prompt["messages"] if isinstance(prompt, dict) else prompt
        tools = prompt.get("tools") if isinstance(prompt, dict) else item.get("tools")
        if self.system_message and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": self.system_message})
        sft_item = {"messages": messages, "tools": tools}
        data = super().__call__(sft_item, media_root=media_root)
        num_tokens = data["num_tokens"]
        too_long = self.max_length is not None and num_tokens > self.max_length
        if self.state == "cache":
            return CacheItem(
                num_tokens=0 if too_long else num_tokens,
                proxy_attn_flops=data.get("proxy_attn_flops", float(num_tokens)),
            )
        if too_long:
            raise ValueError(f"GLM53 RL prompt length {num_tokens} exceeds max_length={self.max_length}")
        data = cast(Glm53VLDataItem, data)
        # Rollout receives one raw placeholder per media item; training receives HF-expanded slots.
        raw_prompt, _ = self._render_base(sft_item)
        prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        extra = copy.deepcopy(item.get("extra_info", {}))
        extra.update(raw_prompt=raw_prompt, train_prompt_ids=data["input_ids"])
        if self._image_path:
            extra["image_data"] = [
                path if is_data_uri(path) else get_image_path(path, media_root) for path in self._image_path
            ]
        if self._video_path:
            extra["video_data"] = [get_image_path(path, media_root) for path in self._video_path]
        data_source = item["data_source"]
        extra["origin_data_source"] = data_source
        mapped_source = (
            self.data_judger_mapping.get(data_source) if self.data_judger_mapping is not None else {data_source: 1.0}
        )
        return RolloutState(
            message=messages,
            tools=tools,
            prompt_ids=prompt_ids,
            num_tokens=num_tokens,
            proxy_attn_flops=data.get("proxy_attn_flops", float(num_tokens)),
            data_source=mapped_source,
            reward_model=copy.deepcopy(item.get("reward_model", {})),
            mm_info=None if self.ignore_multimodal_info else self._multimodal_info(data),
            extra_fields=extra,
        )

    def hash(self) -> str:
        """Return an RL-specific cache key covering the inherited processor
        settings.

        Returns:
            str: Cache key distinct from SFT tokenization.
        """
        # v1 is the RL cache namespace version, separate from the inherited SFT cache.
        return "rl_glm53_vl_v1_" + super().hash()

    def _truncated_input_and_labels(
        self, input_ids: list[int], labels: list[int] | torch.Tensor | None = None
    ) -> tuple[list[int], list[int] | torch.Tensor | None]:
        # RL filters whole prompts by length; truncating would detach media from rollout inputs.
        return input_ids, labels

    @staticmethod
    def _multimodal_info(data: Glm53VLDataItem) -> MultimodalInfo:
        info: dict[str, Any] = {"num_img_tokens": data["num_img_tokens"]}
        for name in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids"):
            value = cast(torch.Tensor | None, data.get(name))
            if value is not None:
                info[name] = value.detach().cpu().numpy()
        return cast(MultimodalInfo, info)


class RLGlm53VLTokenizeFnConfig(Glm53VLTokenizeFnConfig):
    add_generation_prompt: bool = True
    ignore_multimodal_info: bool = False
    data_judger_mapping: dict | None = None

    def build(
        self, tokenizer: PreTrainedTokenizerBase, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs: Any
    ) -> RLGlm53VLTokenizeFunction:
        # Only inherited pretraining-template flags are unused by the GLM renderer.
        options = {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name not in {"chat_template", "add_bos_token", "add_eos_token"}
        }
        return RLGlm53VLTokenizeFunction(tokenizer, anno_name=anno_name, tokenizer_hash=tokenizer_hash, **options)
