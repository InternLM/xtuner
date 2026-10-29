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

    # 复用 SFT 的渲染和多模态展开，只在外层适配 RL 的输入、缓存和 RolloutState。
    def __init__(
        self, *args: Any, ignore_multimodal_info: bool = False, data_judger_mapping: dict | None = None, **kwargs: Any
    ) -> None:
        # 仅保存 RL 专属选项：是否输出训练媒体元信息，以及数据源到评分器权重的映射。
        self.ignore_multimodal_info = ignore_multimodal_info
        self.data_judger_mapping = data_judger_mapping
        super().__init__(*args, **kwargs)

    def __call__(self, item: dict, media_root: str = "", **kwargs: Any) -> RolloutState | CacheItem:
        # 1. 兼容 messages 列表和带 tools 的 prompt 字典；深拷贝后补 system，避免改动原始样本。
        prompt = copy.deepcopy(item["prompt"])
        messages = prompt["messages"] if isinstance(prompt, dict) else prompt
        tools = prompt.get("tools") if isinstance(prompt, dict) else item.get("tools")
        if self.system_message and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": self.system_message})
        sft_item = {"messages": messages, "tools": tools}
        # 2. 由 SFT 父类统一生成展开后的训练 token、像素、grid 和逐 token 模态标记。
        data = super().__call__(sft_item, media_root=media_root)
        num_tokens = data["num_tokens"]
        too_long = self.max_length is not None and num_tokens > self.max_length
        # 3. 缓存预处理以 num_tokens=0 标记超长样本，供上层过滤；正式读取遇到超长则直接报错。
        if self.state == "cache":
            return CacheItem(
                num_tokens=0 if too_long else num_tokens,
                proxy_attn_flops=data.get("proxy_attn_flops", float(num_tokens)),
            )
        if too_long:
            raise ValueError(f"GLM53 RL prompt length {num_tokens} exceeds max_length={self.max_length}")
        data = cast(Glm53VLDataItem, data)
        # 4. rollout 使用未展开的媒体占位符，训练使用 HF 展开后的 token，两套 prompt 不能互换。
        raw_prompt, _ = self._render_base(sft_item)
        prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        extra = copy.deepcopy(item.get("extra_info", {}))
        extra.update(raw_prompt=raw_prompt, train_prompt_ids=data["input_ids"])
        # 5. 父类 __call__ 的媒体收集阶段已记录路径；补齐相对路径供 rollout 加载，图片 data URI 保持原样。
        if self._image_path:
            extra["image_data"] = [
                path if is_data_uri(path) else get_image_path(path, media_root) for path in self._image_path
            ]
        if self._video_path:
            # 视频字段供训练回放使用；误配到内置在线生成或评分入口时会被显式拒绝。
            extra["video_data"] = [get_image_path(path, media_root) for path in self._video_path]
        # 6. 保留原数据源供追踪；未配置评分器映射时，使用原数据源且权重为 1。
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
        # RL 与 SFT 的缓存产物不同，用独立命名空间隔离；v1 表示此 RL 缓存格式的版本。
        return "rl_glm53_vl_v1_" + super().hash()

    def _truncated_input_and_labels(
        self, input_ids: list[int], labels: list[int] | torch.Tensor | None = None
    ) -> tuple[list[int], list[int] | torch.Tensor | None]:
        # 覆盖 SFT 的截断行为：RL 按整条 prompt 过滤，避免训练 token 被截断而 rollout 条件仍完整。
        # 此处保留原 token 和 labels，长度检查统一由 __call__ 处理。
        return input_ids, labels

    @staticmethod
    def _multimodal_info(data: Glm53VLDataItem) -> MultimodalInfo:
        # 将训练所需媒体字段打包为 CPU numpy 数据，供后续 sampler 放入 Ray 对象存储。
        # 纯文本或缺失模态对应的可选字段为 None 时不写入，像素、grid 和模态标记保持父类顺序。
        info: dict[str, Any] = {"num_img_tokens": data["num_img_tokens"]}
        for name in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids"):
            value = cast(torch.Tensor | None, data.get(name))
            if value is not None:
                info[name] = value.detach().cpu().numpy()
        return cast(MultimodalInfo, info)


class RLGlm53VLTokenizeFnConfig(Glm53VLTokenizeFnConfig):
    # 继承 SFT 的 processor 配置；RL 默认追加生成前缀，并增加媒体输出和评分器映射选项。
    add_generation_prompt: bool = True
    ignore_multimodal_info: bool = False
    data_judger_mapping: dict | None = None

    def build(
        self, tokenizer: PreTrainedTokenizerBase, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs: Any
    ) -> RLGlm53VLTokenizeFunction:
        # 透传继承和新增的配置字段，只排除 GLM renderer 不使用的三个继承选项。
        options = {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name not in {"chat_template", "add_bos_token", "add_eos_token"}
        }
        return RLGlm53VLTokenizeFunction(tokenizer, anno_name=anno_name, tokenizer_hash=tokenizer_hash, **options)
