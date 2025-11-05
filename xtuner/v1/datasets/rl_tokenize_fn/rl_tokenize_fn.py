# Copyright (c) OpenMMLab. All rights reserved.
from typing import cast

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.rl_data import RLDatasetItem
from xtuner.v1.utils import get_logger

from ..data_item import OmniDataItem
from ..mllm_tokenize_fn.intern_s1_vl_tokenize_fn import InternS1VLTokenizeFunction
from ..mllm_tokenize_fn.qwen3_vl_tokenize_fn import Qwen3VLTokenizeFunction
from ..utils import CachableTokenizeFunction, replace_image_context_and_collect_media_data


logger = get_logger()


class RLTokenizeFn(CachableTokenizeFunction[RLDatasetItem]):
    def __init__(
        self,
        tokenizer_fn: CachableTokenizeFunction | None,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
        ignore_mm_process: bool = False,
    ):
        super().__init__(tokenizer)
        self.tokenizer_fn = tokenizer_fn
        self.max_length = max_length
        self.ignore_mm_process = ignore_mm_process

        self.model_name = "default"
        if self.tokenizer_fn:
            if isinstance(self.tokenizer_fn, Qwen3VLTokenizeFunction):
                self.model_name = "qwen3_vl"
            elif isinstance(self.tokenizer_fn, InternS1VLTokenizeFunction):
                self.model_name = "intern_s1_vl"
            else:
                raise ValueError(f"Unsupported tokenizer_fn type: {type(self.tokenizer_fn)}")

    def __call__(self, item: dict, **kwargs) -> RLDatasetItem:
        """example:
        item = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
        """

        extra_info = item.get("extra_info", {})
        messages = item["prompt"]
        if self.tokenizer_fn is None:
            # pure text
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            data = self.tokenizer(raw_prompt, add_special_tokens=False)
            prompt_token_ids = data["input_ids"]
            num_tokens = len(data["input_ids"])
        else:
            # mllm
            self.tokenizer_fn.state = self.state
            if self.ignore_mm_process:
                # 如果是评估模式下，不需要 pixel_values 等计算负担比较大的属性
                self.tokenizer_fn.state = "cache"

            data = self.tokenizer_fn({"messages": messages}, **kwargs)
            data = cast(OmniDataItem, data)
            num_tokens = data["num_tokens"]

            media_root = kwargs.get("media_root", "")
            if self.model_name == "qwen3_vl":
                image_data, _ = replace_image_context_and_collect_media_data(messages, media_root, True)
            elif self.model_name == "intern_s1_vl":
                image_data, _ = replace_image_context_and_collect_media_data(messages, media_root, False)
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")
            if image_data:
                extra_info["image_data"] = image_data

            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_token_ids = self.tokenizer(raw_prompt, add_special_tokens=False)["input_ids"]

        multimodal_train_info = {}
        extra_info["raw_prompt"] = raw_prompt

        if self.state == "cache":
            if self.max_length is not None and num_tokens > self.max_length:
                num_tokens = 0  # will be filtered out by the dataset filter
        else:
            if self.max_length is not None:
                assert num_tokens <= self.max_length, f"num_tokens {num_tokens} > max_length {self.max_length}"
            if not self.ignore_mm_process:
                if "pixel_values" in data:
                    multimodal_train_info["pixel_values"] = data["pixel_values"]
                if "image_flags" in data:
                    multimodal_train_info["image_flags"] = data["image_flags"]  # intern-s1 or intern-vl
                if "image_grid_thw" in data:
                    multimodal_train_info["image_grid_thw"] = data["image_grid_thw"]  # qwen3-vl
                if "position_ids" in data:
                    multimodal_train_info["position_ids"] = data["position_ids"]  # qwen3-vl

                # 在多模态场景下，训练和 rollout 的 prompt ids 是不一样的
                # 为了统一训练处理逻辑，额外保存 train_prompt_ids
                extra_info["train_prompt_ids"] = data["input_ids"]
            else:
                assert "pixel_values" not in data, "在评估模式下，不应该有 pixel_values"

        rl_out_data = {
            "messages": messages,
            "input_ids": prompt_token_ids,
            "num_tokens": num_tokens,
            "reward_model": item["reward_model"],
            "ability": item.get("ability", None),
            "data_source": {item.get("data_source"): 1.0},
            "extra_info": extra_info,
            "multimodal_train_info": multimodal_train_info,
        }
        return rl_out_data  # type: ignore

    def hash(self) -> str:
        raise ValueError("不应该触发这个方法, 因为 RLTokenizeFn 不需要缓存。")


class RLTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base RL dataset config for xtuner", extra="forbid")
    tokenize_fn_cfg: BaseModel | None = None
    max_length: int | None = None
    ignore_mm_process: bool = False

    def build(
        self, tokenizer: PreTrainedTokenizer, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs
    ) -> RLTokenizeFn:
        tokenizer_fn = None
        if self.tokenize_fn_cfg:
            tokenizer_fn = self.tokenize_fn_cfg.build(
                tokenizer=tokenizer,
                tokenizer_hash=tokenizer_hash,
                anno_name=anno_name,
                **kwargs,
            )
        return RLTokenizeFn(
            tokenizer_fn, tokenizer=tokenizer, max_length=self.max_length, ignore_mm_process=self.ignore_mm_process
        )
