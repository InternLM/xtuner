# Copyright (c) OpenMMLab. All rights reserved.
from pydantic import BaseModel, ConfigDict
from xtuner.v1.utils import get_logger
from xtuner.v1.data_proto.rl_data import RLDatasetItem

from ..utils import CachableTokenizeFunction

logger = get_logger()


class RLTokenizeFn(CachableTokenizeFunction[RLDatasetItem]):
    def __init__(self, tokenizer_fn: CachableTokenizeFunction, max_length: int | None = None, is_training: bool = True):
        super().__init__()
        self.tokenizer_fn = tokenizer_fn
        self.max_length = max_length
        self.is_training = is_training

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
        if 'media_root' in kwargs:
            media_root = kwargs['media_root']
            extra_info['media_root'] = media_root

        messages = item["prompt"]

        self.tokenizer_fn.state = self.state
        if not self.is_training:
            # 如果是评估模式下，不需要 pixel_values 等计算负担比较大的属性
            self.tokenizer_fn.state = 'cache'

        data = self.tokenizer_fn({'messages': messages}, **kwargs)
        num_tokens = data["num_tokens"]
        multimodal_train_info = {}

        if self.state == 'cache':
            if self.max_length is not None and num_tokens > self.max_length:
                num_tokens = 0  # will be filtered out by the dataset filter
        else:
            assert num_tokens <= self.max_length, f"num_tokens {num_tokens} > max_length {self.max_length}"
            if self.is_training:
                if 'pixel_values' in data:
                    multimodal_train_info['pixel_values'] = data['pixel_values']
                if 'image_flags' in data:
                    multimodal_train_info['image_flags'] = data['image_flags']  # intern-s1 or intern-vl
                if 'image_grid_thw' in data:
                    multimodal_train_info['image_grid_thw'] = data['image_grid_thw']  # qwen3-vl
                if 'position_ids' in data:
                    multimodal_train_info['position_ids'] = data['position_ids']  # qwen3-vl

                # 在多模态场景下，训练和 rollout 的 prompt ids 是不一样的
                # 为了统一训练处理逻辑，额外保存 train_prompt_ids
                extra_info['train_prompt_ids'] = data['input_ids']
            else:
                assert 'pixel_values' not in data, "在评估模式下，不应该有 pixel_values"

        rl_out_data = {
            "messages": messages,
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
    model_config = ConfigDict(title="Base RL dataset config for xtuner", extra="allow")
    sft_tokenize_fn_cfg: BaseModel
    max_length: int | None = None
    is_training: bool = True

    def build(
            self, tokenizer, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs
    ) -> RLTokenizeFn:
        sft_tokenizer_fn = self.sft_tokenize_fn_cfg.build(
            tokenizer=tokenizer,
            tokenizer_hash=tokenizer_hash,
            anno_name=anno_name,
            **kwargs,
        )
        return RLTokenizeFn(sft_tokenizer_fn, max_length=self.max_length, is_training=self.is_training)
