# Copyright (c) OpenMMLab. All rights reserved.
from pydantic import BaseModel, ConfigDict
from xtuner.v1.utils import get_logger
from xtuner.v1.data_proto.rl_data import RLDatasetItem

from ..utils import CachableTokenizeFunction

logger = get_logger()


class RLTokenizeFn(CachableTokenizeFunction[RLDatasetItem]):
    def __init__(self, tokenizer_fn: CachableTokenizeFunction, max_length: int | None = None):
        super().__init__()
        self.tokenizer_fn = tokenizer_fn
        self.max_length = max_length

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
        messages = item["prompt"]
        num_tokens = self.tokenizer_fn(messages, **kwargs)["num_tokens"]
        if self.max_length is not None and num_tokens > self.max_length:
            num_tokens = 0  # will be filtered out by the dataset filter

        extra_info = item.get("extra_info", {})
        if 'media_root' in kwargs:
            media_root = kwargs['media_root']
            extra_info['media_root'] = media_root

        rl_out_data = {
            "messages": messages,
            "num_tokens": num_tokens,
            "reward_model": item["reward_model"],
            "ability": item.get("ability", None),
            "data_source": {item.get("data_source"): 1.0},
            "extra_info": extra_info,
        }
        return rl_out_data  # type: ignore

    def hash(self) -> str:
        raise ValueError("不应该触发这个方法, 因为 RLTokenizeFn 不需要缓存。")


class RLTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base RL dataset config for xtuner", extra="allow")
    sft_tokenize_fn_cfg: BaseModel
    max_length: int | None = None

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs
    ) -> RLTokenizeFn:
        sft_tokenizer_fn = self.sft_tokenize_fn_cfg.build(
            tokenizer=tokenizer,
            tokenizer_hash=tokenizer_hash,
            anno_name=anno_name,
            **kwargs,
        )
        return RLTokenizeFn(sft_tokenizer_fn, max_length=self.max_length)
