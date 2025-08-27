# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from xtuner.v1.datasets.data_item import RLTextDataItem
from xtuner.v1.utils import get_logger

from ..utils import CachableTokenizeFunction


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_logger()


# refer from verl
class RLTextTokenizeFn(CachableTokenizeFunction[RLTextDataItem]):
    def __init__(self, tokenizer: "PreTrainedTokenizer", max_length: int | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, item: dict, **kwargs) -> RLTextDataItem:
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
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")[0]
        num_tokens = len(input_ids)
        if self.max_length is not None and num_tokens > self.max_length:
            num_tokens = 0  # will be filtered out by the dataset filter

        extra_info = item.get("extra_info", {})
        # extra_info["raw_prompt"] = raw_prompt

        rl_out_data = {
            "input_ids": input_ids,
            "prompt_str": raw_prompt,
            "num_tokens": num_tokens,
            "reward_model": item["reward_model"],
            "ability": item.get("ability", None),
            # todo: update train.jsonl
            "data_source": {item.get("data_source"): 1.0},
            "extra_info": extra_info,
        }

        return rl_out_data  # type: ignore

    def hash(self) -> str:
        raise ValueError("不应该触发这个方法, 因为 RLTextTokenizeFn 不需要缓存。")


class RLTextTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base RL dataset config for xtuner", extra="allow")
    max_length: int | None = None

    def build(self, tokenizer: "PreTrainedTokenizer", **kwargs) -> RLTextTokenizeFn:
        return RLTextTokenizeFn(
            tokenizer,
            max_length=self.max_length,
        )
