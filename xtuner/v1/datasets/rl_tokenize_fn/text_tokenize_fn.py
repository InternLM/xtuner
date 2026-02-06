# Copyright (c) OpenMMLab. All rights reserved.
from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto import RolloutState
from xtuner.v1.utils import get_logger

from ..utils import CachableTokenizeFunction


logger = get_logger()


class RLTextTokenizeFn(CachableTokenizeFunction[RolloutState]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
    ):
        super().__init__(tokenizer)
        self.max_length = max_length

    def __call__(self, item: dict, **kwargs) -> RolloutState:
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
        message = item["prompt"]

        raw_prompt = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        data = self.tokenizer(raw_prompt, add_special_tokens=False)
        prompt_token_ids = data["input_ids"]
        num_tokens = len(data["input_ids"])

        if self.state == "cache":
            if self.max_length is not None and num_tokens > self.max_length:
                num_tokens = 0  # will be filtered out by the dataset filter
        else:
            if self.max_length is not None:
                assert num_tokens <= self.max_length, f"num_tokens {num_tokens} > max_length {self.max_length}"

        rollout_state = RolloutState(
            prompt_ids=prompt_token_ids,
            message=message,
            data_source=item.get("data_source", "default"),
            reward_model=item.get("reward_model", {}),
            num_tokens=num_tokens,
            extra_fields=extra_info,
        )
        return rollout_state

    def hash(self) -> str:
        raise ValueError("不应该触发这个方法, 因为 RLTokenizeFn 不需要缓存。")


class RLTextTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base RL dataset config for xtuner", extra="forbid")
    max_length: int | None = None

    def build(self, tokenizer: PreTrainedTokenizer, **kwargs) -> RLTextTokenizeFn:
        return RLTextTokenizeFn(tokenizer=tokenizer, max_length=self.max_length)
