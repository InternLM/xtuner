# Copyright (c) OpenMMLab. All rights reserved.
from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.utils import get_logger

from ..utils import CachableTokenizeFunction


logger = get_logger()


class RLTextTokenizeFn(CachableTokenizeFunction[RolloutState]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
        tools_schema: list | None = None,
        data_judger_mapping: dict | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__(tokenizer)
        self.max_length = max_length
        self.tools_schema = tools_schema if tools_schema is not None else []
        self.data_judger_mapping = data_judger_mapping
        self.system_prompt = system_prompt

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

        if self.system_prompt:
            if message[0]["role"] == "system":
                message = message[1:]
            message = [{"role": "system", "content": self.system_prompt}] + message
        raw_prompt = self.tokenizer.apply_chat_template(
            message, tools=self.tools_schema, add_generation_prompt=True, tokenize=False
        )
        extra_info["raw_prompt"] = raw_prompt
        data = self.tokenizer(raw_prompt, add_special_tokens=False)
        prompt_token_ids = data["input_ids"]
        num_tokens = len(data["input_ids"])

        if self.state == "cache":
            if self.max_length is not None and num_tokens > self.max_length:
                num_tokens = 0  # will be filtered out by the dataset filter
        else:
            if self.max_length is not None:
                assert num_tokens <= self.max_length, f"num_tokens {num_tokens} > max_length {self.max_length}"

        mapped_judger_name_and_weight = None
        if self.state != "cache":
            data_source = item.get("data_source")
            assert data_source is not None, "data_source is required in item"
            extra_info["origin_data_source"] = data_source
            if self.data_judger_mapping is not None:
                mapped_judger_name_and_weight = self.data_judger_mapping.get(data_source)
            else:
                mapped_judger_name_and_weight = {data_source: 1.0}

        rollout_state = RolloutState(
            prompt_ids=prompt_token_ids,
            message=message,
            reward_model=item.get("reward_model", {}),
            num_tokens=num_tokens,
            proxy_attn_flops=float(num_tokens),
            data_source=mapped_judger_name_and_weight,
            extra_fields=extra_info,
        )
        return rollout_state

    def hash(self) -> str:
        return "RLTextTokenizeFn"


class RLTextTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Text RL dataset config for xtuner", extra="forbid")
    max_length: int | None = None
    tools_schema: list | None = None

    def build(self, tokenizer: PreTrainedTokenizer, **kwargs) -> RLTextTokenizeFn:
        return RLTextTokenizeFn(tokenizer=tokenizer, max_length=self.max_length, tools_schema=self.tools_schema)
