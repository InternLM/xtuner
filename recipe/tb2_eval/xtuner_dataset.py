# Copyright (c) OpenMMLab. All rights reserved.
"""tb2-eval XTuner training tokenize function.

Reads jsonl records produced by ``recipe.tb2_eval.scripts.generate_jsonl``
and emits ``RolloutState`` for the trainer.  The per-task
``AgentRolloutItem`` (carrying pipeline + per-task overrides) lives in
``RolloutState.extra_fields["rollout_item"]`` for the runner to consume at
rollout time.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedTokenizer

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.datasets.data_item import CacheItem
from xtuner.v1.datasets.utils import CachableTokenizeFunction
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem
from xtuner.v1.utils import get_logger

logger = get_logger()


class RLTB2EvalTokenizeFn(CachableTokenizeFunction[RolloutState]):
    """tb2-eval tokenize function aligned with ``RLTextTokenizeFn``."""

    DATA_SOURCE_NAME = "tb2-eval"
    PIPELINE_DOTTED = "recipe.tb2_eval.pipeline.runner"

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

    def __call__(self, item: dict, **kwargs) -> RolloutState | CacheItem:
        task_dir = Path(item["task_dir"])
        instruction_path = task_dir / item["instruction"]
        context = instruction_path.read_text(encoding="utf-8")

        message = [{"role": "user", "content": context}]
        if self.system_prompt:
            message = [{"role": "system", "content": self.system_prompt}] + message

        raw_prompt = self.tokenizer.apply_chat_template(
            message, tools=self.tools_schema, add_generation_prompt=True, tokenize=False
        )
        data = self.tokenizer(raw_prompt, add_special_tokens=False)
        prompt_token_ids = data["input_ids"]
        num_tokens = len(prompt_token_ids)

        if self.state == "cache":
            if self.max_length is not None and num_tokens > self.max_length:
                num_tokens = 0  # filtered out by the dataset filter
            return CacheItem(
                num_tokens=num_tokens,
                proxy_attn_flops=float(num_tokens),
            )

        if self.max_length is not None:
            assert num_tokens <= self.max_length, f"num_tokens {num_tokens} > max_length {self.max_length}"

        if self.data_judger_mapping is not None:
            data_source = self.data_judger_mapping.get(self.DATA_SOURCE_NAME)
        else:
            data_source = {self.DATA_SOURCE_NAME: 1.0}

        rollout_item = AgentRolloutItem(
            id=item["id"],
            data_source=self.DATA_SOURCE_NAME,
            ability=item.get("ability"),
            tags=item.get("tags", []),
            instruction=item["instruction"],
            task_root=task_dir,
            pipeline=self.PIPELINE_DOTTED,
            pipeline_overrides=item.get("pipeline_overrides", {}),
        )

        return RolloutState(
            prompt_ids=prompt_token_ids,
            message=message,
            reward_model={"style": self.DATA_SOURCE_NAME},
            num_tokens=num_tokens,
            proxy_attn_flops=float(num_tokens),
            data_source=data_source,
            extra_fields={
                "rollout_item": rollout_item,
                "ability": item.get("ability"),
            },
        )

    def hash(self) -> str:
        return type(self).__name__


class RLTB2EvalTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="tb2-eval RL dataset config for xtuner", extra="forbid")
    max_length: int | None = None
    tools_schema: list | None = None
    data_judger_mapping: dict | None = None
    system_prompt: str | None = None

    def build(self, tokenizer: PreTrainedTokenizer, **kwargs) -> RLTB2EvalTokenizeFn:
        return RLTB2EvalTokenizeFn(
            tokenizer=tokenizer,
            max_length=self.max_length,
            tools_schema=self.tools_schema,
            data_judger_mapping=self.data_judger_mapping,
            system_prompt=self.system_prompt,
        )
