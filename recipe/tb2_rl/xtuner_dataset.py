# Copyright (c) OpenMMLab. All rights reserved.
"""tb2-rl XTuner training tokenize function.

Reads jsonl records produced by ``recipe.tb2_rl.scripts.generate_jsonl``.
Each record fully describes a task — this module does field-mapping +
instruction tokenization, no ``task.toml`` parsing.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedTokenizer

from xtuner.v1.data_proto.rl_data import RLDatasetItem
from xtuner.v1.datasets.rl_tokenize_fn.rl_tokenize_fn import RLTokenizeFn
from xtuner.v1.datasets.utils import CachableTokenizeFunction
from xtuner.v1.rl.agent_loop.rl_task.schemas import AgentRolloutItem
from xtuner.v1.utils import get_logger

logger = get_logger()


class RLTB2RLTokenizeFn(RLTokenizeFn):
    def __init__(
        self,
        tokenizer_fn: CachableTokenizeFunction | None,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
        ignore_multimodal_info: bool = False,
        data_judger_mapping: dict | None = None,
        system_prompt: str | None = None,
        tokenizer_hash: str | None = None,
        hash: str | None = None,
    ):
        super().__init__(tokenizer_fn=tokenizer_fn, tokenizer=tokenizer)
        self.tokenizer_fn = tokenizer_fn
        self.max_length = max_length
        self.ignore_multimodal_info = ignore_multimodal_info
        self.data_judger_mapping = data_judger_mapping
        self.system_prompt = system_prompt
        self._tokenizer_hash = tokenizer_hash
        self._hash = hash

    def __call__(self, item: dict, **kwargs) -> RLDatasetItem:
        task_dir = Path(item["task_dir"])
        instruction_path = task_dir / item["instruction"]
        context = instruction_path.read_text(encoding="utf-8")
        if self.system_prompt:
            context = f"{self.system_prompt}\n\n{context}"
        data = self.tokenizer(context, add_special_tokens=False)
        prompt_token_ids = data["input_ids"]
        num_tokens = len(data["input_ids"])

        if self.state == "cache":
            if self.max_length is not None and num_tokens > self.max_length:
                num_tokens = 0  # will be filtered out by the dataset filter
        else:
            if self.max_length is not None:
                assert num_tokens <= self.max_length, f"num_tokens {num_tokens} > max_length {self.max_length}"

        rollout_item = AgentRolloutItem(
            id=item["id"],
            data_source="tb2-rl",
            ability=item.get("ability"),
            tags=item.get("tags", []),
            instruction=item["instruction"],
            task_root=task_dir,
            pipeline="recipe.tb2_rl.pipeline.runner",
            pipeline_overrides=item.get("pipeline_overrides", {}),
        )

        rl_out_data = {
            "messages": [{"role": "user", "content": context}],
            "input_ids": prompt_token_ids,
            "num_tokens": num_tokens,
            "proxy_attn_flops": float(num_tokens),  # unused for RL. for compatibility of jsonldataset
            "reward_model": {"style": "tb2_rl"},
            "ability": item.get("ability"),
            "data_source": {"tb2-rl": 1.0},
            "extra_info": {
                "rollout_item": rollout_item,
            },
        }
        return rl_out_data  # type: ignore

    def hash(self) -> str:
        raise ValueError("不应该触发这个方法, 因为 RLTokenizeFn 不需要缓存。")


class RLTB2RLTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base RL dataset config for xtuner", extra="forbid")
    tokenize_fn_cfg: BaseModel | None = None
    max_length: int | None = None
    ignore_multimodal_info: bool = False  # eval is True
    system_prompt: str | None = None
    hash: str | None = None

    def build(
        self, tokenizer: PreTrainedTokenizer, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs
    ) -> RLTB2RLTokenizeFn:
        tokenizer_fn = None
        if self.tokenize_fn_cfg:
            tokenizer_fn = self.tokenize_fn_cfg.build(
                tokenizer=tokenizer,
                tokenizer_hash=tokenizer_hash,
                anno_name=anno_name,
                **kwargs,
            )
        return RLTB2RLTokenizeFn(
            tokenizer_fn=tokenizer_fn,
            tokenizer=tokenizer,
            max_length=self.max_length,
            ignore_multimodal_info=self.ignore_multimodal_info,
            system_prompt=self.system_prompt,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
        )
