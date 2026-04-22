# Copyright (c) OpenMMLab. All rights reserved.
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.rl_data import RLDatasetItem
from xtuner.v1.utils import get_logger

from xtuner.v1.utils import CachableTokenizeFunction
from xtuner.v1.datasets.rl_tokenize_fn import RLTokenizeFn
from xtuner.v1.ray.environment.rl_task.schemas import TaskData

logger = get_logger()


def _load_task_toml(path: Path) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if "task" in raw and isinstance(raw["task"], dict):
        task_section = raw.pop("task")
        for k, v in task_section.items():
            raw.setdefault(k, v)
    return raw


class RLClawTokenizeFn(RLTokenizeFn):
    def __init__(
        self,
        root_path: str,
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
        self.root_path = root_path
        self.tokenizer_fn = tokenizer_fn
        self.max_length = max_length
        self.ignore_multimodal_info = ignore_multimodal_info
        self.data_judger_mapping = data_judger_mapping
        self.system_prompt = system_prompt
        self._tokenizer_hash = tokenizer_hash
        self._hash = hash

    def __call__(self, item: dict, **kwargs) -> RLDatasetItem:
        task_dir = item["task_dir"]
        task_dir = Path(self.root_path) / task_dir
        toml = _load_task_toml(task_dir / "task.toml")

        instruction_path = task_dir / "instruction.md"
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
        
        task_data = TaskData(
            id=toml.get("id") or task_dir.name,
            data_source=self.name,
            ability=toml.get("domain"),
            tags=list(toml.get("tags") or []),
            instruction="instruction.md",
        )

        rl_out_data = {
            "messages": [{"role": "user", "content": context}],
            "input_ids": prompt_token_ids,
            "num_tokens": num_tokens,
            "proxy_attn_flops": float(num_tokens),  # unused for RL. for comatibility of jsonldataset
            "reward_model": {"style": "claw"},
            "ability": toml.get("domain", None),
            "data_source": {"claw-bench": 1.0},
            "extra_info": {
                "task_data": task_data,
                "task_dir": item["task_dir"],
                "pipeline": "claw_bench.pipeline.claw_pipeline",
            },
        }
        return rl_out_data  # type: ignore

    def hash(self) -> str:
        raise ValueError("不应该触发这个方法, 因为 RLTokenizeFn 不需要缓存。")


class RLClawTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base RL dataset config for xtuner", extra="forbid")
    root_path: str
    tokenize_fn_cfg: BaseModel | None = None
    max_length: int | None = None
    ignore_multimodal_info: bool = False  # eval is True
    system_prompt: str | None = None
    hash: str | None = None

    def build(
        self, tokenizer: PreTrainedTokenizer, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs
    ) -> RLClawTokenizeFn:
        tokenizer_fn = None
        if self.tokenize_fn_cfg:
            tokenizer_fn = self.tokenize_fn_cfg.build(
                tokenizer=tokenizer,
                tokenizer_hash=tokenizer_hash,
                anno_name=anno_name,
                **kwargs,
            )
        return RLClawTokenizeFn(
            root_path=self.root_path,
            tokenizer_fn=tokenizer_fn,
            tokenizer=tokenizer,
            max_length=self.max_length,
            ignore_multimodal_info=self.ignore_multimodal_info,
            system_prompt=self.system_prompt,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
        )
