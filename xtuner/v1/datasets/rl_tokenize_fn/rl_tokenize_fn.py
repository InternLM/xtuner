# Copyright (c) OpenMMLab. All rights reserved.
from pydantic import BaseModel, ConfigDict
import os

from xtuner.v1.datasets.data_item import RLTextDataItem
from xtuner.v1.utils import get_logger

from ..intern_s1_fn.tokenizer_fn import InternS1TokenizeFunction
from ..utils import CachableTokenizeFunction


logger = get_logger()


# https://github.com/volcengine/verl/blob/main/verl/utils/dataset/rl_dataset.py
class RLTokenizeFn(CachableTokenizeFunction[RLTextDataItem]):
    def __init__(self, tokenizer_fn: CachableTokenizeFunction,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_fn = tokenizer_fn

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
                "image": image_path, # if multi-modal
                "image_wh": [[640, 480]] # if multi-modal
            }
        """
        messages = item["prompt"]
        extra_info = item.get("extra_info", {})
        # TODO: 多模态需要支持 openai message 格式输入，否则这个地方要特殊处理
        if isinstance(self.tokenizer_fn, InternS1TokenizeFunction):
            assert 'media_root' in kwargs
            new_jsonl_dict = {"image": item["image"], "image_wh": item["image_wh"]}

            # TODO: Hard code now
            prompt = "<image>" + messages[0]['content'][1]['text']
            conversations = [{"from": "human", "value": prompt}]
            new_jsonl_dict["conversations"] = conversations
            num_tokens = self.tokenizer_fn(new_jsonl_dict, media_root=kwargs['media_root'])["num_tokens"]

            extra_info["image"] = os.path.join(kwargs['media_root'], item["image"])
            extra_info["image_wh"] = item["image_wh"]
        else:
            num_tokens = self.tokenizer_fn(messages,)["num_tokens"]

        rl_out_data = {
            # "input_ids": input_ids,
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
    sft_tokenize_fn_cfg: BaseModel  # TODO: 如何写 typehint

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs
    ) -> RLTokenizeFn:
        sft_tokenizer_fn = self.sft_tokenize_fn_cfg.build(
            tokenizer=tokenizer,
            tokenizer_hash=tokenizer_hash,
            anno_name=anno_name,
            **kwargs,
        )
        return RLTokenizeFn(
            sft_tokenizer_fn,
        )
