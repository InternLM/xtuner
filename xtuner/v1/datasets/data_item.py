from typing import Any, Dict, List

import torch
from typing_extensions import TypedDict


class CacheItem(TypedDict):
    num_tokens: int


class DataItem(CacheItem):
    input_ids: list[int]
    labels: list[int]


class BaseMLLMDataItem(DataItem):
    num_img_tokens: list[int]
    num_imgs: list[int]
    num_patches: list[int]


class InternS1DataItem(BaseMLLMDataItem):
    pixel_values: torch.Tensor
    image_flags: torch.Tensor


class QwenVL3DataItem(BaseMLLMDataItem, total=False):
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    position_ids: torch.Tensor


class RLTextDataItem(CacheItem, total=False):
    env: str
    group_id: int
    prompt_id: int
    input_ids: list[int]
    messages: str | List[Dict[str, Any]]
    prompt: str
    data_source: dict | None  # e.g., {"math" : "0.8", "code": "0.2"}
    ability: str | None  # math, code
    reward_model: dict
    reward: float | None
    num_return_tokens: int | None
    response_ids: list[int] | None
    response_str: str | None
    state: str
    retry_times: int
    extra_info: dict
