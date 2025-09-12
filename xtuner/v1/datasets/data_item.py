from typing import Any, Dict, List, Optional

import torch
from typing_extensions import TypedDict


class CacheItem(TypedDict):
    num_tokens: int


class DataItem(CacheItem):
    input_ids: list[int]
    labels: list[int]


class BaseMLLMDataItem(DataItem):
    num_img_tokens: list[int] | None
    num_imgs: list[int] | None
    num_patches: list[int] | None
    pixel_values: Optional[torch.Tensor]


class InternS1DataItem(BaseMLLMDataItem):
    image_flags: torch.Tensor


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
