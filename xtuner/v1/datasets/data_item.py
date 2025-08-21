from typing import TypedDict

import torch


class DataItem(TypedDict):
    input_ids: list[int]
    labels: list[int]
    num_tokens: int


class InternS1DataItem(TypedDict):
    input_ids: list[int]
    labels: list[int]
    pixel_values: torch.Tensor
    num_tokens: int
    image_flags: torch.Tensor
    num_img_tokens: list[int]
    num_imgs: list[int]
    num_patches: list[int]


class RLTextDataItem(TypedDict, total=False):
    env: str
    group_id: int
    prompt_id: int
    input_ids: list[int]
    prompt_str: str
    num_tokens: int
    data_source: str | None  # e.g., math, code
    ability: str | None  # math, code
    reward_model: dict
    reward: float | None
    num_return_tokens: int | None
    response_ids: list[int] | None
    response_str: str | None
    state: str
    retry_times: int
    extra_info: dict
