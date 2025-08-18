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


class RLTextDataItem(TypedDict):
    input_ids: list[int]
    num_tokens: int
    data_source: str | None  # e.g., math, code
    ability: str | None  # math, code
    reward_model: dict
    extra_info: dict | None
