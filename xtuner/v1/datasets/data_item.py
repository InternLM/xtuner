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
