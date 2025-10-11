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
