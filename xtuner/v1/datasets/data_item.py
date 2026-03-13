import torch
from typing_extensions import NotRequired, TypedDict


class CacheItem(TypedDict, total=False):
    num_tokens: int
    num_img_tokens: NotRequired[list[int]]


class DataItem(CacheItem):
    input_ids: list[int]
    labels: list[int]


class BaseMLLMDataItem(DataItem):
    num_imgs: list[int]


class InternS1DataItem(BaseMLLMDataItem, total=False):
    pixel_values: torch.Tensor


class QwenVL3DataItem(BaseMLLMDataItem, total=False):
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    position_ids: torch.Tensor


class OmniDataItem(BaseMLLMDataItem, total=False):
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    position_ids: torch.Tensor
