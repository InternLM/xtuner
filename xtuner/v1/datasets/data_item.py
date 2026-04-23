import torch
from typing_extensions import NotRequired, TypedDict

from xtuner.v1.data_proto.cache_item import CacheItem


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


class LongTextDataItem(DataItem):
    char_start: int
    char_end: int
    token_start_offset: int
