import torch
from typing_extensions import NotRequired, TypedDict


class CacheItem(TypedDict):
    num_tokens: int
    num_img_tokens: NotRequired[list[int]]
    proxy_attn_flops: NotRequired[float]


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


class Glm53VLDataItem(BaseMLLMDataItem, total=False):
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor
    mm_token_type_ids: torch.Tensor  # 0=text, 1=image, 2=video


class LongTextDataItem(DataItem):
    char_start: int
    char_end: int
    token_start_offset: int
