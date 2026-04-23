from typing_extensions import NotRequired, TypedDict


class CacheItem(TypedDict):
    num_tokens: int
    num_img_tokens: NotRequired[list[int]]
    proxy_attn_flops: NotRequired[float]


__all__ = ["CacheItem"]
