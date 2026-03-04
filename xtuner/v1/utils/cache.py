# Copyright (c) OpenMMLab. All rights reserved.
"""Minimal cache types shared by data_proto and datasets to avoid circular
imports."""

from typing_extensions import TypedDict


class CacheDict(TypedDict, total=False):
    num_tokens: int


class CacheObj:
    num_tokens: int | None = None
