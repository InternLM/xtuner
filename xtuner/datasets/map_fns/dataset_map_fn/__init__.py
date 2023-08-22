# Copyright (c) OpenMMLab. All rights reserved.
from .alpaca_map_fn import alpaca_dataset_map_fn
from .alpaca_zh_map_fn import alpaca_zh_dataset_map_fn
from .arxiv_map_fn import arxiv_dataset_map_fn
from .cmd_map_fn import cmd_dataset_map_fn
from .oasst1_map_fn import oasst1_dataset_map_fn
from .openorca_map_fn import openorca_dataset_map_fn

__all__ = [
    'alpaca_dataset_map_fn', 'alpaca_zh_dataset_map_fn',
    'oasst1_dataset_map_fn', 'arxiv_dataset_map_fn', 'cmd_dataset_map_fn',
    'openorca_dataset_map_fn'
]
