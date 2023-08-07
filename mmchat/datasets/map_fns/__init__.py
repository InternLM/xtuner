# Copyright (c) OpenMMLab. All rights reserved.
from .dataset_map_fn import (alpaca_map_fn, alpaca_zh_map_fn, arxiv_map_fn,
                             cmd_map_fn, oasst1_map_fn)
from .model_map_fn import internlm_map_fn, llama2_map_fn

__all__ = [
    'alpaca_map_fn', 'alpaca_zh_map_fn', 'oasst1_map_fn', 'arxiv_map_fn',
    'cmd_map_fn', 'internlm_map_fn', 'llama2_map_fn'
]
