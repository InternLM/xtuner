# Copyright (c) OpenMMLab. All rights reserved.
from .dataset_map_fn import (alpaca_dataset_map_fn, alpaca_zh_dataset_map_fn,
                             arxiv_dataset_map_fn, cmd_dataset_map_fn,
                             oasst1_dataset_map_fn, openorca_dataset_map_fn)
from .prompt_template_map_fn import (
    alpaca_template_map_fn, arxiv_template_map_fn, cmd_template_map_fn,
    internlm_template_map_fn, llama2_template_map_fn, oasst1_template_map_fn)

__all__ = [
    'alpaca_dataset_map_fn', 'alpaca_zh_dataset_map_fn',
    'oasst1_dataset_map_fn', 'arxiv_dataset_map_fn', 'cmd_dataset_map_fn',
    'openorca_dataset_map_fn', 'internlm_template_map_fn',
    'llama2_template_map_fn', 'arxiv_template_map_fn',
    'alpaca_template_map_fn', 'cmd_template_map_fn', 'oasst1_template_map_fn'
]
