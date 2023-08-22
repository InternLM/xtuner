# Copyright (c) OpenMMLab. All rights reserved.
from .alpaca_template_map_fn import alpaca_template_map_fn
from .arxiv_template_map_fn import arxiv_template_map_fn
from .cmd_template_map_fn import cmd_template_map_fn
from .internlm_template_map_fn import internlm_template_map_fn
from .llama2_template_map_fn import llama2_template_map_fn
from .oasst1_template_map_fn import oasst1_template_map_fn

__all__ = [
    'alpaca_template_map_fn', 'internlm_template_map_fn',
    'llama2_template_map_fn', 'arxiv_template_map_fn', 'cmd_template_map_fn',
    'oasst1_template_map_fn'
]
