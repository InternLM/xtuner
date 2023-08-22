# Copyright (c) OpenMMLab. All rights reserved.
from .alpaca_map_fn import alpaca_dataset_map_fn
from .alpaca_zh_map_fn import alpaca_zh_dataset_map_fn
from .arxiv_map_fn import arxiv_dataset_map_fn
from .cmd_map_fn import cmd_dataset_map_fn
from .code_alpaca_map_fn import code_alpaca_dataset_map_fn
from .colors_map_fn import colors_dataset_map_fn
from .crime_kg_assitant_map_fn import crime_kg_assitant_dataset_map_fn
from .law_reference_map_fn import law_reference_dataset_map_fn
from .oasst1_map_fn import oasst1_dataset_map_fn
from .openorca_map_fn import openorca_dataset_map_fn
from .sql_map_fn import sql_dataset_map_fn
from .tiny_codes_map_fn import tiny_codes_dataset_map_fn

__all__ = [
    'alpaca_dataset_map_fn', 'alpaca_zh_dataset_map_fn',
    'oasst1_dataset_map_fn', 'arxiv_dataset_map_fn', 'cmd_dataset_map_fn',
    'openorca_dataset_map_fn', 'code_alpaca_dataset_map_fn',
    'tiny_codes_dataset_map_fn', 'colors_dataset_map_fn',
    'law_reference_dataset_map_fn', 'crime_kg_assitant_dataset_map_fn',
    'sql_dataset_map_fn'
]
