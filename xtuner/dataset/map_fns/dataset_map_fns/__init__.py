# Copyright (c) OpenMMLab. All rights reserved.
from .alpaca_map_fn import alpaca_map_fn, alpaca_map_fn_v2
from .alpaca_zh_map_fn import alpaca_zh_map_fn
from .arxiv_map_fn import arxiv_map_fn
from .code_alpaca_map_fn import code_alpaca_map_fn
from .colors_map_fn import colors_map_fn
from .crime_kg_assitant_map_fn import crime_kg_assitant_map_fn
from .default_map_fn import default_map_fn
from .law_reference_map_fn import law_reference_map_fn
from .llava_map_fn import llava_image_only_map_fn, llava_map_fn
from .medical_map_fn import medical_map_fn
from .msagent_map_fn import msagent_react_map_fn
from .oasst1_map_fn import oasst1_map_fn, oasst1_map_fn_v2
from .openai_map_fn import openai_map_fn
from .openorca_map_fn import openorca_map_fn
from .pretrain_map_fn import pretrain_map_fn
from .sql_map_fn import sql_map_fn
from .stack_exchange_map_fn import stack_exchange_map_fn
from .tiny_codes_map_fn import tiny_codes_map_fn
from .wizardlm_map_fn import wizardlm_map_fn

__all__ = [
    'alpaca_map_fn', 'alpaca_zh_map_fn', 'oasst1_map_fn', 'arxiv_map_fn',
    'medical_map_fn', 'openorca_map_fn', 'code_alpaca_map_fn',
    'tiny_codes_map_fn', 'colors_map_fn', 'law_reference_map_fn',
    'crime_kg_assitant_map_fn', 'sql_map_fn', 'openai_map_fn',
    'wizardlm_map_fn', 'stack_exchange_map_fn', 'msagent_react_map_fn',
    'pretrain_map_fn', 'default_map_fn', 'llava_image_only_map_fn',
    'llava_map_fn', 'oasst1_map_fn_v2', 'alpaca_map_fn_v2'
]
