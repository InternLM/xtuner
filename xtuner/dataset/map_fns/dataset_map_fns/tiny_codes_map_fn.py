# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import SYSTEM_TEMPLATE


def tiny_codes_map_fn(example):
    return {
        'conversation': [{
            'system': SYSTEM_TEMPLATE.coder,
            'input': example['prompt'],
            'output': example['response']
        }]
    }
