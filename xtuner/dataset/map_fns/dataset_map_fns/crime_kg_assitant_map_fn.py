# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import SYSTEM_TEMPLATE


def crime_kg_assitant_map_fn(example):
    return {
        'conversation': [{
            'system': SYSTEM_TEMPLATE.lawyer,
            'input': example['input'],
            'output': example['output']
        }]
    }
