# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import SYSTEM_TEMPLATE


def medical_map_fn(example):
    return {
        'conversation': [{
            'system': SYSTEM_TEMPLATE.medical,
            'input': '{instruction}\n{input}'.format(**example),
            'output': example['output']
        }]
    }
