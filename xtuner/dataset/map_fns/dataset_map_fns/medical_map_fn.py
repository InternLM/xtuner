# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import TASK_TEMPLATE


def medical_map_fn(example):
    return {
        'conversation': [{
            'system': TASK_TEMPLATE.medical,
            'input': '{instruction}\n{input}'.format(**example),
            'output': example['output']
        }]
    }
