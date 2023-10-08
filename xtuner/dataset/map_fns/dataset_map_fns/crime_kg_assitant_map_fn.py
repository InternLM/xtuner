# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import TASK_TEMPLATE


def crime_kg_assitant_map_fn(example):
    return {
        'conversation': [{
            'system': TASK_TEMPLATE.lawyer,
            'input': example['input'],
            'output': example['output']
        }]
    }
