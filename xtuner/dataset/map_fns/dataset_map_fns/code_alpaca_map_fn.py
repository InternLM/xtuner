# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import TASK_TEMPLATE


def code_alpaca_map_fn(example):
    return {
        'conversation': [{
            'system': TASK_TEMPLATE.coder,
            'input': example['prompt'],
            'output': example['completion']
        }]
    }
