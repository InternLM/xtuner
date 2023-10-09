# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import SYSTEM_TEMPLATE


def law_reference_map_fn(example):
    return {
        'conversation': [{
            'system': SYSTEM_TEMPLATE.lawyer,
            'input': example['question'],
            'output': example['answer']
        }]
    }
