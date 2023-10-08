# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import TASK_TEMPLATE


def arxiv_map_fn(example):
    return {
        'conversation': [{
            'system': TASK_TEMPLATE.arxiv_gentile,
            'input': example['abstract'],
            'output': example['title']
        }]
    }
