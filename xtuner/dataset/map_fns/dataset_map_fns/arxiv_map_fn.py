# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import SYSTEM_TEMPLATE


def arxiv_map_fn(example):
    return {
        'conversation': [{
            'system': SYSTEM_TEMPLATE.arxiv_gentile,
            'input': example['abstract'],
            'output': example['title']
        }]
    }
