# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import SYSTEM_TEMPLATE


def colors_map_fn(example):
    desc = ':'.join(example['description'].split(':')[1:]).strip()
    return {
        'conversation': [{
            'system': SYSTEM_TEMPLATE.colorist,
            'input': desc,
            'output': example['color']
        }]
    }
