# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import SYSTEM_TEMPLATE
system = SYSTEM_TEMPLATE.zephyr_beta
def ultrafeedback_binarized_map_fn(example):
    r"""Example before preprocessing:
        example = {
                'prompt': 'xxx',
                'chosen': '[ { "content": "xxx", "role": "user" }, { "content": "xxx", "role": "assistant" } ]',
                'rejected': '[ { "content": "xxx", "role": "user" }, { "content": "xxx", "role": "assistant" } ]',
        }

    Example after preprocessing:
        example = {
                'system': 'xxx',
                'prompt': 'xxx',
                'chosen': 'xxx',
                'rejected': 'xxx',
        }
    """
    return {
        'system': system,
        'prompt': example['prompt'],
        'chosen': example['chosen'][1]['content'],
        'rejected': example['rejected'][1]['content'],
    }