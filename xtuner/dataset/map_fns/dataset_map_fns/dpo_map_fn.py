# Copyright (c) OpenMMLab. All rights reserved.
def dpo_map_fn(example):
    r"""Example before preprocessing:
        example = {
                'system': 'xxx',
                'question': 'xxx',
                'chosen': 'xxx',
                'rejected': 'xxx',
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
                'system': example['system'],
                'prompt': example['question'],
                'chosen': example['chosen'],
                'rejected': example['rejected'],
    }
