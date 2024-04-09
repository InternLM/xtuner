# Copyright (c) OpenMMLab. All rights reserved.
def pretrain_map_fn(example):
    r"""Example before preprocessing:
        example['text'] = 'xxx'

    Example after preprocessing:
        example['conversation'] = [
            {
                'input': '',
                'output': 'xxx'
            },
        ]
    """
    return {
        'conversation': [{
            'input': '',
            'output': example['text'].strip(),
            'need_eos_token': False
        }]
    }
