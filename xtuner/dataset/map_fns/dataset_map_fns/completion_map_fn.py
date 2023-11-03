# Copyright (c) OpenMMLab. All rights reserved.
def completion_map_fn(example):
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
    return {'conversation': [{'input': '', 'output': example['text'].strip()}]}
