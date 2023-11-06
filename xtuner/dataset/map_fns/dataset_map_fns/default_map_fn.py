# Copyright (c) OpenMMLab. All rights reserved.
def default_map_fn(example):
    return {
        'conversation': [{
            'input': example['input'],
            'output': example['output']
        }]
    }
