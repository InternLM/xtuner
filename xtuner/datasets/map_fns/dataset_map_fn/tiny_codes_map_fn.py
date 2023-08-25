# Copyright (c) OpenMMLab. All rights reserved.
def tiny_codes_map_fn(example):
    return {
        'conversation': [{
            'input': example['prompt'],
            'output': example['response']
        }]
    }
