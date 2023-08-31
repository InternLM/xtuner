# Copyright (c) OpenMMLab. All rights reserved.
def stack_exchange_map_fn(example):
    return {
        'conversation': [{
            'input': example['question'],
            'output': example['response']
        }]
    }
