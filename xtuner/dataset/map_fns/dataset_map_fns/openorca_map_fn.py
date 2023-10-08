# Copyright (c) OpenMMLab. All rights reserved.
def openorca_map_fn(example):
    return {
        'conversation': [{
            'system': example['system_prompt'],
            'input': example['question'],
            'output': example['response']
        }]
    }
