# Copyright (c) OpenMMLab. All rights reserved.
def code_alpaca_map_fn(example):
    return {
        'conversation': [{
            'input': example['prompt'],
            'output': example['completion']
        }]
    }
