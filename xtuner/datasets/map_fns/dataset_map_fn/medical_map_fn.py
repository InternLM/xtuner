# Copyright (c) OpenMMLab. All rights reserved.
def medical_map_fn(example):
    return {
        'conversation': [{
            'input': '{instruction}\n{input}'.format(**example),
            'output': example['output']
        }]
    }
