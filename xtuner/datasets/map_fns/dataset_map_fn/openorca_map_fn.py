# Copyright (c) OpenMMLab. All rights reserved.
def openorca_dataset_map_fn(example):
    return {
        'conversation': [{
            'input': example['question'],
            'output': example['response']
        }]
    }
