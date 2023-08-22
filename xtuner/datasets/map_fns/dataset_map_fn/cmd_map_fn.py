# Copyright (c) OpenMMLab. All rights reserved.
def cmd_dataset_map_fn(example):
    return {
        'conversation': [{
            'input': example['ask'],
            'output': example['answer']
        }]
    }
