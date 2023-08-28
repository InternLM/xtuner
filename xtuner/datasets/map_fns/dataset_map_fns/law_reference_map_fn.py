# Copyright (c) OpenMMLab. All rights reserved.
def law_reference_map_fn(example):
    return {
        'conversation': [{
            'input': example['question'],
            'output': example['answer']
        }]
    }
