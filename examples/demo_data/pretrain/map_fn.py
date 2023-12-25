# Copyright (c) OpenMMLab. All rights reserved.
def pretrain_map_fn(example):
    return {
        'conversation': [{
            'input': '',
            'output': example['toy_text'].strip()
        }]
    }
