# Copyright (c) OpenMMLab. All rights reserved.
def code_alpaca_dataset_map_fn(example):
    return {
        'conversation': [{
            'input':
            '### Human: {prompt}\n### Bot: '.format(**example),
            'output':
            example['completion']
        }]
    }
