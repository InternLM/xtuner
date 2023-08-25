# Copyright (c) OpenMMLab. All rights reserved.
def tiny_codes_map_fn(example):
    return {
        'conversation': [{
            'input':
            '### Human: {prompt}\n### Bot: '.format(**example),
            'output':
            example['response']
        }]
    }
