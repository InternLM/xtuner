# Copyright (c) OpenMMLab. All rights reserved.
def alpaca_zh_map_fn(example):
    return {
        'conversation': [{
            'input':
            '{instruction_zh}\n{input_zh}'.format(**example),
            'output':
            example['output_zh']
        }]
    }
