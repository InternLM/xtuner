# Copyright (c) OpenMMLab. All rights reserved.
def alpaca_zh_map_fn(example):
    input = example.get('input_zh', '')
    instruction = example['instruction_zh']

    return {
        'conversation': [{
            'input': f'{instruction} {input}' if input != '' else instruction,
            'output': example['output_zh']
        }]
    }
