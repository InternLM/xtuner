# Copyright (c) OpenMMLab. All rights reserved.
ALPACA_SYSTEM = (
    'Below is an instruction that describes a task. '
    'Write a response that appropriately completes the request.\n')


def alpaca_zh_map_fn(example):
    return {
        'conversation': [{
            'system': ALPACA_SYSTEM,
            'input': f"{example['instruction_zh']}\n{example['input_zh']}",
            'output': example['output_zh']
        }]
    }
