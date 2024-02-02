# Copyright (c) OpenMMLab. All rights reserved.


def alpaca_zh_map_fn(example):
    return {
        'conversation': [{
            'input': f"{example['instruction_zh']}\n{example['input_zh']}",
            'output': example['output_zh']
        }]
    }
