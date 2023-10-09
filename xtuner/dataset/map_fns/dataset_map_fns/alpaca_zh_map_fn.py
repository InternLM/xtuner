# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import SYSTEM_TEMPLATE


def alpaca_zh_map_fn(example):
    return {
        'conversation': [{
            'system': SYSTEM_TEMPLATE.alpaca,
            'input': f"{example['instruction_zh']}\n{example['input_zh']}",
            'output': example['output_zh']
        }]
    }
