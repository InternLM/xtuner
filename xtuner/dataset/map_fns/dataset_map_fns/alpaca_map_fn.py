# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import SYSTEM_TEMPLATE


def alpaca_map_fn(example):
    if example.get('output') == '<nooutput>':
        return {'conversation': []}
    else:
        return {
            'conversation': [{
                'system': SYSTEM_TEMPLATE.alpaca,
                'input': f"{example['instruction']}\n{example['input']}",
                'output': example['output']
            }]
        }
