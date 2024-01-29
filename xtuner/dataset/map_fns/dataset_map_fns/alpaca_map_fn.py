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


def alpaca_map_fn_v2(example):
    messages = []
    if example.get('output') == '<nooutput>':
        return {'messages': []}
    else:

        messages.append({'role': 'system', 'content': SYSTEM_TEMPLATE.alpaca})
        messages.append({
            'role':
            'user',
            'content':
            f"{example['instruction']}\n{example['input']}"
        })
        messages.append({'role': 'assistant', 'content': example['output']})
        return {'messages': messages}
