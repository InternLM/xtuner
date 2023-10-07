# Copyright (c) OpenMMLab. All rights reserved.
ALPACA_SYSTEM = ('Below is an instruction that describes a task. '
                 'Write a response that appropriately completes the request.')


def alpaca_map_fn(example):
    if example.get('output') == '<nooutput>':
        return {'conversation': []}
    else:
        return {
            'conversation': [{
                'system': ALPACA_SYSTEM,
                'input': f"{example['instruction']}\n{example['input']}",
                'output': example['output']
            }]
        }
