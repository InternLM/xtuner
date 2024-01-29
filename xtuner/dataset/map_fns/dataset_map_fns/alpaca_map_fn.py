# Copyright (c) OpenMMLab. All rights reserved.


def alpaca_map_fn(example):
    if example.get('output') == '<nooutput>':
        return {'conversation': []}
    else:
        return {
            'conversation': [{
                'input': f"{example['instruction']}\n{example['input']}",
                'output': example['output']
            }]
        }
