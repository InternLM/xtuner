# Copyright (c) OpenMMLab. All rights reserved.
def alpaca_map_fn(example):
    if example.get('output') == '<nooutput>':
        return {'conversation': [{'input': '', 'output': ''}]}
    else:
        input = example.get('input', '')
        instruction = example['instruction']
        return {
            'conversation': [{
                'input':
                f'{instruction} {input}' if input != '' else instruction,
                'output':
                example['output']
            }]
        }
