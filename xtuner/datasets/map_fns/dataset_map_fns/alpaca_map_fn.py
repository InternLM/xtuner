# Copyright (c) OpenMMLab. All rights reserved.
def alpaca_map_fn(example):
    if example.get('output') == '<nooutput>':
        return {'conversation': [{'input': '', 'output': ''}]}
    else:
        return {
            'conversation': [{
                'input':
                '{instruction}\n{input}'.format(**example),
                'output':
                example['output']
            }]
        }
