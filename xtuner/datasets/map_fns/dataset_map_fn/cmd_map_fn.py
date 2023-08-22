# Copyright (c) OpenMMLab. All rights reserved.
def cmd_dataset_map_fn(example):
    if example.get('ask', '') == '无':
        return {'conversation': [{'input': '', 'output': ''}]}
    return {
        'conversation': [{
            'input': example['ask'],
            'output': example['answer']
        }]
    }
