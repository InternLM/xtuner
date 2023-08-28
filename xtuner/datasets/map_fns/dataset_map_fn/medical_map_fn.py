# Copyright (c) OpenMMLab. All rights reserved.
def medical_map_fn(example):
    input = example.get('input', '')
    instruction = example['instruction']
    return {
        'conversation': [{
            'input':
            f'{instruction}. {input}' if input != '' else instruction,
            'output':
            example['output']
        }]
    }
