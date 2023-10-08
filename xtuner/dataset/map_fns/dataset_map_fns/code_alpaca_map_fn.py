# Copyright (c) OpenMMLab. All rights reserved.
CODER_SYSTEM = ('You are a professional programer. Please provide the '
                'corresponding code based on the description of Human.\n')


def code_alpaca_map_fn(example):
    return {
        'conversation': [{
            'system': CODER_SYSTEM,
            'input': example['prompt'],
            'output': example['completion']
        }]
    }
