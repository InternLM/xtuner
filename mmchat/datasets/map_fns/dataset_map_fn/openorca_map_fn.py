# Copyright (c) OpenMMLab. All rights reserved.
def openorca_map_fn(example):
    PROMPT = ('Below is an instruction that describes a task. '
              'Write a response that appropriately completes the request.\n\n'
              '### Instruction:\n{question}\n\n'
              '### Response: ')

    return {
        'conversation': [{
            'input': PROMPT.format(**example),
            'output': example['response']
        }]
    }
