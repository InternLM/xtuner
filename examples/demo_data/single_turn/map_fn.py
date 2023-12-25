# Copyright (c) OpenMMLab. All rights reserved.
def single_turn_map_fn(example):
    return {
        'conversation': [{
            'system': example['toy_system'],
            'input': example['toy_input'],
            'output': example['toy_output']
        }]
    }
