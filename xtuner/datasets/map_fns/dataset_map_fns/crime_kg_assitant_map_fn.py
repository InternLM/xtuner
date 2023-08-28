# Copyright (c) OpenMMLab. All rights reserved.
def crime_kg_assitant_map_fn(example):
    return {
        'conversation': [{
            'input': example['input'],
            'output': example['output']
        }]
    }
