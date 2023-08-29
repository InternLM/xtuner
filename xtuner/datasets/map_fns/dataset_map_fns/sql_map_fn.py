# Copyright (c) OpenMMLab. All rights reserved.
def sql_map_fn(example):
    return {
        'conversation': [{
            'input': '{context}\n{question}'.format(**example),
            'output': example['answer']
        }]
    }
