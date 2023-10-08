# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import TASK_TEMPLATE


def sql_map_fn(example):
    return {
        'conversation': [{
            'system': TASK_TEMPLATE.sql,
            'input': '{context}\n{question}'.format(**example),
            'output': example['answer']
        }]
    }
