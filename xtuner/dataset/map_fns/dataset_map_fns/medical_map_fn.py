# Copyright (c) OpenMMLab. All rights reserved.
MEDICAL_STSTEM = '如果你是一名医生，请根据患者的描述回答医学问题。\n'


def medical_map_fn(example):
    return {
        'conversation': [{
            'system': MEDICAL_STSTEM,
            'input': '{instruction}\n{input}'.format(**example),
            'output': example['output']
        }]
    }
