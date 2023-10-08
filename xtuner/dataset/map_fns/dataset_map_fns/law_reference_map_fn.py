# Copyright (c) OpenMMLab. All rights reserved.
LAWYER_SYSTEM = '你现在是一名专业的中国律师，请根据Human的问题给出准确、有理有据的回复。\n'


def law_reference_map_fn(example):
    return {
        'conversation': [{
            'system': LAWYER_SYSTEM,
            'input': example['question'],
            'output': example['answer']
        }]
    }
