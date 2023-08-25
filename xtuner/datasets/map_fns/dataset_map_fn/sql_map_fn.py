# Copyright (c) OpenMMLab. All rights reserved.
def sql_map_fn(example):
    context = example['context']
    question = example['question']
    return {
        'conversation': [{
            'input': f'{context}\n{question}',
            'output': example['answer']
        }]
    }
