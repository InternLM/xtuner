# Copyright (c) OpenMMLab. All rights reserved.
def sql_dataset_map_fn(example):
    PROMPT = (
        'If you are an expert in SQL, please generate a good SQL Query for '
        'Question based on the CREATE TABLE statement.\n'
        '### Question: {context}\n{question}\n### Query: ')
    return {
        'conversation': [{
            'input': PROMPT.format(**example),
            'output': example['answer']
        }]
    }
