# Copyright (c) OpenMMLab. All rights reserved.
SQL_SYSTEM = ('If you are an expert in SQL, please generate a good SQL Query '
              'for Question based on the CREATE TABLE statement.\n')


def sql_map_fn(example):
    return {
        'conversation': [{
            'system': SQL_SYSTEM,
            'input': '{context}\n{question}'.format(**example),
            'output': example['answer']
        }]
    }
