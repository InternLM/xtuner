# Copyright (c) OpenMMLab. All rights reserved.
def arxiv_dataset_map_fn(example):
    return {
        'conversation': [{
            'input': example['abstract'],
            'output': example['title']
        }]
    }
