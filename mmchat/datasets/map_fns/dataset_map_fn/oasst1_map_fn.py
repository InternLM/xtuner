# Copyright (c) OpenMMLab. All rights reserved.
def oasst1_map_fn(example):
    return {'input': [''], 'output': [example['text']]}
