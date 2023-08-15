# Copyright (c) OpenMMLab. All rights reserved.
def oasst1_map_fn(example):
    return {'conversation': [{'input': '', 'output': example['text']}]}
