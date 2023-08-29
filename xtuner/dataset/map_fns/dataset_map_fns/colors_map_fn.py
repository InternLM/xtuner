# Copyright (c) OpenMMLab. All rights reserved.
def colors_map_fn(example):
    desc = ':'.join(example['description'].split(':')[1:]).strip()
    return {'conversation': [{'input': desc, 'output': example['color']}]}
