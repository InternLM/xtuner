# Copyright (c) OpenMMLab. All rights reserved.
COLORIST_SYSTEM = ('You are a professional color designer. Please provide the '
                   'corresponding colors based on the description of Human.\n')


def colors_map_fn(example):
    desc = ':'.join(example['description'].split(':')[1:]).strip()
    return {'conversation': [{'input': desc, 'output': example['color']}]}
