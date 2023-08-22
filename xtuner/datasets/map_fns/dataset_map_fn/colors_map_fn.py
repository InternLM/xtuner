# Copyright (c) OpenMMLab. All rights reserved.


def colors_dataset_map_fn(example):
    PROMPT = ('You are a professional color designer. Please provide the '
              'corresponding colors based on the description of Human.\n'
              '### Human: {input}\n### Bot: ')
    desc = ':'.join(example['description'].split(':')[1:]).strip()
    return {
        'conversation': [{
            'input': PROMPT.format(input=desc),
            'output': example['color']
        }]
    }
