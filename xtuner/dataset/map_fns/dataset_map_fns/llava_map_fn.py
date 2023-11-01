# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import DEFAULT_IMAGE_TOKEN


def llava_map_fn(example):
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({
                'system': '',
                'input': input,
                'output': msg['value']
            })
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}
