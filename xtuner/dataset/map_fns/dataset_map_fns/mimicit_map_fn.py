# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import DEFAULT_IMAGE_TOKEN


def mimicit_map_fn(example):
    messages = example['messages']
    n_images = len(example['image_ids'])
    image_concatenated = False

    system = ''
    input = ''
    conversation = []
    while messages and messages[0]['role'] == 'assistant':
        # Skip the first one if it is from assistant
        messages = messages[1:]

    for msg in messages:
        if msg['role'] == 'system':
            system = msg['content']
        elif msg['role'] == 'user':
            if not image_concatenated:
                content = '\n'.join([DEFAULT_IMAGE_TOKEN] * n_images +
                                    [msg['content']])
                image_concatenated = True
            else:
                content = msg['content']
            input += content
        elif msg['role'] == 'assistant':
            conversation.append({
                'system': system,
                'input': input,
                'output': msg['content']
            })
            system = ''
            input = ''
        else:
            raise NotImplementedError

    return {'conversation': conversation}
