# Copyright (c) OpenMMLab. All rights reserved.
def multi_turn_2_map_fn(example):
    messages = example['messages']
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
            input += msg['content']
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
