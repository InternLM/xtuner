# Copyright (c) OpenMMLab. All rights reserved.
def multi_turn_1_map_fn(example):
    messages = example['messages']
    conversation = []
    for msg in messages:
        conversation.append({
            'system': msg['toy_system'],
            'input': msg['toy_input'],
            'output': msg['toy_output']
        })
    return {'conversation': conversation}
