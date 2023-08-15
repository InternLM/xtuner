# Copyright (c) OpenMMLab. All rights reserved.
def internlm_map_fn(example):
    user = '<|User|>'
    eoh = '<eoh>'
    assistant = '<|Bot|>'
    conversation = example.get('conversation', [])
    for single_turn_conversation in conversation:
        input = single_turn_conversation['input']
        single_turn_conversation['input'] = \
            f'{user}:{input}{eoh}\n{assistant}:'
    return {'conversation': conversation}
