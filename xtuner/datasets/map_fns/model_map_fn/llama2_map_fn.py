# Copyright (c) OpenMMLab. All rights reserved.
def llama2_map_fn(example):
    B_INST, E_INST = '[INST]', '[/INST]'
    B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'

    DEFAULT_SYSTEM_PROMPT = \
        'You are a helpful, respectful and honest assistant. Always answer ' \
        'as helpfully as possible, while being safe. Your answers should ' \
        'not include any harmful, unethical, racist, sexist, toxic, ' \
        'dangerous, or illegal content. Please ensure that your responses ' \
        'are socially unbiased and positive in nature.'

    conversation = example.get('conversation', [])
    for single_turn_conversation in conversation:
        input = single_turn_conversation['input']
        single_turn_conversation['input'] = f'{B_INST} {B_SYS} ' \
            f'{DEFAULT_SYSTEM_PROMPT} {E_SYS}{input} {E_INST}'

    return {'conversation': conversation}
