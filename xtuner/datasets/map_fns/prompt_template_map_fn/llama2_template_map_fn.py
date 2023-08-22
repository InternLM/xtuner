# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import PROMPT_TEMPLATE


def llama2_template_map_fn(example):

    PROMPT_START = PROMPT_TEMPLATE.internlm_chat.INSTRUCTION_START
    PROMPT = PROMPT_TEMPLATE.internlm_chat.INSTRUCTION

    conversation = example.get('conversation', [])
    for i, single_turn_conversation in enumerate(conversation):
        input = single_turn_conversation['input']
        if i == 0:
            single_turn_conversation['input'] = PROMPT_START.format(
                input=input)
        else:
            single_turn_conversation['input'] = PROMPT.format(input=input)

    return {'conversation': conversation}
