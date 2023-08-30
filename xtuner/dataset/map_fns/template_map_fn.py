# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial


def template_map_fn(example, template):
    conversation = example.get('conversation', [])
    for i, single_turn_conversation in enumerate(conversation):
        input = single_turn_conversation['input']
        if i == 0:
            single_turn_conversation[
                'input'] = template.INSTRUCTION_START.format(
                    input=input, round=i + 1)
        else:
            single_turn_conversation['input'] = template.INSTRUCTION.format(
                input=input, round=i + 1)
    return {'conversation': conversation}


def template_map_fn_factory(template):
    return partial(template_map_fn, template=template)
