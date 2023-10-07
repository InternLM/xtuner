# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial


def template_map_fn(example, template):
    conversation = example.get('conversation', [])
    for i, single_turn_conversation in enumerate(conversation):
        input = single_turn_conversation['input']
        input = template.INSTRUCTION.format(input=input, round=i + 1)
        system = single_turn_conversation.get('system', '')
        if system != '':
            system = template.SYSTEM.format(system=system)
            input = system + input
        single_turn_conversation['input'] = input
    return {'conversation': conversation}


def template_map_fn_factory(template):
    return partial(template_map_fn, template=template)
