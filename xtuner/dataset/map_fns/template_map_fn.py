# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

from mmengine.utils.misc import get_object_from_string


def template_map_fn(example, template):
    conversation = example.get('conversation', [])
    for i, single_turn_conversation in enumerate(conversation):
        input = single_turn_conversation.get('input', '')
        if input is None:
            input = ''
        input_text = template.INSTRUCTION.format(input=input, round=i + 1)
        system = single_turn_conversation.get('system', '')
        if system != '' and system is not None:
            system = template.SYSTEM.format(system=system)
            input_text = system + input_text
        single_turn_conversation['input'] = input_text
    return {'conversation': conversation}


def template_map_fn_factory(template):
    if isinstance(template, str):  # for resume
        template = get_object_from_string(template)
    return partial(template_map_fn, template=template)
