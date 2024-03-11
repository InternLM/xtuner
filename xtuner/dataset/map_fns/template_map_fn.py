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

        if template.get('SUFFIX', None):
            output_text = single_turn_conversation.get('output', '')
            output_text += template.SUFFIX
            single_turn_conversation['output'] = output_text

        # SUFFIX_AS_EOS is False ==> need_eos_token is True
        single_turn_conversation['need_eos_token'] = \
            not template.get('SUFFIX_AS_EOS', False)
        single_turn_conversation['sep'] = template.get('SEP', '')

    return {'conversation': conversation}


def template_map_fn_factory(template):
    if isinstance(template, str):  # for resume
        template = get_object_from_string(template)
    return partial(template_map_fn, template=template)
