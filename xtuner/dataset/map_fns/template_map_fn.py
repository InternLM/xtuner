# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

from mmengine.utils.misc import get_object_from_string


def template_map_fn(example, template):
    conversation_original = example.get('conversation', [])
    conversation = template.build_conversation(conversation_original)
    return {'conversation': conversation}


def template_map_fn_factory(template):
    if isinstance(template, str):  # for resume
        template = get_object_from_string(template)
    return partial(template_map_fn, template=template)
