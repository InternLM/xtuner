# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.utils.misc import get_object_from_string


def template_map_fn_factory(template):
    if isinstance(template, str):  # for resume
        template = get_object_from_string(template)
    return template.template_map_fn_v1
