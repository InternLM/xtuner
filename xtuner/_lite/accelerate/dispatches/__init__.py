# Copyright (c) OpenMMLab. All rights reserved.
import types

from xtuner._lite import get_logger

logger = get_logger()


def _dispatch_forward_fn(module, dispatch_fn):
    module.forward = types.MethodType(dispatch_fn, module)


def dispatch_internlm_varlen_attn_forward(module):
    assert module.__class__.__name__ == 'InternLM2FlashAttention2'
    from .internlm2 import internlm2_varlen_attn_forward
    _dispatch_forward_fn(module, internlm2_varlen_attn_forward)
    return internlm2_varlen_attn_forward.__name__


def dispatch_clip_attn_forward(module):
    assert module.__class__.__name__ == 'CLIPAttention'
    from .clip import clip_flash_attn_forward
    _dispatch_forward_fn(module, clip_flash_attn_forward)
    return clip_flash_attn_forward.__name__


DISPATCH_MAP = {
    'InternLM2FlashAttention2': dispatch_internlm_varlen_attn_forward,
    'CLIPAttention': dispatch_clip_attn_forward
}


def dispatch_modules(model, use_varlen_attn=False):
    from xtuner._lite import get_logger
    logger = get_logger()

    for name, module in model.named_modules():
        module_cls = module.__class__.__name__
        if module_cls in DISPATCH_MAP:
            dispatched = DISPATCH_MAP[module_cls](module)
            logger.info(
                f'Dispatch {name}({module_cls}) forward to `{dispatched}`')
