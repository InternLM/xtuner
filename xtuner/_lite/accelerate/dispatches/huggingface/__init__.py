# Copyright (c) OpenMMLab. All rights reserved.
import types

from xtuner._lite import get_logger

logger = get_logger()


def _dispatch_forward_fn(module, dispatch_fn):
    module.forward = types.MethodType(dispatch_fn, module)


def _dispatch_internlm_varlen_attn_forward(module):
    assert module.__class__.__name__ == 'InternLM2FlashAttention2'
    from .internlm2 import internlm2_varlen_attn_forward
    _dispatch_forward_fn(module, internlm2_varlen_attn_forward)
    return internlm2_varlen_attn_forward.__name__


def _dispatch_internlm_reward_forward(module):
    assert module.__class__.__name__ == 'InternLM2ForRewardModel'
    from .internlm2 import internlm2_reward_forward
    _dispatch_forward_fn(module, internlm2_reward_forward)
    return internlm2_reward_forward.__name__


def _dispatch_clip_attn_forward(module):
    assert module.__class__.__name__ == 'CLIPAttention'
    from .clip import clip_flash_attn_forward
    _dispatch_forward_fn(module, clip_flash_attn_forward)
    return clip_flash_attn_forward.__name__


def _dispatch_rms_norm_forward(module):
    from .._fused import rms_norm_forward
    _dispatch_forward_fn(module, rms_norm_forward)
    return rms_norm_forward.__name__


DISPATCH_MAP = {
    'InternLM2FlashAttention2': _dispatch_internlm_varlen_attn_forward,
    'CLIPAttention': _dispatch_clip_attn_forward,
    'InternLM2ForRewardModel': _dispatch_internlm_reward_forward,
    'InternLM2RMSNorm': _dispatch_rms_norm_forward
}


def dispatch_hf_code(model):
    from xtuner._lite import get_logger
    logger = get_logger()

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in DISPATCH_MAP:
            dispatched = DISPATCH_MAP[cls_name](module)
            # breakpoint()
            logger.debug(
                f'Dispatch {name}({cls_name}) forward to `{dispatched}`')
