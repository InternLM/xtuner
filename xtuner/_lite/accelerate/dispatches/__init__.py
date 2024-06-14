# Copyright (c) OpenMMLab. All rights reserved.
import types

from xtuner._lite import get_logger
from ._fused import rms_norm_forward

logger = get_logger()


def dispatch_internlm2_attn_forward(model, use_varlen_attn):

    from .internlm2 import (internlm2_attn_forward,
                            internlm2_varlen_attn_forward)

    for module in model.modules():
        # Do not need to dispatch if
        # type(module).__name__ == 'InternLM2Attention'.
        # If we set `attn_implementation` to `eager` in xtuner
        # configs, we can not use varlen attn and sequence parallel.
        if type(module).__name__ == 'InternLM2FlashAttention2':
            if use_varlen_attn:

                module.forward = types.MethodType(
                    internlm2_varlen_attn_forward, module)
            else:

                module.forward = types.MethodType(internlm2_attn_forward,
                                                  module)


def dispatch_internlm2_rmsnorm_forward(model):

    for module in model.modules():
        if type(module).__name__ == 'InternLM2RMSNorm':
            logger.info('dispatch internlm2 rmsnorm forward', 'current')
            module.forward = types.MethodType(rms_norm_forward, module)


def dispatch_internlm2_model_forward(model, use_varlen_attn):

    from .internlm2 import internlm2_model_forward

    for module in model.modules():
        # Do not need to dispatch if
        # type(module).__name__ in ('LlamaAttention', 'LlamaSdpaAttention').
        # If we set `attn_implementation` to `sdpa` or `eager` in xtuner
        # configs, we can not use varlen attn and sequence parallel.
        if type(module).__name__ == 'InternLM2Model':

            logger.info('dispatch internlm2 model forward')

            module.forward = types.MethodType(internlm2_model_forward, module)


def replace_internlm2_rote(model):
    from .internlm2 import InternLM2RotaryEmbedding

    rotary_base = model.config.rope_theta

    def traverse(module):
        for name, child in module.named_children():
            if type(child).__name__ in (
                    'InternLM2RotaryEmbedding',
                    'InternLM2LinearScalingRotaryEmbedding',
                    'InternLM2DynamicNTKScalingRotaryEmbedding'):
                logger.info('replace internlm2 rope')
                dim_model = child.inv_freq.shape[0] * 2
                child_new = InternLM2RotaryEmbedding(
                    dim_model, child.max_position_embeddings, rotary_base).to(
                        device=child.inv_freq.device,
                        dtype=child.inv_freq.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


def dispatch_modules(model, use_varlen_attn=False):

    model_name = model.__class__.__name__.lower()
    if 'internlm2' in model_name:
        dispatch_internlm2_attn_forward(model, use_varlen_attn)
        # dispatch_internlm2_model_forward(model, use_varlen_attn)

        dispatch_internlm2_rmsnorm_forward(model)
        replace_internlm2_rote(model)
