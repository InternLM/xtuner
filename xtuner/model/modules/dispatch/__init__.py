# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import types

import torch
import transformers
from mmengine.config.lazy import LazyObject
from mmengine.utils import digit_version
from transformers.utils.import_utils import is_flash_attn_2_available

TRANSFORMERS_VERSION = digit_version(transformers.__version__)
IS_LOW_VERSION_TRANSFORMERS = TRANSFORMERS_VERSION < digit_version('4.38')
# Transformers requires torch version >= 2.1.1 when using Torch SDPA.
# Refer to https://github.com/huggingface/transformers/blob/caa5c65db1f4db617cdac2ad667ba62edf94dd98/src/transformers/modeling_utils.py#L1611  # noqa: E501
SUPPORT_FLASH1 = digit_version(torch.__version__) >= digit_version('2.1.1')
SUPPORT_FLASH2 = is_flash_attn_2_available()
SUPPORT_FLASH = SUPPORT_FLASH1 or SUPPORT_FLASH2

USE_TRITON_KERNEL = bool(os.getenv('USE_TRITON_KERNEL', default=0))
SUPPORT_TRITON = False
try:
    import triton  # pre-check # noqa: F401
    import triton.language as tl  # pre-check # noqa: F401
    SUPPORT_TRITON = True
except ImportError:
    if USE_TRITON_KERNEL:
        raise RuntimeError(
            'USE_TRITON_KERNEL is set to 1, but triton has not been installed.'
            ' Run `pip install triton==2.1.0` to install triton.')

NO_ATTN_WEIGHTS_MSG = (
    'Due to the implementation of the PyTorch version of flash attention, '
    'even when the `output_attentions` flag is set to True, it is not '
    'possible to return the `attn_weights`.')

LOWEST_TRANSFORMERS_VERSION = dict(
    internlm2=digit_version('4.36'),
    internlm=digit_version('4.36'),
    llama=digit_version('4.36'),
    phi3=digit_version('4.39'),
    yi=digit_version('4.36'),
    mistral=digit_version('4.36'),
    # Training mixtral with lower version may lead to nccl timeout
    # Refer to https://github.com/microsoft/DeepSpeed/issues/5066
    mixtral=digit_version('4.40'),
    cohere=digit_version('4.40'),
    qwen2=digit_version('4.39'),
    qwen2_moe=digit_version('4.40'),
)

DISPATCH_MAPPING = dict(
    internlm2=dict(
        attn_module_name='InternLM2FlashAttention2',
        attn=LazyObject('xtuner.model.modules.dispatch.internlm2',
                        'internlm2_attn_forward'),
        varlen_attn=LazyObject('xtuner.model.modules.dispatch.internlm2',
                               'internlm2_varlen_attn_forward'),
        rms_module_name='InternLM2RMSNorm',
        rms=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                       'rms_norm_forward'),
        rote_module_name='InternLM2RotaryEmbedding',
        rote=LazyObject('xtuner.model.modules.dispatch.internlm2',
                        'InternLM2RotaryEmbedding'),
    ),
    internlm=dict(
        attn_module_name='InternLMAttention',
        attn=LazyObject('xtuner.model.modules.dispatch.internlm',
                        'internlm_attn_forward'),
        varlen_attn=LazyObject('xtuner.model.modules.dispatch.internlm',
                               'internlm_varlen_attn_forward'),
        rms_module_name='InternLMRMSNorm',
        rms=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                       'rms_norm_forward'),
        rote_module_name='InternLMRotaryEmbedding',
        rote=LazyObject('xtuner.model.modules.dispatch.internlm',
                        'InternLMRotaryEmbedding'),
    ),
    llama=dict(
        attn_module_name='LlamaFlashAttention2',
        attn=LazyObject('xtuner.model.modules.dispatch.llama',
                        'llama_attn_forward'),
        attn_legacy=LazyObject('xtuner.model.modules.dispatch.llama',
                               'llama_attn_forward_legacy'),
        varlen_attn=LazyObject('xtuner.model.modules.dispatch.llama',
                               'llama_varlen_attn_forward'),
        varlen_attn_legacy=LazyObject('xtuner.model.modules.dispatch.llama',
                                      'llama_varlen_attn_forward_legacy'),
        rms_module_name='LlamaRMSNorm',
        rms=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                       'rms_norm_forward'),
    ),
    phi3=dict(
        attn_module_name='Phi3FlashAttention2',
        attn=LazyObject('xtuner.model.modules.dispatch.phi3',
                        'phi3_attn_forward'),
        varlen_attn=LazyObject('xtuner.model.modules.dispatch.phi3',
                               'phi3_varlen_attn_forward'),
        rms_module_name='Phi3RMSNorm',
        rms=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                       'rms_norm_forward'),
    ),
    mistral=dict(
        attn_module_name='MistralFlashAttention2',
        attn=LazyObject('xtuner.model.modules.dispatch.mistral',
                        'mistral_attn_forward'),
        varlen_attn=LazyObject('xtuner.model.modules.dispatch.mistral',
                               'mistral_varlen_attn_forward'),
        rms_module_name='MistralRMSNorm',
        rms=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                       'rms_norm_forward'),
        rote_module_name='MistralRotaryEmbedding',
        rote=LazyObject('xtuner.model.modules.dispatch.mistral',
                        'MistralRotaryEmbedding'),
    ),
    mixtral=dict(
        attn_module_name='MixtralFlashAttention2',
        attn=LazyObject('xtuner.model.modules.dispatch.mistral',
                        'mistral_attn_forward'),
        varlen_attn=LazyObject('xtuner.model.modules.dispatch.mistral',
                               'mistral_varlen_attn_forward'),
        rms_module_name='MixtralRMSNorm',
        rms=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                       'rms_norm_forward'),
        rote_module_name='MixtralRotaryEmbedding',
        rote=LazyObject('xtuner.model.modules.dispatch.mistral',
                        'MistralRotaryEmbedding'),
    ),
    cohere=dict(
        attn_module_name='CohereFlashAttention2',
        attn=LazyObject('xtuner.model.modules.dispatch.cohere',
                        'cohere_attn_forward'),
        rms_module_name='CohereLayerNorm',
        rms=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                       'layer_norm_forward'),
    ),
    qwen2=dict(
        attn_module_name='Qwen2FlashAttention2',
        attn=LazyObject('xtuner.model.modules.dispatch.qwen2',
                        'qwen2_attn_forward'),
        varlen_attn=LazyObject('xtuner.model.modules.dispatch.qwen2',
                               'qwen2_varlen_attn_forward'),
        rms_module_name='Qwen2RMSNorm',
        rms=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                       'rms_norm_forward'),
    ),
    qwen2moe=dict(
        attn_module_name='Qwen2MoeFlashAttention2',
        attn=LazyObject('xtuner.model.modules.dispatch.qwen2',
                        'qwen2_attn_forward'),
        varlen_attn=LazyObject('xtuner.model.modules.dispatch.qwen2',
                               'qwen2_varlen_attn_forward'),
        rms_module_name='Qwen2MoeRMSNorm',
        rms=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                       'rms_norm_forward'),
    ),
)

# Sorting is necessary. We aim for the Moe model to precede
# the corresponding Dense model in ranking as we traverse the dictionary.
DISPATCH_MAPPING = {
    key: DISPATCH_MAPPING[key]
    for key in sorted(DISPATCH_MAPPING.keys(), reverse=True)
}


def log_once(func):
    logged = False

    def wrapper(*args, **kwargs):
        nonlocal logged
        if not logged:
            logged = True
            func(*args, **kwargs)
        return

    return wrapper


def dispatch_attn_forward(model, mapping):

    if not SUPPORT_FLASH2:
        return

    attn_forward = mapping.get('attn', None)
    if attn_forward is None:
        return

    from mmengine import print_log

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    attn_name = mapping['attn_module_name']
    attn_forward = attn_forward.build()
    attn_forward_legacy = mapping.get('attn_legacy', None)
    if attn_forward_legacy:
        attn_forward_legacy = attn_forward_legacy.build()

    print_log = log_once(print_log)

    for module in model.modules():
        if type(module).__name__ == attn_name:
            if attn_forward_legacy and IS_LOW_VERSION_TRANSFORMERS:
                print_log(f'dispatch legacy {attn_name} forward', 'current')
                module.forward = types.MethodType(attn_forward_legacy, module)
            else:
                print_log(f'dispatch {attn_name} forward', 'current')
                module.forward = types.MethodType(attn_forward, module)


def dispatch_varlen_attn_forward(model, mapping):

    assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
        'flash_attn and triton is required if you want to use varlen_attn.'

    varlen_attn_forward = mapping.get('varlen_attn', None)
    if varlen_attn_forward is None:
        return

    from mmengine import print_log

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    attn_name = mapping['attn_module_name']
    varlen_attn_forward = varlen_attn_forward.build()
    varlen_attn_forward_legacy = mapping.get('varlen_attn_legacy', None)
    if varlen_attn_forward_legacy:
        varlen_attn_forward_legacy = varlen_attn_forward_legacy.build()

    print_log = log_once(print_log)

    for module in model.modules():
        if type(module).__name__ == attn_name:
            if varlen_attn_forward_legacy and IS_LOW_VERSION_TRANSFORMERS:
                print_log(f'dispatch legacy {attn_name} varlen forward',
                          'current')
                module.forward = types.MethodType(varlen_attn_forward_legacy,
                                                  module)
            else:
                print_log(f'dispatch {attn_name} varlen forward', 'current')
                module.forward = types.MethodType(varlen_attn_forward, module)


def dispatch_rmsnorm_forward(model, mapping):

    if (not SUPPORT_TRITON) or (not USE_TRITON_KERNEL):
        return

    rms_forward = mapping.get('rms', None)
    if rms_forward is None:
        return

    from mmengine import print_log
    print_log = log_once(print_log)

    rms_module_name = mapping['rms_module_name']
    rms_forward = rms_forward.build()

    for module in model.modules():
        if type(module).__name__ == rms_module_name:
            print_log(f'dispatch {rms_module_name} forward', 'current')
            module.forward = types.MethodType(rms_forward, module)


def replace_rote(model, mapping):

    rote = mapping.get('rote', None)
    if rote is None:
        return

    from mmengine import print_log
    print_log = log_once(print_log)

    rote_module_name = mapping['rote_module_name']
    rote = rote.build()

    def traverse(module):
        for name, child in module.named_children():
            if type(child).__name__ == rote_module_name:
                print_log(f'replace {rote_module_name}', 'current')
                dim_model = child.inv_freq.shape[0] * 2
                child_new = rote(dim_model, child.max_seq_len_cached).to(
                    device=child.inv_freq.device, dtype=child.inv_freq.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


def dispatch_modules(model, use_varlen_attn=False):

    def check(model_name):
        msg = '{} requires transformers version at least {}, but got {}'
        assert TRANSFORMERS_VERSION >= LOWEST_TRANSFORMERS_VERSION[
            model_name], msg.format(model_name,
                                    LOWEST_TRANSFORMERS_VERSION[model_name],
                                    TRANSFORMERS_VERSION)

    model_name = model.__class__.__name__.lower()
    dispatch_mapping = None
    for key, mapping in DISPATCH_MAPPING.items():
        if key in model_name:
            check(key)
            dispatch_mapping = mapping
            break

    assert dispatch_mapping

    if use_varlen_attn:
        dispatch_varlen_attn_forward(model, dispatch_mapping)
    else:
        dispatch_attn_forward(model, dispatch_mapping)
    dispatch_rmsnorm_forward(model, dispatch_mapping)
    replace_rote(model, dispatch_mapping)


__all__ = ['dispatch_modules']
