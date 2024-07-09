# Copyright (c) OpenMMLab. All rights reserved.
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
    InternLM2ForCausalLM=digit_version('4.36'),
    InternLMForCausalLM=digit_version('4.36'),
    LlamaForCausalLM=digit_version('4.36'),
    Phi3ForCausalLM=digit_version('4.39'),
    MistralForCausalLM=digit_version('4.36'),
    # Training mixtral with lower version may lead to nccl timeout
    # Refer to https://github.com/microsoft/DeepSpeed/issues/5066
    MixtralForCausalLM=digit_version('4.40'),
    CohereForCausalLM=digit_version('4.40'),
    Qwen2ForCausalLM=digit_version('4.39'),
    Qwen2MoeForCausalLM=digit_version('4.40'),
    DeepseekV2ForCausalLM=digit_version('4.40'),
)

ATTN_DISPATCH_MAPPING = dict(
    InternLM2FlashAttention2=LazyObject(
        'xtuner.model.modules.dispatch.internlm2', 'internlm2_attn_forward'),
    InternLMAttention=LazyObject('xtuner.model.modules.dispatch.internlm',
                                 'internlm_attn_forward'),
    LlamaFlashAttention2=LazyObject('xtuner.model.modules.dispatch.llama',
                                    'llama_attn_forward'),
    Phi3FlashAttention2=LazyObject('xtuner.model.modules.dispatch.phi3',
                                   'phi3_attn_forward'),
    MistralFlashAttention2=LazyObject('xtuner.model.modules.dispatch.mistral',
                                      'mistral_attn_forward'),
    MixtralFlashAttention2=LazyObject('xtuner.model.modules.dispatch.mistral',
                                      'mistral_attn_forward'),
    CohereFlashAttention2=LazyObject('xtuner.model.modules.dispatch.cohere',
                                     'cohere_attn_forward'),
    Qwen2FlashAttention2=LazyObject('xtuner.model.modules.dispatch.qwen2',
                                    'qwen2_attn_forward'),
    Qwen2MoeFlashAttention2=LazyObject('xtuner.model.modules.dispatch.qwen2',
                                       'qwen2_attn_forward'),
    DeepseekV2FlashAttention2=LazyObject(
        'xtuner.model.modules.dispatch.deepseek_v2', 'deepseek_attn_forward'),
)

ATTN_LEGACY_DISPATCH_MAPPING = dict(
    LlamaFlashAttention2=LazyObject('xtuner.model.modules.dispatch.llama',
                                    'llama_attn_forward_legacy'), )

VARLEN_ATTN_DISPATCH_MAPPING = dict(
    InternLM2FlashAttention2=LazyObject(
        'xtuner.model.modules.dispatch.internlm2',
        'internlm2_varlen_attn_forward'),
    InternLMAttention=LazyObject('xtuner.model.modules.dispatch.internlm',
                                 'internlm_varlen_attn_forward'),
    LlamaFlashAttention2=LazyObject('xtuner.model.modules.dispatch.llama',
                                    'llama_varlen_attn_forward'),
    Phi3FlashAttention2=LazyObject('xtuner.model.modules.dispatch.phi3',
                                   'phi3_varlen_attn_forward'),
    MistralFlashAttention2=LazyObject('xtuner.model.modules.dispatch.mistral',
                                      'mistral_varlen_attn_forward'),
    MixtralFlashAttention2=LazyObject('xtuner.model.modules.dispatch.mistral',
                                      'mistral_varlen_attn_forward'),
    CohereFlashAttention2=None,
    Qwen2FlashAttention2=LazyObject('xtuner.model.modules.dispatch.qwen2',
                                    'qwen2_varlen_attn_forward'),
    Qwen2MoeFlashAttention2=LazyObject('xtuner.model.modules.dispatch.qwen2',
                                       'qwen2_varlen_attn_forward'),
    DeepseekV2FlashAttention2=LazyObject(
        'xtuner.model.modules.dispatch.deepseek_v2',
        'deepseek_varlen_attn_forward'),
)

VARLEN_ATTN_LEGACY_DISPATCH_MAPPING = dict(
    LlamaFlashAttention2=LazyObject('xtuner.model.modules.dispatch.llama',
                                    'llama_varlen_attn_forward_legacy'), )

RMS_DISPATCH_MAPPING = dict(
    InternLM2RMSNorm=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                                'rms_norm_forward'),
    InternLMRMSNorm=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                               'rms_norm_forward'),
    LlamaRMSNorm=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                            'rms_norm_forward'),
    Phi3RMSNorm=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                           'rms_norm_forward'),
    MistralRMSNorm=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                              'rms_norm_forward'),
    MixtralRMSNorm=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                              'rms_norm_forward'),
    CohereLayerNorm=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                               'layer_norm_forward'),
    Qwen2RMSNorm=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                            'rms_norm_forward'),
    Qwen2MoeRMSNorm=LazyObject('xtuner.model.modules.dispatch.triton_kernels',
                               'rms_norm_forward'),
)

ROTE_DISPATCH_MAPPING = dict(
    InternLMRotaryEmbedding=LazyObject(
        'xtuner.model.modules.dispatch.internlm', 'InternLMRotaryEmbedding'),
    MistralRotaryEmbedding=LazyObject('xtuner.model.modules.dispatch.mistral',
                                      'MistralRotaryEmbedding'),
    MixtralRotaryEmbedding=LazyObject('xtuner.model.modules.dispatch.mistral',
                                      'MistralRotaryEmbedding'),
)


def log_once(func):
    logged = False

    def wrapper(*args, **kwargs):
        nonlocal logged
        if not logged:
            logged = True
            func(*args, **kwargs)
        return

    return wrapper


def dispatch_attn_forward(model):

    if not SUPPORT_FLASH2:
        return

    from mmengine import print_log
    print_log = log_once(print_log)

    attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if (IS_LOW_VERSION_TRANSFORMERS
                and name in ATTN_LEGACY_DISPATCH_MAPPING):
            if attn_forward is None:
                attn_forward = ATTN_LEGACY_DISPATCH_MAPPING[name]
                attn_forward = attn_forward.build()
            print_log(f'Dispatch {name} legacy forward. {NO_ATTN_WEIGHTS_MSG}',
                      'current')
            module.forward = types.MethodType(attn_forward, module)
        elif name in ATTN_DISPATCH_MAPPING:
            if attn_forward is None:
                attn_forward = ATTN_DISPATCH_MAPPING[name]
                attn_forward = attn_forward.build()
            print_log(f'Dispatch {name} forward. {NO_ATTN_WEIGHTS_MSG}',
                      'current')
            module.forward = types.MethodType(attn_forward, module)


def dispatch_varlen_attn_forward(model):

    if not SUPPORT_FLASH2:
        return

    from mmengine import print_log
    print_log = log_once(print_log)

    varlen_attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if (IS_LOW_VERSION_TRANSFORMERS
                and name in VARLEN_ATTN_LEGACY_DISPATCH_MAPPING):
            if varlen_attn_forward is None:
                varlen_attn_forward = VARLEN_ATTN_LEGACY_DISPATCH_MAPPING[name]
                varlen_attn_forward = varlen_attn_forward.build()
            print_log(
                f'Dispatch legacy {name} varlen forward. '
                f'{NO_ATTN_WEIGHTS_MSG}', 'current')
            module.forward = types.MethodType(varlen_attn_forward, module)
        elif name in VARLEN_ATTN_DISPATCH_MAPPING:
            if varlen_attn_forward is None:
                varlen_attn_forward = VARLEN_ATTN_DISPATCH_MAPPING[name]
                varlen_attn_forward = varlen_attn_forward.build()
            print_log(f'Dispatch {name} varlen forward. {NO_ATTN_WEIGHTS_MSG}',
                      'current')
            module.forward = types.MethodType(varlen_attn_forward, module)


def dispatch_rmsnorm_forward(model):

    if (not SUPPORT_TRITON) or (not USE_TRITON_KERNEL):
        return

    from mmengine import print_log
    print_log = log_once(print_log)

    rms_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in RMS_DISPATCH_MAPPING:
            if rms_forward is None:
                rms_forward = RMS_DISPATCH_MAPPING[name]
                rms_forward = rms_forward.build()
            print_log(f'Dispatch {name} forward.', 'current')
            module.forward = types.MethodType(rms_forward, module)


def replace_rote(model):

    from mmengine import print_log
    print_log = log_once(print_log)

    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if cls_name in ROTE_DISPATCH_MAPPING:
                assert hasattr(model.config, 'rope_theta'), \
                    '`rope_theta` should be in the model config.'
                rope_theta = model.config.rope_theta

                rote = ROTE_DISPATCH_MAPPING[cls_name]
                rote = rote.build()
                print_log(f'replace {cls_name}', 'current')
                dim_model = child.inv_freq.shape[0] * 2
                child_new = rote(dim_model, child.max_seq_len_cached,
                                 rope_theta).to(
                                     device=child.inv_freq.device,
                                     dtype=child.inv_freq.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


def dispatch_modules(model, use_varlen_attn=False):

    def check(model_name):
        if 'ForCausalLM' not in model_name and model_name.endswith('Model'):
            # a walkaround for reward model
            model_name = model_name[:-5] + 'ForCausalLM'
        msg = '{} requires transformers version at least {}, but got {}'
        if model_name in LOWEST_TRANSFORMERS_VERSION:
            assert TRANSFORMERS_VERSION >= LOWEST_TRANSFORMERS_VERSION[
                model_name], msg.format(
                    model_name, LOWEST_TRANSFORMERS_VERSION[model_name],
                    TRANSFORMERS_VERSION)

    check(type(model).__name__)
    if use_varlen_attn:
        dispatch_varlen_attn_forward(model)
    else:
        dispatch_attn_forward(model)
    dispatch_rmsnorm_forward(model)
    replace_rote(model)


__all__ = ['dispatch_modules']
