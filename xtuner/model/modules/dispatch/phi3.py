# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import transformers
from mmengine import MessageHub
from mmengine.utils import digit_version

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      post_process_for_sequence_parallel_attn,
                                      pre_process_for_sequence_parallel_attn)
from .attention import flash_attn_wo_mask, varlen_flash_attn

try:
    from transformers.cache_utils import Cache
except ImportError:

    class Cache:
        pass


TRANSFORMERS_VERSION = digit_version(transformers.__version__)
IS_LOW_VERSION_TRANSFORMERS = TRANSFORMERS_VERSION < digit_version('4.43')

if not IS_LOW_VERSION_TRANSFORMERS:
    from transformers.modeling_flash_attention_utils import \
        _flash_attention_forward

_flash_supports_window_size = False
try:
    from flash_attn import flash_attn_func

    _flash_supports_window_size = 'window_size' in list(
        inspect.signature(flash_attn_func).parameters)

    if not _flash_supports_window_size:
        raise ValueError(
            'Please update flash-attention to support window size.')
# else:
except ImportError:
    pass


# Copied from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/3a811845d89f3c1b3f41b341d0f9f05104769f35/modeling_phi3.py#L302  # noqa:E501
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/3a811845d89f3c1b3f41b341d0f9f05104769f35/modeling_phi3.py#L247  # noqa:E501
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/3a811845d89f3c1b3f41b341d0f9f05104769f35/modeling_phi3.py#L255  # noqa:E501
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """  # noqa:E501
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def phi3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if not _flash_supports_window_size:
        raise ValueError(
            'The current flash attention version does not support '
            'sliding window attention.')

    output_attentions = False

    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in '
            'v4.37. Please make sure use `attention_mask` instead.`')

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos:query_pos +
                     self.num_key_value_heads * self.head_dim]
    value_states = qkv[...,
                       query_pos + self.num_key_value_heads * self.head_dim:]

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                'The cache structure has changed since version v4.36. '
                f'If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, '
                'please make sure to initialize the attention class '
                'with a layer index.')
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                       self.layer_idx)

    rotary_seq_len = max(kv_seq_len, position_ids.max().item() + 1)
    cos, sin = self.rotary_emb(
        value_states, position_ids, seq_len=rotary_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, 'sliding_window', None) is not None
        and kv_seq_len > self.config.sliding_window)

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value
        # `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (getattr(self.config, 'sliding_window', None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    'past key must have a shape of (`batch_size, num_heads, '
                    'self.config.sliding_window-1, head_dim`), got'
                    f' {past_key.shape}')

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat(
                    [attention_mask,
                     torch.ones_like(attention_mask[:, -1:])],
                    dim=-1)

        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_dropout = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32.

    if query_states.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.qkv_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    enable_sequence_parallel = (
        dist.is_initialized() and get_sequence_parallel_world_size() > 1
        and self.training)
    if enable_sequence_parallel:
        # (b, s // sp_world_size, nd, dim) -> (b, s, nd // sp_world_size, dim)
        query_states, key_states, value_states = \
            pre_process_for_sequence_parallel_attn(
                query_states, key_states, value_states,
                scatter_dim=2, gather_dim=1)
        # num_heads has been changed because of sequence parallel
        # `self.num_heads`` is not used in self._flash_attention_forward
        # in mistral/mixtral, we are doing this to avoid some unnecessary risk
        ori_num_head = self.num_heads
        self.num_heads = query_states.shape[-2]

    if IS_LOW_VERSION_TRANSFORMERS:
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_states.shape[1],
            dropout=attn_dropout,
            use_sliding_windows=use_sliding_windows,
        )
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_states.shape[1],
            dropout=attn_dropout,
            sliding_window=getattr(self.config, 'sliding_window', None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

    if enable_sequence_parallel:
        # (b, s, nd // sp_world_size, dim) -> (b, s // sp_world_size, nd, dim)
        attn_output = post_process_for_sequence_parallel_attn(
            attn_output, scatter_dim=1, gather_dim=2)
        self.num_heads = ori_num_head

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def phi3_varlen_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if not _flash_supports_window_size:
        raise ValueError(
            'The current flash attention version does not support '
            'sliding window attention.')

    output_attentions = False

    is_training = self.training

    message_hub = MessageHub.get_instance('varlen_attn_args')
    rank = dist.get_rank()
    cumulative_len = message_hub.get_info(f'cumulative_len_rank_{rank}')
    max_seqlen = message_hub.get_info(f'max_seqlen_rank_{rank}')

    assert is_training == (past_key_value is None)
    use_varlen_atten = (cumulative_len is not None)

    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37'
            ' Please make sure use `attention_mask` instead.`')

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1, (f'If utilizing local attention, the batch size should be'
                      f' set to 1, but got {bsz}')
    # attention_mask is set to None if no padding token in input_ids
    # varlen attn need data packing so no padding tokens in input_ids
    assert attention_mask is None

    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos:query_pos +
                     self.num_key_value_heads * self.head_dim]
    value_states = qkv[...,
                       query_pos + self.num_key_value_heads * self.head_dim:]

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                'The cache structure has changed since version v4.36. '
                f'If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, '
                'please make sure to initialize the attention class '
                'with a layer index.')
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                       self.layer_idx)

    assert position_ids is not None
    rotary_seq_len = max(kv_seq_len, position_ids.max().item() + 1)
    cos, sin = self.rotary_emb(
        value_states, position_ids, seq_len=rotary_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, 'sliding_window', None) is not None
        and kv_seq_len > self.config.sliding_window)

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value
        # `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (getattr(self.config, 'sliding_window', None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    'past key must have a shape of (`batch_size, num_heads, '
                    'self.config.sliding_window-1, head_dim`), got'
                    f' {past_key.shape}')

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat(
                    [attention_mask,
                     torch.ones_like(attention_mask[:, -1:])],
                    dim=-1)

        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # In PEFT, usually we cast the layer norms in float32 for
    # training stability reasons, therefore the input hidden states gets
    # silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.

    if query_states.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.qkv_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # ----------------- flash attention forward ------------------------#

    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and q_len != 1

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, 'sliding_window', None) is not None
        and kv_seq_len > self.config.sliding_window)

    window_size = (self.config.sliding_window,
                   self.config.sliding_window) if use_sliding_windows else (-1,
                                                                            -1)
    attn_dropout = self.attention_dropout if self.training else 0.0

    if use_varlen_atten:
        attn_output = varlen_flash_attn(
            query_states,
            key_states,
            value_states,
            cumulative_len,
            max_seqlen,
            causal=causal,
            dropout_p=attn_dropout,
            window_size=window_size,
            training=self.training)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=causal,
            dropout_p=attn_dropout,
            window_size=window_size,
            training=self.training)

    # ---------------- flash attention forward end ------------------- #

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
