# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import MessageHub
from transformers.models.llama.modeling_llama import (apply_rotary_pos_emb,
                                                      repeat_kv)

from .triton_kernels import apply_rotary_emb

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    SUPPORT_FLASH2 = True
except ImportError:
    pass

try:
    from transformers.cache_utils import Cache
except ImportError:

    class Cache:
        pass


def llama_attn_forward(
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
    # Modified from https://github.com/huggingface/transformers/blob/66ce9593fdb8e340df546ddd0774eb444f17a12c/src/transformers/models/llama/modeling_llama.py#L422  # noqa:E501
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    past_key_value = getattr(self, 'past_key_value', past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models;
        # cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    if SUPPORT_FLASH2:

        # In PEFT, usually we cast the layer norms in float32 for training
        # stability reasons therefore the input hidden states gets silently casted
        # in float32. Hence, we need cast them back in the correct dtype just to
        # be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast
        # the LayerNorms in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate)
    else:
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # use flash attention implemented by pytorch
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len,
                                      self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attn_forward_legacy(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    # Modified from https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/models/llama/modeling_llama.py#L331  # noqa:E501
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in '
            'v4.37. Please make sure use `attention_mask` instead.`')

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

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
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    if SUPPORT_FLASH2:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = flash_attn_func(
            query_states, key_states, value_states, causal=True)
        attn_output = attn_output.contiguous()
    else:
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # use flash attention implemented by pytorch
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


def llama_varlen_attn_forward(
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
    is_training = self.training

    message_hub = MessageHub.get_instance('varlen_attn_args')
    rank = dist.get_rank()
    cumulative_len = message_hub.get_info(f'cumulative_len_rank_{rank}')
    indexes = message_hub.get_info(f'indexes_rank_{rank}')
    max_seqlen = message_hub.get_info(f'max_seqlen_rank_{rank}')
    assert is_training == (cumulative_len is not None)

    if 'padding_mask' in kwargs:
        warnings.warn('Passing `padding_mask` is deprecated and will be '
                      'removed in v4.37. Please make sure use '
                      '`attention_mask` instead.`')
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states,
                               indexes if is_training else position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    past_key_value = getattr(self, 'past_key_value', past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models;
        # cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently casted
    # in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    assert SUPPORT_FLASH2
    if is_training:
        q_unpad, k_unpad, v_unpad = query_states.flatten(
            0, 1), key_states.flatten(0, 1), value_states.flatten(0, 1)
        cumulative_len = torch.cat(cumulative_len, dim=0)
        attn_output = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cumulative_len,
            cumulative_len,
            max_seqlen,
            max_seqlen,
            dropout_rate,
            return_attn_probs=False,
            causal=True,
        )
    else:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout_rate,
            causal=True)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def llama_varlen_attn_forward_legacy(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    is_training = self.training

    message_hub = MessageHub.get_instance('varlen_attn_args')
    rank = dist.get_rank()
    cumulative_len = message_hub.get_info(f'cumulative_len_rank_{rank}')
    indexes = message_hub.get_info(f'indexes_rank_{rank}')
    max_seqlen = message_hub.get_info(f'max_seqlen_rank_{rank}')
    assert is_training == (cumulative_len is not None)

    if 'padding_mask' in kwargs:
        warnings.warn('Passing `padding_mask` is deprecated and will be '
                      'removed in v4.37. Please make sure use '
                      '`attention_mask` instead.`')
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads *  # noqa: W504
            self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim)

    kv_seq_len = key_states.shape[-3]
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

    if is_training:
        cos, sin = self.rotary_emb(value_states, max_seqlen)
        query_states = apply_rotary_emb(query_states, cos[indexes].squeeze(0),
                                        sin[indexes].squeeze(0))
        key_states = apply_rotary_emb(key_states, cos[indexes].squeeze(0),
                                      sin[indexes].squeeze(0))
    else:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    assert SUPPORT_FLASH2
    if is_training:
        q_unpad, k_unpad, v_unpad = query_states.flatten(
            0, 1), key_states.flatten(0, 1), value_states.flatten(0, 1)
        cumulative_len = torch.cat(cumulative_len, dim=0)
        attn_output = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cumulative_len,
            cumulative_len,
            max_seqlen,
            max_seqlen,
            0,
            return_attn_probs=False,
            causal=True,
        )
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, causal=True)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value
