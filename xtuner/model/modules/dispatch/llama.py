# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import MessageHub
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.utils import logging

from xtuner.parallel.sequence import sequence_parallel_wrapper
from .triton_kernels import apply_rotary_emb
from .utils import upad_qkv

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
    SUPPORT_FLASH2 = True
except ImportError:
    pass

try:
    from transformers.cache_utils import Cache
except ImportError:

    class Cache:
        pass


logger = logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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


def repeat_kv_bshd(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """The hidden states go from (batch, seqlen, num_key_value_heads, head_dim)
    to (batch, seqlen, num_attention_heads, head_dim)"""
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :,
                                  None, :].expand(batch, slen,
                                                  num_key_value_heads, n_rep,
                                                  head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep,
                                 head_dim)


@sequence_parallel_wrapper
def flash_attn_wo_mask(query_states,
                       key_states,
                       value_states,
                       causal,
                       dropout_rate=0.0):
    attn_output = flash_attn_func(
        query_states, key_states, value_states, dropout_rate, causal=causal)
    return attn_output


@sequence_parallel_wrapper
def flash_attn_w_mask(
        query_states,  # bs, q_len, nhead, h_dim
        key_states,
        value_states,
        attention_mask,
        causal,
        dropout_rate=0.0):
    batch_size, q_len = query_states.shape[:2]
    query_states, key_states, value_states, indices_q, \
        cu_seq_lens, max_seq_lens = upad_qkv(
            query_states, key_states, value_states, attention_mask, q_len)

    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
    attn_output_unpad = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_in_batch_q,
        max_seqlen_k=max_seqlen_in_batch_k,
        dropout_p=dropout_rate,
        causal=causal,
    )
    attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)
    return attn_output


def flash_attn1_pytorch(query_states, key_states, value_states, *args,
                        **kwargs):
    # hacky: pytorch flash attn need (bs, n_head, seq_len, h_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    attn_output = F.scaled_dot_product_attention(query_states, key_states,
                                                 value_states, *args, **kwargs)
    attn_output = attn_output.transpose(1, 2)
    return attn_output


@sequence_parallel_wrapper
def varlen_flash_attn(query_states, key_states, value_states, cumulative_len,
                      max_seqlen):
    q_unpad, k_unpad, v_unpad = query_states.flatten(0, 1), key_states.flatten(
        0, 1), value_states.flatten(0, 1)
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
    attn_output = attn_output.unsqueeze(0)
    return attn_output


def llama_attn_forward_legacy(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    # Modified from https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/models/llama/modeling_llama.py#L331  # noqa:E501

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

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat kv for sequence parallel
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # flash attn 2 need (bs, seq_len, nhead, h_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if SUPPORT_FLASH2:
        causal = self.is_causal and q_len != 1

        if attention_mask is not None:
            attn_output = flash_attn_w_mask(
                query_states,
                key_states,
                value_states,
                attention_mask,
                causal,
                training=self.training)
        else:
            attn_output = flash_attn_wo_mask(
                query_states,
                key_states,
                value_states,
                causal,
                training=self.training)
    else:
        # use flash attention implemented by pytorch
        attn_output = flash_attn1_pytorch(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            training=self.training)

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


def llama_attn_forward(
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
    # LlamaFlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37'
            ' Please make sure use `attention_mask` instead.`')

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

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

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                       self.layer_idx)

    if self.training:
        assert position_ids is not None
        cos, sin = self.rotary_emb(
            value_states, seq_len=position_ids.max() + 1)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention
    # requires the layout [batch_size, sequence_length, num_heads, head_dim].
    # We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons, therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32. (LlamaRMSNorm handles it correctly)
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        # Handle the case where the model is quantized
        if hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, '
            f'this might be related to the fact you have upcasted embedding '
            f'or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # flash attn
    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        # TODO: Remove the `q_len != 1` check once Flash Attention for RoCm
        # is bumped to 2.1. For details, please see the comment in
        # LlamaFlashAttention2 __init__.
        causal = self.is_causal and q_len != 1

    # repeat kv for sequence parallel
    key_states = repeat_kv_bshd(key_states, self.num_key_value_groups)
    value_states = repeat_kv_bshd(value_states, self.num_key_value_groups)

    if attention_mask is not None:
        attn_output = flash_attn_w_mask(
            query_states,
            key_states,
            value_states,
            attention_mask,
            causal,
            dropout_rate,
            training=self.training)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal,
            dropout_rate,
            training=self.training)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_varlen_attn_forward_legacy(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    is_training = self.training

    message_hub = MessageHub.get_instance('varlen_attn_args')
    rank = dist.get_rank()
    cumulative_len = message_hub.get_info(f'cumulative_len_rank_{rank}')
    # position_ids = message_hub.get_info(f'position_ids_rank_{rank}')
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
        kv_seq_len += past_key_value[0].shape[-2]

    if is_training:
        cos, sin = self.rotary_emb(value_states, max_seqlen)
        query_states = apply_rotary_emb(query_states,
                                        cos[position_ids].squeeze(0),
                                        sin[position_ids].squeeze(0))
        key_states = apply_rotary_emb(key_states, cos[position_ids].squeeze(0),
                                      sin[position_ids].squeeze(0))
    else:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    # repeat kv for sequence parallel
    key_states = repeat_kv_bshd(key_states, self.num_key_value_groups)
    value_states = repeat_kv_bshd(value_states, self.num_key_value_groups)

    assert SUPPORT_FLASH2
    if is_training:
        attn_output = varlen_flash_attn(query_states, key_states, value_states,
                                        cumulative_len, max_seqlen)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=True,
            training=False)

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
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    is_training = self.training

    message_hub = MessageHub.get_instance('varlen_attn_args')
    rank = dist.get_rank()
    cumulative_len = message_hub.get_info(f'cumulative_len_rank_{rank}')
    # position_ids = message_hub.get_info(f'position_ids_rank_{rank}')
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
        query_states = apply_rotary_emb(query_states,
                                        cos[position_ids].squeeze(0),
                                        sin[position_ids].squeeze(0))
        key_states = apply_rotary_emb(key_states, cos[position_ids].squeeze(0),
                                      sin[position_ids].squeeze(0))
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

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    # repeat kv for sequence parallel
    key_states = repeat_kv_bshd(key_states, self.num_key_value_groups)
    value_states = repeat_kv_bshd(value_states, self.num_key_value_groups)

    assert SUPPORT_FLASH2
    if is_training:
        attn_output = varlen_flash_attn(query_states, key_states, value_states,
                                        cumulative_len, max_seqlen)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=True,
            training=False)

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
