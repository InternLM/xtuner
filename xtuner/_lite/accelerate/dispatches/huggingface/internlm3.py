# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange
from mmengine import MessageHub
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import lmdeploy_is_available
from .._attention import flash_attn_wo_mask, varlen_flash_attn

logger = get_logger()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):  # pylint: disable=unused-argument
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
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    if k is None:
        return q_embed, None
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


def _internlm3_self_attn_varlen_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    # Modified from https://huggingface.co/internlm/internlm-7b/blob/939a68c0dc1bd5f35b63c87d44af05ce33379061/modeling_internlm.py#L161  # noqa:E501
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            '`static` cache implementation is not compatible with '
            '`attn_implementation==flash_attention_2` make sure to use `sdpa` '
            'in the mean time, and open an issue at '
            'https://github.com/huggingface/transformers')

    bsz, q_len, _ = hidden_states.size()
    attn_context = MessageHub.get_instance('packed_sequence')


    _position_ids = attn_context.get_info('position_ids')
    if _position_ids is not None:
        assert _position_ids.size(1) == q_len, f'{_position_ids.size(1)} {q_len}'
        position_ids = _position_ids
    

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., :self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout
    # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # dropout_rate = self.attention_dropout if self.training else 0.0
    dropout_rate = 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (InternLM3RMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.wqkv.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # repeat kv for sequence parallel
    key_states = repeat_kv_bshd(key_states, self.num_key_value_groups)
    value_states = repeat_kv_bshd(value_states, self.num_key_value_groups)

    cumulative_lengths = attn_context.get_info('cumulative_lengths')
    if cumulative_lengths is not None and bsz == 1:

        max_seqlen = attn_context.get_info('max_seqlen')
        attn_output = varlen_flash_attn(query_states, key_states, value_states,
                                        cumulative_lengths, max_seqlen)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=True,
            training=self.training)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


def _internlm3_self_attn_contiguous_forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    """Rewrite implementation of LlamaAttention.forward.

    Add continuous batching support. Add paged attention support. TP support.
    """
    from lmdeploy.pytorch.kernels import \
        apply_rotary_pos_emb as apply_rotary_pos_emb_lmdeploy
    from lmdeploy.pytorch.kernels import fill_kv_cache, paged_attention_fwd
    attn_ctx = MessageHub.get_instance('paged_attention')
    kv_seq_length = attn_ctx.get_info('kv_seq_length')
    q_seq_length = attn_ctx.get_info('q_seq_length')
    q_start_loc = attn_ctx.get_info('q_start_loc')
    block_offsets = attn_ctx.get_info('block_offsets')
    max_q_seq_length = attn_ctx.get_info('max_q_seq_length')
    max_kv_seq_length = attn_ctx.get_info('max_kv_seq_length')
    cumulative_length = attn_ctx.get_info('cumulative_length')
    is_prefilling = attn_ctx.get_info('is_prefilling')

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., :self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)


    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    fill_kv_cache(
        key_states,
        value_states,
        past_key_value[self.layer_idx][0],
        past_key_value[self.layer_idx][1],
        q_start_loc,
        q_seq_length,
        kv_seq_length=kv_seq_length,
        max_q_seq_length=max_q_seq_length,
        block_offsets=block_offsets,
    )

    

    # attn_output = query_states

    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    if is_prefilling:
        # breakpoint()
        
        key_states = repeat_kv_bshd(
            key_states, self.num_key_value_groups)
        value_states = repeat_kv_bshd(
            value_states, self.num_key_value_groups)
        # breakpoint()
        attn_output = flash_attn_varlen_func(
            query_states.squeeze(0),
            key_states.squeeze(0),
            value_states.squeeze(0),
            cumulative_length,
            cumulative_length,
            max_q_seq_length,
            max_kv_seq_length,
            causal=True)
        # attn_output = varlen_flash_attn(query_states, key_states, value_states,
        #                             cumulative_length, max_q_seq_length)
    else:
        query_states = query_states.transpose(0,1)
        attn_output = flash_attn_with_kvcache(
            query_states,
            past_key_value[self.layer_idx][0],
            past_key_value[self.layer_idx][1],
            cache_seqlens=kv_seq_length,
            block_table=block_offsets,
            causal=True)
        attn_output = attn_output.squeeze(1)
    
    attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

    attn_output = self.wo(attn_output)

    return attn_output, None, past_key_value


def _internlm3_cross_attn_varlen_forward(
    self,
    hidden_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    attn_context = MessageHub.get_instance('packed_sequence')

    _position_ids = attn_context.get_info('position_ids')
    if _position_ids is not None:
        assert _position_ids.size(1) == q_len, f'{_position_ids.size(1)} {q_len}'
        position_ids = _position_ids
    

    if self.config.pretraining_tp > 1:
        # split qkv_states by tp size
        key_value_slicing = self.hidden_size // self.config.pretraining_tp
        q_slices = self.wq.weight.split(key_value_slicing, dim=0)
        query_states = torch.cat(
            [F.linear(hidden_states, q_slice) for q_slice in q_slices], dim=-1  # pylint: disable=E1102
        )
    else:
        query_states = self.wq(hidden_states)

    query_states = rearrange(query_states, "b q (h d) -> b q h d", d=self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    # Only query_states are rotated in cross-attention
    query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout
    # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # dropout_rate = self.attention_dropout if self.training else 0.0
    dropout_rate = 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (InternLM3RMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.wqkv.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    cumulative_lengths = attn_context.get_info('cumulative_lengths')
    if cumulative_lengths is not None and bsz == 1:
        max_seqlen = attn_context.get_info('max_seqlen')
        # breakpoint()
        attn_output = varlen_flash_attn(query_states, key_states, value_states,
                                        cumulative_lengths, max_seqlen)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=True,
            training=self.training)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    return attn_output, None


def _internlm3_cross_attn_contiguous_forward(
    self,
    hidden_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    from lmdeploy.pytorch.kernels import \
        apply_rotary_pos_emb as apply_rotary_pos_emb_lmdeploy
    from lmdeploy.pytorch.kernels import fill_kv_cache, paged_attention_fwd
    attn_ctx = MessageHub.get_instance('paged_attention')
    kv_seq_length = attn_ctx.get_info('kv_seq_length')
    q_seq_length = attn_ctx.get_info('q_seq_length')
    q_start_loc = attn_ctx.get_info('q_start_loc')
    block_offsets = attn_ctx.get_info('block_offsets')
    max_q_seq_length = attn_ctx.get_info('max_q_seq_length')
    max_kv_seq_length = attn_ctx.get_info('max_kv_seq_length')
    cumulative_length = attn_ctx.get_info('cumulative_length')
    is_prefilling = attn_ctx.get_info('is_prefilling')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    
    
    if self.config.pretraining_tp > 1:
        # split qkv_states by tp size
        key_value_slicing = self.hidden_size // self.config.pretraining_tp
        q_slices = self.wq.weight.split(key_value_slicing, dim=0)
        query_states = torch.cat(
            [F.linear(hidden_states, q_slice) for q_slice in q_slices], dim=-1  # pylint: disable=E1102
        )
    else:
        query_states = self.wq(hidden_states)

    query_states = rearrange(query_states, "b q (h d) -> b q h d", d=self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    # Only query_states are rotated in cross-attention
    query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout
    # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)

    # key_states = key_states.transpose(1, 2)
    # value_states = value_states.transpose(1, 2)

    # dropout_rate = self.attention_dropout if self.training else 0.0
    dropout_rate = 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (InternLM3RMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.wqkv.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    if is_prefilling:
        # breakpoint()
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        key_states = repeat_kv_bshd(
            key_states, self.num_key_value_groups)
        value_states = repeat_kv_bshd(
            value_states, self.num_key_value_groups)
        # breakpoint()
        attn_output = flash_attn_varlen_func(
            query_states.squeeze(0),
            key_states.squeeze(0),
            value_states.squeeze(0),
            cumulative_length,
            cumulative_length,
            max_q_seq_length,
            max_kv_seq_length,
            causal=True)
        # attn_output = varlen_flash_attn(query_states, key_states, value_states,
        #                             cumulative_length, max_q_seq_length)
    else:
        # breakpoint()
        query_states = query_states.transpose(0,1)
        
        attn_output = flash_attn_with_kvcache(
            query_states,
            key_states,
            value_states,
            cache_seqlens=kv_seq_length,
            block_table=block_offsets,
            causal=True)
        attn_output = attn_output.squeeze(1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    return attn_output, None


def internlm3_self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    lmdeploy_ctx = MessageHub.get_instance('paged_attention')

    if lmdeploy_is_available() and len(lmdeploy_ctx.runtime_info) > 0:

        # return _contiguous_batching_forward_impl(
        #     self, hidden_states, position_ids, past_key_value)
        return _internlm3_self_attn_contiguous_forward(self, hidden_states, position_ids,
                                past_key_value)
    else:
        return _internlm3_self_attn_varlen_forward(self, hidden_states,
                                              attention_mask, position_ids,
                                              past_key_value,
                                              output_attentions, use_cache)



def internlm3_cross_attn_forward(
    self,
    hidden_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    lmdeploy_ctx = MessageHub.get_instance('paged_attention')

    if lmdeploy_is_available() and len(lmdeploy_ctx.runtime_info) > 0:

        # return _contiguous_batching_forward_impl(
        #     self, hidden_states, position_ids, past_key_value)
        return _internlm3_cross_attn_contiguous_forward(self, hidden_states,key_states, value_states,
                    attention_mask, position_ids,output_attentions,use_cache, cache_position)
    else:
        return _internlm3_cross_attn_varlen_forward(self, hidden_states,key_states, value_states,
                    attention_mask, position_ids,output_attentions,use_cache, cache_position)


def internlm3_cross_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    output_hidden_states: bool = False,
    all_hidden_states: Optional[Tuple[torch.Tensor]] = None,
    all_self_attns: Optional[Tuple[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_values (`List[Cache]`, *optional*): cached past key and value projection states
    """

    
    next_decoder_cache = None

    hidden_states_norm = self.norm(hidden_states)

    key_states = self.wk(hidden_states_norm)
    value_states = self.wv(hidden_states_norm)

    key_states = rearrange(key_states, "b q (h d) -> b q h d", d=self.head_dim).transpose(1, 2)
    value_states = rearrange(value_states, "b q (h d) -> b q h d", d=self.head_dim).transpose(1, 2)

    # breakpoint()
    
    if past_key_values is not None:
        from lmdeploy.pytorch.kernels import \
        apply_rotary_pos_emb as apply_rotary_pos_emb_lmdeploy
        from lmdeploy.pytorch.kernels import fill_kv_cache, paged_attention_fwd
        attn_ctx = MessageHub.get_instance('paged_attention')
        kv_seq_length = attn_ctx.get_info('kv_seq_length')
        q_seq_length = attn_ctx.get_info('q_seq_length')
        q_start_loc = attn_ctx.get_info('q_start_loc')
        block_offsets = attn_ctx.get_info('block_offsets')
        max_q_seq_length = attn_ctx.get_info('max_q_seq_length')
        max_kv_seq_length = attn_ctx.get_info('max_kv_seq_length')
        cumulative_length = attn_ctx.get_info('cumulative_length')
        is_prefilling = attn_ctx.get_info('is_prefilling')
        # breakpoint()
        cos, sin = self.rotary_emb(value_states, position_ids)
        key_states, _ = apply_rotary_pos_emb(key_states, None, cos, sin, position_ids)

        # breakpoint()
        fill_kv_cache(
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            past_key_values[len(self.layers)][0],
            past_key_values[len(self.layers)][1],
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
        )
        if not is_prefilling:
            key_states = past_key_values[len(self.layers)][0]
            value_states = past_key_values[len(self.layers)][1]

    else:
        attn_context = MessageHub.get_instance('packed_sequence')

        position_ids = attn_context.get_info('position_ids')
        # assert position_ids.size(1) == q_len, f'{position_ids.size(1)} {q_len}'
        cos, sin = self.rotary_emb(value_states, position_ids)
        key_states, _ = apply_rotary_pos_emb(key_states, None, cos, sin, position_ids)

        
    for _, layer_module in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = layer_module(
            hidden_states,
            key_states,
            value_states,
            attention_mask,
            position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    return (layer_outputs[0], all_hidden_states, all_self_attns, next_decoder_cache)
