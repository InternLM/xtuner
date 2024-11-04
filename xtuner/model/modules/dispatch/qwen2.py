# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from mmengine import MessageHub
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (apply_rotary_pos_emb,
                                                      repeat_kv)

from xtuner.parallel.sequence import get_sequence_parallel_world_size
from xtuner.parallel.sequence.attention import (
    post_process_for_sequence_parallel_attn,
    pre_process_for_sequence_parallel_attn)
from .attention import flash_attn_wo_mask, varlen_flash_attn

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func
    _flash_supports_window_size = 'window_size' in list(
        inspect.signature(flash_attn_func).parameters)
    SUPPORT_FLASH2 = True
except ImportError:
    pass

try:
    from transformers.modeling_flash_attention_utils import \
        _flash_attention_forward
except ImportError:
    _flash_attention_forward = None


# Modified from https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/models/qwen2/modeling_qwen2.py#L364  # noqa: E501
# and sequence parallel is supported.
def qwen2_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor,
                  torch.Tensor]] = None,  # will become mandatory in v4.46
):
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

    if position_embeddings is None:
        warnings.warn(
            'The attention layers in this model are transitioning from'
            ' computing the RoPE embeddings internally through `position_ids` '
            '(2D tensor with the indexes of the tokens), to using externally '
            'computed `position_embeddings` (Tuple of tensors, containing cos '
            'and sin). In v4.46 `position_ids` will be removed and '
            '`position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    if past_key_value is not None:
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for
    # training stability reasons, therefore the input hidden states gets
    # silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        warnings.warn(
            f'The input hidden states seems to be silently casted in float32,'
            ' this might be related to  the fact you have upcasted embedding '
            'or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

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
        query_states, key_states, value_states = \
            pre_process_for_sequence_parallel_attn(
                query_states, key_states, value_states)

    if (self.config.use_sliding_window
            and getattr(self.config, 'sliding_window', None) is not None
            and self.layer_idx >= self.config.max_window_layers):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    if _flash_attention_forward is None:
        raise RuntimeError('Please install Transformers >= 4.46.1.')

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_states.shape[1],
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    if enable_sequence_parallel:
        attn_output = post_process_for_sequence_parallel_attn(attn_output)

    attn_output = attn_output.reshape(bsz, q_len,
                                      self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def qwen2_varlen_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor,
                  torch.Tensor]] = None,  # will become mandatory in v4.46
):
    message_hub = MessageHub.get_instance('varlen_attn_args')
    rank = dist.get_rank()
    cumulative_len = message_hub.get_info(f'cumulative_len_rank_{rank}')
    max_seqlen = message_hub.get_info(f'max_seqlen_rank_{rank}')
    use_varlen_atten = (cumulative_len is not None)

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

    if position_embeddings is None:
        warnings.warn(
            'The attention layers in this model are transitioning from'
            ' computing the RoPE embeddings internally through `position_ids` '
            '(2D tensor with the indexes of the tokens), to using externally '
            'computed `position_embeddings` (Tuple of tensors, containing cos '
            'and sin). In v4.46 `position_ids` will be removed and '
            '`position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    if past_key_value is not None:
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for
    # training stability reasons, therefore the input hidden states gets
    # silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        warnings.warn(
            f'The input hidden states seems to be silently casted in float32,'
            ' this might be related to  the fact you have upcasted embedding '
            'or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # ----------------- flash attention forward ------------------------#

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, 'sliding_window', None) is not None
        and key_states.shape[1] > self.config.sliding_window
        and self.config.use_sliding_window)
    # Decide whether to use SWA or not by layer index.
    if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
        use_sliding_windows = False

    window_size = (self.config.sliding_window,
                   self.config.sliding_window) if use_sliding_windows else (-1,
                                                                            -1)

    if use_varlen_atten:
        attn_output = varlen_flash_attn(
            query_states,
            key_states,
            value_states,
            cumulative_len,
            max_seqlen,
            causal=self.is_causal,
            dropout_p=dropout_rate,
            window_size=window_size,
            training=self.training)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=self.is_causal,
            dropout_p=dropout_rate,
            window_size=window_size,
            training=self.training)

    # ---------------- flash attention forward end ------------------- #

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
