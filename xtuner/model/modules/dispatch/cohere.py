# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.distributed as dist
import transformers
from mmengine.utils import digit_version
from transformers.models.cohere.modeling_cohere import apply_rotary_pos_emb

from xtuner.parallel.sequence import get_sequence_parallel_world_size
from xtuner.parallel.sequence.attention import (
    post_process_for_sequence_parallel_attn,
    pre_process_for_sequence_parallel_attn)

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


def cohere_attn_forward(
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
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim)
    if self.use_qk_norm:
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    past_key_value = getattr(self, 'past_key_value', past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for
        # the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires
    # the layout [batch_size, sequence_length, num_heads, head_dim].
    # We would need to refactor the KV cache to be able to avoid many of
    # these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # Ignore copy
    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
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

    enable_sequence_parallel = (
        dist.is_initialized() and get_sequence_parallel_world_size() > 1
        and self.training)
    if enable_sequence_parallel:
        query_states, key_states, value_states = \
            pre_process_for_sequence_parallel_attn(
                query_states, key_states, value_states)
        # self.num_heads is used in self._upad_input method
        # num_heads has been changed because of sequence parallel
        ori_num_head = self.num_heads
        self.num_heads = query_states.shape[-2]

    if IS_LOW_VERSION_TRANSFORMERS:
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_states.shape[1],
            dropout=dropout_rate)
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_states.shape[1],
            dropout=dropout_rate,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

    if enable_sequence_parallel:
        attn_output = post_process_for_sequence_parallel_attn(attn_output)
        self.num_heads = ori_num_head

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
