# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
from mmengine import MessageHub
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.processing_utils import Unpack

from xtuner.parallel.sequence import get_sequence_parallel_world_size
from xtuner.parallel.sequence.attention import (
    post_process_for_sequence_parallel_attn,
    pre_process_for_sequence_parallel_attn,
)


# modified from transformers.model.mistral.modeling_mistral.MistralAttention.forward and  # noqa: E501
# support sequence parallel
def mistral_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed
        # for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # different from MistralAttention.forward
    # repeat k/v heads if n_kv_heads < n_heads for sequence parallel
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    enable_sequence_parallel = (
        dist.is_initialized()
        and get_sequence_parallel_world_size() > 1
        and self.training
    )
    if enable_sequence_parallel:
        # Reashape for `pre_process_for_sequence_parallel_attn`
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states, key_states, value_states = pre_process_for_sequence_parallel_attn(
            query_states, key_states, value_states
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            warnings.warn(
                "`torch.nn.functional.scaled_dot_product_attention` does not "
                "support `output_attentions=True`. Falling back to eager "
                "attention. This warning can be removed using the argument"
                ' `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    message_hub = MessageHub.get_instance("varlen_attn_args")
    rank = dist.get_rank()
    cumulative_len = message_hub.get_info(f"cumulative_len_rank_{rank}")
    use_varlen_atten = cumulative_len is not None
    if use_varlen_atten:
        # When gradient_checkpointing is enabled, the flash_attn_kwargs
        # parameter is not automatically passed to the model. In such
        # cases, parameters like cu_seq_lens_q and max_length_q are
        # computed based on position_ids. However, when sequence
        # parallel is enabled, position_ids is split along the
        # sequence length, leading to incorrect calculations of these
        # parameters.
        # To address this issue, it is necessary to manually provide
        # the flash_attn_kwargs parameters.
        max_seqlen = message_hub.get_info(f"max_seqlen_rank_{rank}")
        kwargs["cu_seq_lens_q"] = cumulative_len
        kwargs["cu_seq_lens_k"] = cumulative_len
        kwargs["max_length_q"] = max_seqlen
        kwargs["max_length_k"] = max_seqlen
        kwargs.pop("position_ids", None)

    # Hacky: `sdpa_attention_forward` does repeat_kv based on
    # module.num_key_value_groups but it is done before
    num_key_value_groups = self.num_key_value_groups
    self.num_key_value_groups = 1
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(
            self.config, "sliding_window", None
        ),  # main diff with Llama
        **kwargs,
    )
    self.num_key_value_groups = num_key_value_groups

    # different from MistralAttention.forward
    if enable_sequence_parallel:
        attn_output = post_process_for_sequence_parallel_attn(attn_output)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
