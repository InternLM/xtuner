from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionModel

from ._attention import flash_attn_wo_mask


def clip_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel."""

    bsz, tgt_len, embed_dim = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states).view(bsz, tgt_len,
                                                   self.num_heads, -1)
    key_states = self.k_proj(hidden_states).view(bsz, tgt_len, self.num_heads,
                                                 -1)
    value_states = self.v_proj(hidden_states).view(bsz, tgt_len,
                                                   self.num_heads, -1)

    # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    # key_states = key_states.view(*proj_shape)
    # value_states = value_states.view(*proj_shape)

    # src_len = key_states.size(1)
    # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    #     raise ValueError(
    #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
    #         f" {attn_weights.size()}"
    #     )

    # # apply the causal_attention_mask first
    # if causal_attention_mask is not None:
    #     if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
    #         raise ValueError(
    #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
    #             f" {causal_attention_mask.size()}"
    #         )
    #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
    #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    # if attention_mask is not None:
    #     if attention_mask.size() != (bsz, 1, tgt_len, src_len):
    #         raise ValueError(
    #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
    #         )
    #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    # attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # if output_attentions:
    #     # this operation is a bit akward, but it's required to
    #     # make sure that attn_weights keeps its gradient.
    #     # In order to do so, attn_weights have to reshaped
    #     # twice and have to be reused in the following
    #     attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
    #     attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    # else:
    #     attn_weights_reshaped = None

    # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    # attn_output = torch.bmm(attn_probs, value_states)

    # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )

    # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    # attn_output = attn_output.transpose(1, 2)
    # attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = flash_attn_wo_mask(
        query_states,
        key_states,
        value_states,
        self.dropout if self.training else 0,
        causal=causal_attention_mask is not None).view(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None
