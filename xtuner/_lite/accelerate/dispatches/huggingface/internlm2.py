# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union
import inspect

import torch
from einops import rearrange
from mmengine import MessageHub
from transformers.cache_utils import StaticCache, Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast

from xtuner._lite.accelerate import lmdeploy_is_available, liger_kernel_is_available
from .._attention import flash_attn_wo_mask, varlen_flash_attn


class InternLM2RotaryEmbedding(torch.nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=1000000,
                 device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (
            base**(torch.arange(0, dim, 2).float().to(device) / dim))

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if (seq_len > self.max_seq_len_cached
                or self.cos_cached.device != x.device
                or self.cos_cached.dtype != x.dtype):
            self.max_seq_len_cached = seq_len
            assert self.inv_freq.dtype == torch.float32
            t = torch.arange(
                self.max_seq_len_cached,
                device=x.device,
                dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(t.device))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().to(x.dtype)
            self.sin_cached = emb.sin().to(x.dtype)
        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_old(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
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


def _internlm2_varlen_attn_forward(
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

    position_ids = attn_context.get_info('position_ids')
    sp_mesh = attn_context.get_info('sp_mesh')
    assert position_ids.size(1) == q_len, f'{position_ids.size(1)} {q_len}'

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

    signature = inspect.signature(self.rotary_emb.forward)
    if 'seq_len' in signature.parameters:
        # old
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)
        query_states, key_states = apply_rotary_pos_emb_old(query_states, key_states, cos, sin, position_ids)
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin)

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

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32. (InternLM2RMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.wqkv.weight.dtype

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
                                        cumulative_lengths, max_seqlen,training=self.training, sp_mesh=sp_mesh)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=True,
            training=self.training,
            sp_mesh=sp_mesh)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


def _contiguous_batching_forward_impl(
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
    position_ids = attn_ctx.get_info('position_ids')

    # position_ids
    def __qkv_proj(hidden_states):
        """qkv_proj."""
        # from torch.distributed import get_rank
        # if get_rank() == 0:
        #     breakpoint()
        # else:
        #     import time
        #     time.sleep(10000)
        qkv_states = self.wqkv(hidden_states[0]).unsqueeze(0)

        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> (b q) h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )
        query_states = qkv_states[..., :self.num_key_value_groups, :]
        query_states = query_states.flatten(1, 2)
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]
        return query_states, key_states, value_states

    from lmdeploy.pytorch.kernels import \
        apply_rotary_pos_emb as apply_rotary_pos_emb_lmdeploy

    def __rotary_emb_fn(query_states, key_states, value_states):
        """rotary embedding func."""
        # breakpoint()
        # query_states = query_states.unsqueeze(0).transpose(1, 2)
        # key_states = key_states.unsqueeze(0).transpose(1, 2)
        # value_states = value_states.unsqueeze(0).transpose(1, 2)
        if self.layer_idx == 0:

            cos, sin = self.rotary_emb(value_states, position_ids)
            attn_ctx.update_info('rotary_cos_sin', (cos, sin))
        else:
            cos, sin = attn_ctx.get_info('rotary_cos_sin')

        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
        #                                         cos, sin)
        # query_states = query_states.transpose(1, 2).squeeze(0)
        # key_states = key_states.transpose(1, 2).squeeze(0)
        # value_states = value_states.transpose(1, 2).squeeze(0)
        query_states, key_states = apply_rotary_pos_emb_lmdeploy(
            query_states,
            key_states,
            cos,
            sin,
            q_embed=query_states,
            k_embed=key_states)

        return query_states, key_states, value_states

    query_states, key_states, value_states = __qkv_proj(hidden_states)

    query_states, key_states, value_states = __rotary_emb_fn(
        query_states, key_states, value_states)

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

    attn_output = query_states
    paged_attention_fwd(
        query_states,
        past_key_value[self.layer_idx][0],
        past_key_value[self.layer_idx][1],
        attn_output,
        block_offsets,
        q_start_loc=q_start_loc,
        q_seqlens=q_seq_length,
        kv_seqlens=kv_seq_length,
        max_seqlen=max_q_seq_length,
        # max_kv_seq_length=max_kv_seq_length,
    )
    attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

    attn_output = self.wo(attn_output)

    return attn_output, None, past_key_value


def _flash_att_infer(
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
        'b q (h gs d) -> (b q) h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )
    query_states = qkv_states[..., :self.num_key_value_groups, :]
    query_states = query_states.flatten(1, 2)
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]


    from lmdeploy.pytorch.kernels import \
        apply_rotary_pos_emb as apply_rotary_pos_emb_lmdeploy

    if self.layer_idx == 0:

        cos, sin = self.rotary_emb(value_states, position_ids)
        attn_ctx.update_info('rotary_cos_sin', (cos, sin))
    else:
        cos, sin = attn_ctx.get_info('rotary_cos_sin')

    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
    #                                         cos, sin)
    # query_states = query_states.transpose(1, 2).squeeze(0)
    # key_states = key_states.transpose(1, 2).squeeze(0)
    # value_states = value_states.transpose(1, 2).squeeze(0)
    query_states, key_states = apply_rotary_pos_emb_lmdeploy(
        query_states,
        key_states,
        cos,
        sin,
        q_embed=query_states,
        k_embed=key_states)

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
            key_states.unsqueeze(0), self.num_key_value_groups).squeeze(0)
        value_states = repeat_kv_bshd(
            value_states.unsqueeze(0), self.num_key_value_groups).squeeze(0)
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cumulative_length,
            cumulative_length,
            max_q_seq_length,
            max_kv_seq_length,
            causal=True)
        # attn_output = varlen_flash_attn(query_states, key_states, value_states,
        #                             cumulative_length, max_q_seq_length)
    else:
        query_states = query_states.unsqueeze(1)
        attn_output = flash_attn_with_kvcache(
            query_states,
            past_key_value[self.layer_idx][0],
            past_key_value[self.layer_idx][1],
            cache_seqlens=kv_seq_length,
            block_table=block_offsets,
            causal=True)
        attn_output = attn_output.squeeze(1)
    # paged_attention_fwd(
    #     query_states,
    #     past_key_value[self.layer_idx][0],
    #     past_key_value[self.layer_idx][1],
    #     attn_output,
    #     block_offsets,
    #     q_start_loc=q_start_loc,
    #     q_seqlens=q_seq_length,
    #     kv_seqlens=kv_seq_length,
    #     max_seqlen=max_q_seq_length,
    # )
    attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

    attn_output = self.wo(attn_output)

    return attn_output, None, past_key_value


def internlm2_varlen_attn_forward(
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
        return _flash_att_infer(self, hidden_states, position_ids,
                                past_key_value)
    else:
        return _internlm2_varlen_attn_forward(self, hidden_states,
                                              attention_mask, position_ids,
                                              past_key_value,
                                              output_attentions, use_cache)


def internlm2_reward_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, SequenceClassifierOutputWithPast]:
    """labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*): Labels
    for computing the sequence classification/regression loss.

    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    reward_scores = self.v_head(hidden_states).squeeze(-1)

    loss = None

    # hidden_states = outputs[0]
    # hidden_states = self.v_head(hidden_states)
    # # get end reward token's score
    # ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)

    # reward_scores = torch.gather(hidden_states.squeeze(-1), 1, ends)

    loss = None

    # if not return_dict:
    #     ssoutput = (reward_scores,) + outputs[1:]
    #     return (loss,) + output if loss is not None else output

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=reward_scores,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )




def internlm2_causal_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    label_shifted: bool = False,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    Returns:
    Example:
    ```python
    >>> from transformers import AutoTokenizer, InternLM2ForCausalLM
    >>> model = InternLM2ForCausalLM.from_pretrained("meta-InternLM2/InternLM2-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-InternLM2/InternLM2-2-7b-hf")
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")
    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    loss = None
    if labels is None:
        logits = self.output(hidden_states)
    else:

        if liger_kernel_is_available():
            # unable to return logits when using Liger Kernel
            logits = None

            if label_shifted:
                shift_hidden_states = hidden_states
                shift_labels = labels
            else:
                shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_hidden_states.device)

            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

            loss_fct = LigerFusedLinearCrossEntropyLoss()
            loss = loss_fct(self.output.weight, shift_hidden_states, shift_labels, self.output.bias)

        else:
            logits = self.output(hidden_states)

            if label_shifted:
                shift_logits = logits
                shift_labels = labels
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )