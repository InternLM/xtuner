import traceback
from functools import lru_cache
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
)
from torch.nn.attention.flex_attention import (
    create_block_mask as create_block_causal_mask_flex,
)
from torch.nn.attention.flex_attention import (
    flex_attention as torch_flex_attention,
)

from transformers.models.llama.modeling_llama import repeat_kv


try:
    from .flash_attn import flash_attn_varlen_func

    flash_attn_exception = None
except (ImportError, ModuleNotFoundError) as e:
    flash_attn_varlen_func = None  # type: ignore[assignment]
    flash_attn_exception = e

try:
    from .flash_attn.flash_sink_varlen_attn_gpt_oss import flash_sink_attn_varlen_func

    flash_sink_attn_exception = None
except (ImportError, ModuleNotFoundError) as e:
    flash_sink_attn_varlen_func = None  # type: ignore[assignment]
    flash_sink_attn_exception = e


def get_flex_attention_compiled():
    torch._dynamo.config.cache_size_limit = 128
    return torch.compile(torch_flex_attention, dynamic=False)


flex_attention_compiled = None


class AttnOpOutputs(TypedDict):
    raw_output: torch.Tensor
    softmax_lse: torch.Tensor | None
    attn_logits: torch.Tensor | None


# Refer to TorchTune
# We cannot do nested compile, but flex attention only has perf benefits
# when compiled. To insulate it from the compiler, we wrap it with
# compiler.disable so that it can be used regardless of whether the model
# is compiled or not, and flex attention always remains compiled.
@torch.compiler.disable(recursive=False)
def compile_friendly_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask,
    score_mod=None,
    enable_gqa: bool = False,
    scale: float | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    global flex_attention_compiled
    if flex_attention_compiled is None:
        flex_attention_compiled = get_flex_attention_compiled()
    return flex_attention_compiled(  # type: ignore
        q, k, v, block_mask=block_mask, score_mod=score_mod, scale=scale, enable_gqa=enable_gqa, return_lse=return_lse
    )


def _get_document_ids_from_seq_lens(
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    seq_lens = cu_seq_lens[1:] - cu_seq_lens[:-1]
    document_ids = []
    for i, seq_len in enumerate(seq_lens):
        document_id = torch.full((seq_len,), i, dtype=torch.long, device=seq_lens.device)
        document_ids.append(document_id)
    document_ids = torch.cat(document_ids, dim=0)  # type: ignore
    return document_ids[None]  # type: ignore


def _create_grouped_causal_mask(document_ids):
    _, seq_len = document_ids.shape

    doc_matrix = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=document_ids.device)).bool()

    final_mask = doc_matrix & causal_mask

    return torch.where(final_mask, 0.0, float("-inf"))


def _create_windowed_grouped_causal_mask(document_ids, window_size):
    _, seq_len = document_ids.shape

    positions = torch.arange(seq_len, device=document_ids.device)
    rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
    window_mask = (rel_pos >= 0) & (rel_pos < window_size)
    doc_mask = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)

    final_mask = doc_mask & window_mask

    return torch.where(final_mask, 0.0, float("-inf"))


@lru_cache
def create_packing_block_causal_mask(seq_lens: torch.Tensor, window_size=(-1, -1), causal=True) -> BlockMask:
    document_ids = _get_document_ids_from_seq_lens(seq_lens)
    _, max_seq_len = document_ids.shape

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]

        if causal:
            document_causal_mask = causal_mask & document_mask
        else:
            document_causal_mask = document_mask

        if window_size != (-1, -1):
            window_mask = q_idx - kv_idx < window_size[0]
            document_causal_mask = document_causal_mask & window_mask
        return document_causal_mask

    return create_block_causal_mask_flex(mask_mod, None, None, max_seq_len, max_seq_len)


def eager_attention(
    q, k, v, cu_seqlens_q, softmax_scale, window_size=(-1, -1), dropout_p=0.0, s_aux=None, **kwargs
) -> AttnOpOutputs:
    # TODO(HHA): Currently, the mask is recalculated each time, which is quite time-consuming.
    # It should be refactored to be calculated only once.

    # q, k, v: [b, n_head, seq, head_dim]
    num_kv_heads = k.size(1)
    num_q_heads = q.size(1)
    if num_kv_heads != num_q_heads:
        k = repeat_kv(k, num_q_heads // num_kv_heads)
        v = repeat_kv(v, num_q_heads // num_kv_heads)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) * softmax_scale  # type: ignore

    batch_document_ids = _get_document_ids_from_seq_lens(cu_seqlens_q)
    if window_size == (-1, -1):
        # Generate casual mask, the lower left corner is 0, and the other positions are -inf
        attention_mask = _create_grouped_causal_mask(batch_document_ids)
    else:
        attention_mask = _create_windowed_grouped_causal_mask(batch_document_ids, window_size[0])  # type: ignore

    attention_mask = attention_mask[None].to(attn_weights.dtype)  # 1,1,seq,seq
    causal_mask = attention_mask[:, :, :, : k.shape[-2]]
    attn_weights = attn_weights + causal_mask

    attn_logits = None
    if s_aux is not None:
        # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
        # when training with bsz>1 we clamp max values.
        sinks = s_aux
        sinks = sinks.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)  # type: ignore
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]  # we drop the sink here
        attn_logits = combined_logits.detach()
    else:
        scores = torch.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)
        attn_logits = attn_weights.detach()

    attn_scores = nn.functional.dropout(scores, p=dropout_p, training=True)
    raw_output = torch.matmul(attn_scores, v).transpose(1, 2).contiguous()
    attn_outputs: AttnOpOutputs = {
        "raw_output": raw_output,
        "attn_logits": attn_logits,
        "softmax_lse": None,
    }
    return attn_outputs


def flex_attention(
    q, k, v, cu_seqlens_q, softmax_scale=None, window_size=(-1, -1), dropout_p=0.0, s_aux=None, causal=True, **kwargs
) -> AttnOpOutputs:
    # q, k, v: [b, n_head, seq, head_dim]
    assert dropout_p == 0.0, "Dropout is not supported in flex attention"

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        if s_aux is not None:
            raise NotImplementedError("s_aux is not supported in flex attention yet")
        return score

    if s_aux is None:
        score_mod_fn = None
    else:
        score_mod_fn = score_mod

    mask = create_packing_block_causal_mask(cu_seqlens_q, window_size=window_size, causal=causal)
    enable_gqa = k.size(1) != q.size(1)

    raw_output, softmax_lse = compile_friendly_flex_attention(
        q,
        k,
        v,
        block_mask=mask,
        score_mod=score_mod_fn,
        scale=softmax_scale,
        enable_gqa=enable_gqa,
        return_lse=True,
    )
    raw_output = raw_output.transpose(1, 2).contiguous()
    attn_outputs: AttnOpOutputs = {
        "raw_output": raw_output,
        "softmax_lse": softmax_lse.detach(),
        "attn_logits": None,
    }
    return attn_outputs


def flash_attention(q, k, v, window_size=(-1, -1), s_aux=None, **kwargs) -> AttnOpOutputs:
    # q, k, v: [b, n_head, seq , head_dim]
    assert q.size(0) == 1, "Only support batch size 1 for flash attention"
    q = q.transpose(1, 2).squeeze(0)  # [seq, head, dim]
    k = k.transpose(1, 2).squeeze(0)
    v = v.transpose(1, 2).squeeze(0)

    raw_output: torch.Tensor | None = None
    softmax_lse: torch.Tensor | None = None
    if s_aux is None:
        if flash_attn_exception is not None:
            traceback.print_exception(flash_attn_exception)
            raise flash_attn_exception
        fla_outputs = flash_attn_varlen_func(q, k, v, return_attn_probs=True, **kwargs)  # type: ignore
        if isinstance(fla_outputs, tuple):
            raw_output = fla_outputs[0]
            softmax_lse = fla_outputs[1].detach()
        else:  # npu fused attn doesn't support softmax_lse
            raw_output = fla_outputs
    else:
        if flash_sink_attn_exception is not None:
            traceback.print_exception(flash_sink_attn_exception)
            raise flash_sink_attn_exception
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        fla_outputs = flash_sink_attn_varlen_func(q, k, v, s_aux, cu_seqlens_q, window_size[0])
        raw_output = fla_outputs[0]
        softmax_lse = fla_outputs[1].detach()
    attn_outputs: AttnOpOutputs = {
        "raw_output": raw_output[None],
        "softmax_lse": softmax_lse,
        "attn_logits": None,
    }
    return attn_outputs


attn_impl_mapping = {
    "eager_attention": eager_attention,
    "flash_attention": flash_attention,
    "flex_attention": flex_attention,
}
