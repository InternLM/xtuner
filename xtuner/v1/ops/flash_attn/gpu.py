import torch
from flash_attn_interface import flash_attn_3_cuda, maybe_contiguous


@torch.library.custom_op("flash_attn::_flash_attn_varlen_forward_v3", mutates_args=(), device_types="cuda")
def _flash_attn_varlen_forward_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,  # -1 means infinite context window
    window_size_right: int = -1,
    softcap: float = 0.0,  # 0.0 means deactivated
) -> tuple[torch.Tensor, torch.Tensor]:
    out, softmax_lse, *rest = flash_attn_3_cuda.fwd(
        q,
        k,
        v,
        None,
        None,  # k_new, v_new
        None,  # qv
        None,  # out
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        None,
        None,  # seqused_q / seqused_k
        max_seqlen_q,
        max_seqlen_k,
        None,
        None,
        None,  # page_table, kv_batch_idx, leftpad_k,
        None,
        None,  # rotary_cos/sin
        None,  # seqlens_rotary
        None,
        None,
        None,  # q_descale, k_descale, v_descale
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        0,  # attention_chunk
        softcap,
        True,  # rotary_interleaved
        None,  # scheduler_metadata
        1,  # num_splits
        None,  # pack_gqa
        0,  # sm_margin
    )
    return out, softmax_lse


@torch.library.register_fake("flash_attn::_flash_attn_varlen_forward_v3")
def _flash_attn_varlen_forward_v3_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,  # -1 means infinite context window
    window_size_right: int = -1,
    softcap: float = 0.0,  # 0.0 means deactivated
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_heads, _ = q.shape
    q = q.contiguous()
    out = torch.empty_like(q)
    softmax_lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device, layout=q.layout)
    return out, softmax_lse


@torch.library.custom_op(
    "flash_attn::_flash_attn_varlen_backward_v3", mutates_args=("dq", "dk", "dv"), device_types="cuda"
)
def _flash_attn_varlen_backward_v3(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,  # -1 means infinite context window
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
) -> None:
    flash_attn_3_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,  # sequed_q, sequed_k
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        0,  # sm_margin
    )


@torch.library.register_fake("flash_attn::_flash_attn_varlen_backward_v3")
def _flash_attn_varlen_backward_v3_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,  # -1 means infinite context window
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
) -> None:
    return


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        softcap=0.0,
        deterministic=False,
        return_softmax=False,
    ):
        # modified from https://github.com/Dao-AILab/flash-attention/blob/afc97c60f799e470886c154e3473df938f8fa93d/hopper/flash_attn_interface.py#L369
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        q, k = (maybe_contiguous(x) for x in (q, k))
        v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
        cu_seqlens_q, cu_seqlens_k = (maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k))
        out, softmax_lse = _flash_attn_varlen_forward_v3(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            softcap,
        )
        # torch.distributed.breakpoint()
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size[0]
        ctx.window_size_right = window_size[1]
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        # torch.compile(fullgraph=True) can't handle dynamic tensor stride checks
        # since it needs to create a fully static computation graph.
        dout = dout.contiguous()
        out = maybe_contiguous(out)
        _flash_attn_varlen_backward_v3(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.softcap,
            ctx.deterministic,
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def gpu_flash_varlen_attn_v3(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    assert alibi_slopes is None, "Alibi is not supported yet."
    assert block_table is None, "block_table is not supported yet."
    assert dropout_p == 0.0, "Dropout is not supported yet."
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        softcap,
        deterministic,
        return_attn_probs,
    )
