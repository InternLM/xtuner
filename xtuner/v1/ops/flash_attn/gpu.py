from typing import Optional, Tuple

import torch


try:
    from flash_attn.flash_attn_interface import flash_attn_gpu, round_multiple
    from flash_attn.flash_attn_interface import maybe_contiguous as maybe_contiguous_v3

    @torch.library.custom_op("flash_attn::_flash_attn_varlen_forward_v3", mutates_args=(), device_types="cuda")
    def _flash_attn_varlen_forward_v3(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
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
            max_seqlen_q.item(),
            max_seqlen_k.item(),
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
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
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
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
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
            max_seqlen_q.item(),
            max_seqlen_k.item(),
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
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
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

    class FlashAttnVarlenFuncV3(torch.autograd.Function):
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
            q, k = (maybe_contiguous_v3(x) for x in (q, k))
            v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
            cu_seqlens_q, cu_seqlens_k = (maybe_contiguous_v3(x) for x in (cu_seqlens_q, cu_seqlens_k))
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
            out = maybe_contiguous_v3(out)
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
        return FlashAttnVarlenFuncV3.apply(
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
except ImportError:
    gpu_flash_varlen_attn_v3 = None  # type: ignore


try:
    from flash_attn_interface import flash_attn_3_cuda
    from flash_attn_interface import maybe_contiguous as maybe_contiguous_v2

    def flash_attn_varlen_func_v2(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
    ):
        """dropout_p should be set to 0.0 during evaluation Supports multi-
        query and grouped-query attention (MQA/GQA) by passing in K, V with
        fewer heads than Q. Note that the number of heads in Q must be
        divisible by the number of heads in KV. For example, if Q has 6 heads
        and K, V have 2 heads, head 0, 1, 2 of Q will attention to head 0 of K,
        V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

        If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
            1 1 1 1 0
            1 1 1 1 1
        If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
            0 0
            0 0
            0 0
            1 0
            1 1
        If the row of the mask is all zero, the output will be zero.

        If window_size != (-1, -1), implements sliding window local attention. Query at position i
        will only attend to keys between
        [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

        Arguments:
            q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
            k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
               of the sequences in the batch, used to index into q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
               of the sequences in the batch, used to index into kv.
            max_seqlen_q: int. Maximum query sequence length in the batch.
            max_seqlen_k: int. Maximum key sequence length in the batch.
            dropout_p: float. Dropout probability.
            softmax_scale: float. The scaling of QK^T before applying softmax.
                Default to 1 / sqrt(headdim).
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            window_size: (left, right). If not (-1, -1), implements sliding window local attention.
            softcap: float. Anything > 0 activates softcapping attention.
            alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
                (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
                is added to the attention score of query i and key j.
            deterministic: bool. Whether to use the deterministic implementation of the backward pass,
                which is slightly slower and uses more memory. The forward pass is always deterministic.
            return_attn_probs: bool. Whether to return the attention probabilities. This option is for
               testing only. The returned probabilities are not guaranteed to be correct
               (they might not have the right scaling).
        Return:
            out: (total, nheads, headdim).
            softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
                logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
                normalization factor).
            S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
                The output of softmax (possibly with different scaling). It also encodes the dropout
                pattern (negative means that location was dropped, nonnegative means it was kept).
        """
        return FlashAttnVarlenFunc.apply(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            block_table,
            torch.is_grad_enabled(),
        )

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
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_softmax,
            block_table,
            is_grad_enabled,
        ):
            is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            head_size_og = q.size(2)
            if head_size_og % 8 != 0:
                q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
                k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
                v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
            out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=window_size[0],
                window_size_right=window_size[1],
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                return_softmax=return_softmax and dropout_p > 0,
                block_table=block_table,
            )
            if is_grad:
                ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
                ctx.dropout_p = dropout_p
                ctx.max_seqlen_q = max_seqlen_q
                ctx.max_seqlen_k = max_seqlen_k
                ctx.softmax_scale = softmax_scale
                ctx.causal = causal
                ctx.window_size = window_size
                ctx.softcap = softcap
                ctx.alibi_slopes = alibi_slopes
                ctx.deterministic = deterministic

            out = out_padded[..., :head_size_og]
            return out if not return_softmax else (out, softmax_lse, S_dmask)

        @staticmethod
        def backward(ctx, dout, *args):
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            head_size_og = dout.size(2)
            dout_padded = dout
            if head_size_og % 8 != 0:
                dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
            _flash_attn_varlen_backward(
                dout_padded,
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
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                ctx.window_size[0],
                ctx.window_size[1],
                ctx.softcap,
                ctx.alibi_slopes,
                ctx.deterministic,
                rng_state=rng_state,
            )
            dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
            dk = dk[..., : dout.shape[-1]]
            dv = dv[..., : dout.shape[-1]]
            return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    @torch.library.custom_op("flash_attn::_flash_attn_varlen_forward_v2", mutates_args=(), device_types="cuda")
    def _flash_attn_varlen_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int = -1,
        window_size_right: int = -1,
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        return_softmax: bool = False,
        block_table: Optional[torch.Tensor] = None,
        leftpad_k: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        zero_tensors: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = (maybe_contiguous_v2(x) for x in (q, k, v))
        out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.varlen_fwd(
            q,
            k,
            v,
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            leftpad_k,
            block_table,
            alibi_slopes,
            max_seqlen_q.item(),
            max_seqlen_k.item(),
            dropout_p,
            softmax_scale,
            zero_tensors,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            return_softmax,
            None,
        )
        # if out.isnan().any() or softmax_lse.isnan().any():
        #     breakpoint()
        return out, softmax_lse, S_dmask, rng_state

    @torch.library.register_fake("flash_attn::_flash_attn_varlen_forward_v2")
    def _flash_attn_varlen_forward_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int = -1,
        window_size_right: int = -1,
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        return_softmax: bool = False,
        block_table: Optional[torch.Tensor] = None,
        leftpad_k: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        zero_tensors: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = (maybe_contiguous_v2(x) for x in (q, k, v))
        batch_size = cu_seqlens_q.numel() - 1
        total_q, num_heads, _ = q.shape

        out = torch.empty_like(q)
        softmax_lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device, layout=q.layout)
        p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
        seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
        seqlen_k_rounded = round_multiple(max_seqlen_k, 128)
        if return_softmax:
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                dtype=q.dtype,
                device=q.device,
                layout=q.layout,
            )
        rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
        return out, softmax_lse, p, rng_state

    @torch.library.custom_op(
        "flash_attn::_flash_attn_varlen_backward_v2", mutates_args=("dq", "dk", "dv"), device_types="cuda"
    )
    def _flash_attn_varlen_backward(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: Optional[torch.Tensor],
        dk: Optional[torch.Tensor],
        dv: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        rng_state: Optional[torch.Tensor] = None,
        zero_tensors: bool = False,
    ) -> torch.Tensor:
        # dq, dk, dv are allocated by us so they should already be contiguous
        dout, q, k, v, out = (maybe_contiguous_v2(x) for x in (dout, q, k, v, out))
        (
            dq,
            dk,
            dv,
            softmax_d,
        ) = flash_attn_gpu.varlen_bwd(
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
            alibi_slopes,
            max_seqlen_q.item(),
            max_seqlen_k.item(),
            dropout_p,
            softmax_scale,
            zero_tensors,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            deterministic,
            None,
            rng_state,
        )
        return softmax_d

    @torch.library.register_fake("flash_attn::_flash_attn_varlen_backward_v2")
    def _flash_attn_varlen_backward_fake(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: Optional[torch.Tensor],
        dk: Optional[torch.Tensor],
        dv: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        rng_state: Optional[torch.Tensor] = None,
        zero_tensors: bool = False,
    ) -> torch.Tensor:
        dout, q, k, v, out = (maybe_contiguous_v2(x) for x in (dout, q, k, v, out))
        batch_size = cu_seqlens_q.numel() - 1
        total_q, num_heads, _ = q.shape

        if dq is None:
            dq = torch.empty_like(q)
        if dk is None:
            dk = torch.empty_like(k)
        if dv is None:
            dv = torch.empty_like(v)
        softmax_d = torch.empty((num_heads, total_q + 128 * batch_size), device=q.device, dtype=torch.float32)

        return softmax_d
except ImportError:
    flash_attn_varlen_func_v2 = None  # type: ignore
