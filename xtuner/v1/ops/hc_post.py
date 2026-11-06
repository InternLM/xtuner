# Copyright (c) OpenMMLab. All rights reserved.
"""Fused mHC post with FP32 coefficients and a single final activation cast.

Accumulate post*x followed by source streams in index order, matching LMDeploy's post expansion. Backward retains
coefficient precision for the same contract.
"""

import torch
from torch import Tensor


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton always present on the V4 GPU path
    _HAS_TRITON = False


__all__ = ["hc_post_fused", "is_available"]


def is_available() -> bool:
    """Whether the fused Triton ``hc_post`` can run on this build.

    Returns:
        bool: ``True`` when Triton imported successfully.
    """
    return _HAS_TRITON


if _HAS_TRITON:

    @triton.jit
    def _hc_post_fwd_kernel(
        x_ptr,
        res_ptr,
        post_ptr,
        comb_ptr,
        out_ptr,
        D,
        H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0)
        db = tl.program_id(1)
        doff = db * BLOCK_D + tl.arange(0, BLOCK_D)
        dm = doff < D
        xt = tl.load(x_ptr + t * D + doff, mask=dm, other=0.0).to(tl.float32)
        for ho in range(H):
            # Match rollout: retain FP32 coefficients and accumulate post*x first,
            # followed by source streams in order, with only the final BF16 cast.
            p = tl.load(post_ptr + t * H + ho).to(tl.float32)
            acc = p * xt
            for hi in range(H):
                c = tl.load(comb_ptr + t * H * H + hi * H + ho).to(tl.float32)
                r = tl.load(res_ptr + t * H * D + hi * D + doff, mask=dm, other=0.0).to(tl.float32)
                acc += c * r
            tl.store(out_ptr + t * H * D + ho * D + doff, acc, mask=dm)

    @triton.jit
    def _hc_post_bwd_dpar_kernel(
        g_ptr,
        post_ptr,
        comb_ptr,
        gx_ptr,
        gres_ptr,
        D,
        H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # D-parallel grads: grad_x and grad_residual (reduce over h_out).
        t = tl.program_id(0)
        db = tl.program_id(1)
        doff = db * BLOCK_D + tl.arange(0, BLOCK_D)
        dm = doff < D
        hidx = tl.arange(0, H)
        g = tl.load(
            g_ptr + t * H * D + hidx[:, None] * D + doff[None, :],
            mask=dm[None, :],
            other=0.0,
        ).to(tl.float32)  # [H_out, BLOCK_D]
        post = tl.load(post_ptr + t * H + hidx).to(tl.float32)  # [H_out]
        gx = tl.sum(post[:, None] * g, axis=0)  # grad_x[d] = sum_ho post[ho] g[ho,d]
        tl.store(gx_ptr + t * D + doff, gx.to(gx_ptr.dtype.element_ty), mask=dm)
        for hi in range(H):
            # grad_residual[hi, d] = sum_ho comb[hi, ho] g[ho, d]
            comb_row = tl.load(comb_ptr + t * H * H + hi * H + hidx).to(tl.float32)  # comb[hi, :]
            gr = tl.sum(comb_row[:, None] * g, axis=0)
            tl.store(gres_ptr + t * H * D + hi * D + doff, gr.to(gres_ptr.dtype.element_ty), mask=dm)

    @triton.jit
    def _hc_post_bwd_dred_kernel(
        g_ptr,
        x_ptr,
        res_ptr,
        gpost_ptr,
        gcomb_ptr,
        D,
        H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # D-reduction grads: grad_post and grad_comb (reduce over the hidden axis).
        # One program per token loops over D in BLOCK_D chunks, accumulating the
        # [H] and [H, H] grads in registers.
        t = tl.program_id(0)
        hidx = tl.arange(0, H)
        gpost = tl.zeros([H], dtype=tl.float32)
        gcomb = tl.zeros([H, H], dtype=tl.float32)  # [h_in, h_out]
        nblk = tl.cdiv(D, BLOCK_D)
        for b in range(nblk):
            doff = b * BLOCK_D + tl.arange(0, BLOCK_D)
            dm = doff < D
            xt = tl.load(x_ptr + t * D + doff, mask=dm, other=0.0).to(tl.float32)  # [BLOCK_D]
            g = tl.load(
                g_ptr + t * H * D + hidx[:, None] * D + doff[None, :],
                mask=dm[None, :],
                other=0.0,
            ).to(tl.float32)  # [H_out, BLOCK_D]
            res = tl.load(
                res_ptr + t * H * D + hidx[:, None] * D + doff[None, :],
                mask=dm[None, :],
                other=0.0,
            ).to(tl.float32)  # [H_in, BLOCK_D]
            gpost += tl.sum(xt[None, :] * g, axis=1)  # grad_post[ho] = sum_d x[d] g[ho,d]
            # grad_comb[hi, ho] = sum_d res[hi, d] g[ho, d]
            gcomb += tl.sum(res[:, None, :] * g[None, :, :], axis=2)
        tl.store(gpost_ptr + t * H + hidx, gpost.to(gpost_ptr.dtype.element_ty))
        tl.store(
            gcomb_ptr + t * H * H + hidx[:, None] * H + hidx[None, :],
            gcomb.to(gcomb_ptr.dtype.element_ty),
        )

    _BLOCK_D = 2048

    @torch.library.custom_op("xtuner::hc_post_fwd", mutates_args=())
    def _hc_post_fwd_op(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        *batch, H, D = residual.shape
        n = 1
        for b in batch:
            n *= b
        xf = x.reshape(n, D).contiguous()
        rf = residual.reshape(n, H, D).contiguous()
        # Preserve coefficient precision in both forward and its VJP.
        pf = post.reshape(n, H).contiguous()
        cf = comb.reshape(n, H, H).contiguous()
        out = torch.empty_like(rf)
        grid = (n, triton.cdiv(D, _BLOCK_D))
        _hc_post_fwd_kernel[grid](xf, rf, pf, cf, out, D, H=H, BLOCK_D=_BLOCK_D)
        return out.reshape(*batch, H, D)

    @_hc_post_fwd_op.register_fake
    def _(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        return torch.empty_like(residual)

    @torch.library.custom_op("xtuner::hc_post_bwd", mutates_args=())
    def _hc_post_bwd_op(
        grad_out: Tensor, x: Tensor, residual: Tensor, post: Tensor, comb: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        *batch, H, D = residual.shape
        n = 1
        for b in batch:
            n *= b
        g = grad_out.reshape(n, H, D).contiguous()
        xf = x.reshape(n, D).contiguous()
        rf = residual.reshape(n, H, D).contiguous()
        pf = post.reshape(n, H).contiguous()
        cf = comb.reshape(n, H, H).contiguous()

        gx = torch.empty_like(xf)
        gres = torch.empty_like(rf)
        _hc_post_bwd_dpar_kernel[(n, triton.cdiv(D, _BLOCK_D))](g, pf, cf, gx, gres, D, H=H, BLOCK_D=_BLOCK_D)

        # grad_post / grad_comb flow back to the fp32 Sinkhorn outputs, so we
        # accumulate and return them in their original dtype.
        gpost = torch.empty((n, H), device=g.device, dtype=post.dtype)
        gcomb = torch.empty((n, H, H), device=g.device, dtype=comb.dtype)
        _hc_post_bwd_dred_kernel[(n,)](g, xf, rf, gpost, gcomb, D, H=H, BLOCK_D=_BLOCK_D)

        return (
            gx.reshape(*batch, D),
            gres.reshape(*batch, H, D),
            gpost.reshape(*batch, H),
            gcomb.reshape(*batch, H, H),
        )

    @_hc_post_bwd_op.register_fake
    def _(
        grad_out: Tensor, x: Tensor, residual: Tensor, post: Tensor, comb: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            torch.empty_like(x),
            torch.empty_like(residual),
            torch.empty_like(post),
            torch.empty_like(comb),
        )

    def _hc_post_setup_context(ctx, inputs, output) -> None:
        x, residual, post, comb = inputs
        ctx.save_for_backward(x, residual, post, comb)

    def _hc_post_backward(ctx, grad_out):
        x, residual, post, comb = ctx.saved_tensors
        gx, gres, gpost, gcomb = _hc_post_bwd_op(grad_out.contiguous(), x, residual, post, comb)
        return gx, gres, gpost, gcomb

    _hc_post_fwd_op.register_autograd(_hc_post_backward, setup_context=_hc_post_setup_context)

    def hc_post_fused(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        """Fused Triton ``hc_post``: ``post * x + comb^T @ residual``.

        Args:
            x (Tensor): Inner-block output, shape ``[..., hidden_size]``.
            residual (Tensor): HC-expanded streams saved before the block,
                shape ``[..., hc_mult, hidden_size]``.
            post (Tensor): Per-stream post weights, shape ``[..., hc_mult]``.
            comb (Tensor): Doubly-stochastic combination matrix, shape
                ``[..., hc_mult, hc_mult]``.

        Returns:
            Tensor: Updated HC-expanded streams, shape
                ``[..., hc_mult, hidden_size]``, in ``residual.dtype``.
        """
        return _hc_post_fwd_op(x, residual, post, comb)

else:  # pragma: no cover

    def hc_post_fused(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        raise RuntimeError("hc_post_fused requires Triton, which is not available in this build.")
