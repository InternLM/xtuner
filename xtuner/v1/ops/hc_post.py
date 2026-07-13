# Copyright (c) OpenMMLab. All rights reserved.
"""Fused Triton kernel for the DeepSeek-V4-Flash Hyper-Connections ``hc_post``.

``hc_post`` re-expands a single-stream block output into ``hc_mult`` streams::

    mixed[h_out, d] = sum_{h_in} comb[h_in, h_out] * residual[h_in, d]   # H×H mix
    out[h_out, d]   = post[h_out] * x[d] + mixed[h_out, d]               # rank-1 add

with ``H = hc_mult`` small (4 in the release). The reference PyTorch form
(:func:`xtuner.v1.module.decoder_layer.deepseek_v4.hc_block.hc_post`) deliberately avoids
``torch.matmul`` because the ``K = H = 4`` contraction is below Hopper's wgmma
tile floor and cuBLAS falls back to a slow CUDA-core GEMM. It instead uses a
broadcast-multiply + reduce-sum that ``inductor`` fuses into a single kernel.

That fused kernel is still memory-bound and, worse, re-reads ``residual`` once
per ``h_out`` (4×): each output element ``[h_out, d]`` is its own reduction
thread, so ``residual[:, d]`` is loaded ``H_out`` times and the ``1 GB``
(fp32, pack=16384) residual blows past L2. This kernel instead assigns one
program to a whole token's ``[H_out, BLOCK_D]`` tile and reads ``residual``
exactly once, doing the 4×4 mix in registers. Measured ~21× on the forward,
~7× on fwd+bwd vs the eager reference, and slightly *closer* to the fp32 ground
truth than the reference (more of the accumulation stays in fp32).

Numerics differ from the reference by ~1 bf16 ULP (different but equally valid
reduction order), so the caller gates this path on ``not _HC_HF_PARITY`` and
keeps the eager fp32 path for the bit-exact HF-parity tests.
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
        hidx = tl.arange(0, H)
        xt = tl.load(x_ptr + t * D + doff, mask=dm, other=0.0).to(tl.float32)
        # residual[h, d] for all h, read once → [H, BLOCK_D]
        res = tl.load(
            res_ptr + t * H * D + hidx[:, None] * D + doff[None, :],
            mask=dm[None, :],
            other=0.0,
        ).to(tl.float32)
        for ho in range(H):
            # comb_col = comb[:, ho]; mix reduces over the first (h_in) axis,
            # matching the reference's ``comb.transpose(-1, -2)`` contraction.
            comb_col = tl.load(comb_ptr + t * H * H + hidx * H + ho).to(tl.float32)
            acc = tl.sum(comb_col[:, None] * res, axis=0)
            # Replicate the reference's bf16 rounding boundaries (mixed.to(bf16),
            # bf16 ``post * x``) so the only residual delta vs eager is reduction
            # order, not an extra fp32 carry.
            acc = acc.to(tl.bfloat16).to(tl.float32)
            p = tl.load(post_ptr + t * H + ho).to(tl.float32)
            prod = (p * xt).to(tl.bfloat16).to(tl.float32)
            tl.store(
                out_ptr + t * H * D + ho * D + doff,
                (prod + acc).to(out_ptr.dtype.element_ty),
                mask=dm,
            )

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
        dt = residual.dtype
        xf = x.reshape(n, D).contiguous()
        rf = residual.reshape(n, H, D).contiguous()
        # Round post/comb to the activation dtype, mirroring the reference's
        # ``post.to(residual.dtype)`` / ``comb.to(residual.dtype)``.
        pf = post.to(dt).reshape(n, H).contiguous()
        cf = comb.to(dt).reshape(n, H, H).contiguous()
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
        dt = residual.dtype
        g = grad_out.reshape(n, H, D).contiguous()
        xf = x.reshape(n, D).contiguous()
        rf = residual.reshape(n, H, D).contiguous()
        pf = post.to(dt).reshape(n, H).contiguous()
        cf = comb.to(dt).reshape(n, H, H).contiguous()

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
