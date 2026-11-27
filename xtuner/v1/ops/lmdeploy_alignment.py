# Copyright (c) OpenMMLab. All rights reserved.
"""Serving forwards with native mathematical VJPs and fixed routing ids.

BF16 activation casts use the existing straight-through gradient convention.
The inference kernels do not participate in autograd; backward recomputes the
same equations using differentiable FP32 operations and the saved parameters.
"""

import os

import torch
from torch import Tensor
from torch.nn import functional as F


ALIGN_FORWARD = os.getenv("GLM53_ALIGN_FORWARD", "0").lower() in {"1", "true", "yes", "on"}
_HC_BACKWARD_CHUNK_ROWS = max(1, int(os.getenv("XTUNER_GLM53_MHC_BACKWARD_CHUNK_ROWS", "4096")))


def head_linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    if ALIGN_FORWARD and weight.size(0) > 1:
        from lmdeploy.pytorch.nn.glm53_alignment import aligned_linear

        return aligned_linear(x, weight, bias)
    return F.linear(x, weight, bias)


@torch.library.custom_op("xtuner::lmdeploy_hc_pre", mutates_args=())
def hc_pre(
    x: Tensor, fn: Tensor, scale: Tensor, base: Tensor, norm: Tensor, iters: int, eps: float, norm_eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    from lmdeploy.pytorch.nn.glm53_alignment import hc_pre as serving_pre

    return serving_pre(x, fn, scale, base, norm, iters, eps, norm_eps)


@hc_pre.register_fake
def _hc_pre_fake(x, fn, scale, base, norm, iters, eps, norm_eps):
    shape, h = x.shape[:-2], x.size(-2)
    return (x.new_empty((*shape, x.size(-1))), scale.new_empty((*shape, h)), scale.new_empty((*shape, h, h)))


def _hc_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:5])
    ctx.iters, ctx.eps, ctx.norm_eps = inputs[5:]


def _hc_backward_chunked(
    gy: Tensor, gp: Tensor, gc: Tensor, x: Tensor, fn: Tensor, scale: Tensor,
    base: Tensor, norm: Tensor, iters: int, eps: float, norm_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Recompute row-independent VJPs without full activation-sized FP32 copies."""
    from xtuner.v1.module.decoder_layer.mhc import hc_split_sinkhorn

    h, d = x.shape[-2:]
    rows = x.reshape(-1, h, d)
    gy, gp, gc = gy.reshape(-1, d), gp.reshape(-1, h), gc.reshape(-1, h, h)
    # The flattened destination must alias dx even for a strided input view.
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    dx_rows = dx.reshape(-1, h, d)
    parameters = (fn, scale, base, norm)
    totals = [torch.zeros_like(value, dtype=torch.float32) for value in parameters]
    with torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        fp32_parameters = tuple(value.detach().float().requires_grad_() for value in parameters)
        weight, gain, bias, norm_weight = fp32_parameters
        for start in range(0, rows.size(0), _HC_BACKWARD_CHUNK_ROWS):
            stop = min(start + _HC_BACKWARD_CHUNK_ROWS, rows.size(0))
            local = rows[start:stop].detach().float().requires_grad_()
            flat = local.flatten(-2)
            mixes = F.linear(F.rms_norm(flat, (flat.size(-1),), eps=norm_eps), weight)
            pre, post, comb = hc_split_sinkhorn(mixes, gain, bias, h, iters, eps)
            reduced = (pre.unsqueeze(-1) * local).sum(-2)
            reduced = reduced + (reduced.to(x.dtype).float() - reduced).detach()
            y = F.rms_norm(reduced, (d,), eps=norm_eps) * norm_weight
            grads = torch.autograd.grad(
                (y, post, comb), (local, *fp32_parameters),
                (gy[start:stop].float(), gp[start:stop], gc[start:stop]))
            with torch.no_grad():
                dx_rows[start:stop].copy_(grads[0])
                for total, grad in zip(totals, grads[1:]):
                    total.add_(grad)
            # Free this chunk before allocating the next chunk's FP32 graph.
            del local, flat, mixes, pre, post, comb, reduced, y, grads
    return (dx, *(grad.to(value.dtype) for grad, value in zip(totals, parameters)))


def _hc_backward(ctx, gy, gp, gc):
    saved = ctx.saved_tensors
    grads = _hc_backward_chunked(gy, gp, gc, *saved, ctx.iters, ctx.eps, ctx.norm_eps)
    return (*grads, None, None, None)


hc_pre.register_autograd(_hc_backward, setup_context=_hc_setup)


@torch.library.custom_op("xtuner::lmdeploy_router", mutates_args=())
def router(logits: Tensor, bias: Tensor, forced_ids: Tensor | None) -> tuple[Tensor, Tensor]:
    from lmdeploy.pytorch.kernels.cuda.moe.route_noaux_tc import fused_noaux_tc_routing

    return fused_noaux_tc_routing(
        logits,
        bias,
        num_experts=288,
        n_group=1,
        topk_group=1,
        top_k=8,
        routed_scaling_factor=1.0,
        forced_ids=forced_ids,
    )


@router.register_fake
def _router_fake(logits, bias, forced_ids):
    return logits.new_empty((logits.size(0), 8), dtype=torch.float32), logits.new_empty(
        (logits.size(0), 8), dtype=torch.int64
    )


def _router_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], output[1])


def _router_backward(ctx, grad_weights, grad_ids):
    logits, ids = ctx.saved_tensors
    with torch.enable_grad():
        x = logits.detach().float().requires_grad_()
        weights = x.sigmoid().gather(1, ids)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        (grad,) = torch.autograd.grad(weights, x, grad_weights)
    return grad.to(logits.dtype), None, None


router.register_autograd(_router_backward, setup_context=_router_setup)
