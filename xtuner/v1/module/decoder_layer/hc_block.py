# Copyright (c) OpenMMLab. All rights reserved.
#
# The structural reference for the Hyper-Connections wrapper (``hc_pre`` / ``hc_post``
# semantics, parameter shapes, init policy and per-block forward order) is
# DeepSeek-V4-Flash's ``inference/model.py::Block`` (MIT-licensed):
#   https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py
# We re-implement the same numerical contract in pure PyTorch on top of
# :func:`xtuner.v1.module.decoder_layer.hc_sinkhorn.hc_split_sinkhorn` so XTuner
# can train without depending on TileLang.
"""Hyper-Connections (HC) primitives for DeepSeek-V4-Flash.

The HC machinery keeps ``hc_mult`` copies of the hidden state and replaces the
plain ``x = x + block(norm(x))`` residual with a learned mix:

1. :func:`hc_pre` reduces the ``hc_mult`` streams to one weighted stream that
   an attention or FFN sub-block consumes.
2. :func:`hc_post` re-expands the sub-block output into ``hc_mult`` streams
   using a learned doubly-stochastic combination of the original streams plus
   the sub-block output.

These two functions plus :class:`HCWrapperConfig` and
:func:`_unshard_hc_params` are the public surface — they are consumed by
:class:`xtuner.v1.model.moe.deepseek_v4.V4DecoderLayer`, which inlines the
HC residual-mix pattern around its own attention and FFN sub-blocks.

History note. An earlier version of this file shipped a ``HCDecoderLayer``
class plus an ``HCInnerBlock`` structural protocol, intended as a generic
"any attn+ffn block" wrapper. V4 was the only user and the protocol was too
narrow for DSA's kwargs (it forced a ``set_context`` side-channel on the
inner block). The class was removed during the V4 layer consolidation; the
math is now consumed directly as functions, which is also more compile-
friendly (no nested ``nn.Module`` for dynamo to trace into).
"""

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from xtuner.v1.utils.compile import maybe_compile

from .hc_sinkhorn import hc_split_sinkhorn


class HCWrapperConfig(BaseModel):
    """Configuration for the HC residual-mix pattern.

    Mirrors the three HC-related fields of the DeepSeek-V4-Flash config:
    ``hc_mult``, ``hc_eps``, ``hc_sinkhorn_iters``.

    Args:
        hc_mult (int): Number of hyper-connection streams. ``1`` makes the
            HC math degenerate to a plain pre-norm residual block (used as a
            structural parity anchor in tests).
        hc_eps (float): Stabilizer used inside the Sinkhorn normalization.
        hc_sinkhorn_iters (int): Number of Sinkhorn iterations.
    """

    model_config = ConfigDict(extra="forbid")

    hc_mult: int
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20


@maybe_compile
def hc_pre(
    x: Tensor,
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
    norm_eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reduce ``hc_mult`` streams down to one, returning the reduced state and
    the ``post``/``comb`` weights that the matching :func:`hc_post` call will
    consume.

    Faithful port of ``Block.hc_pre`` in DeepSeek-V4-Flash ``inference/model.py``
    (L673-682): apply an RMS-style rescale to the flattened streams, project to
    ``mixes`` via ``hc_fn``, run :func:`hc_split_sinkhorn`, then take a weighted
    sum over the stream axis with ``pre`` as the weights.

    The HC parameters are expected to arrive as plain :class:`torch.Tensor`
    (not :class:`DTensor`); the enclosing :class:`HCDecoderLayer.forward`
    eagerly materialises the locals via ``_unshard_hc_params`` before calling
    in, so this function stays a clean compile region without intermediate
    DTensor unshard graph-breaks.

    Args:
        x (Tensor): Hidden states, shape ``[B, S, hc_mult, hidden_size]``.
        hc_fn (Tensor): Mixing projection, shape ``[(2 + hc_mult) * hc_mult, hc_mult * hidden_size]``.
        hc_scale (Tensor): Sub-block scales, shape ``[3]``.
        hc_base (Tensor): Per-slot bias, shape ``[(2 + hc_mult) * hc_mult]``.
        hc_mult (int): Number of streams (``H``).
        iters (int): Sinkhorn iterations.
        eps (float): Sinkhorn stabilizer.
        norm_eps (float): RMS-norm stabilizer applied before projecting to ``mixes``.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - ``y`` (``[B, S, hidden_size]``): reduced stream consumed by the inner block.
            - ``post`` (``[B, S, hc_mult]``): post weights used by :func:`hc_post`.
            - ``comb`` (``[B, S, hc_mult, hc_mult]``): combination weights used by :func:`hc_post`.
    """
    shape, dtype = x.size(), x.dtype
    # V4 reference does the RMS rescale + linear mix in fp32 to keep the
    # downstream Sinkhorn iterations stable under bf16. Naive: ``x.float()``
    # allocates a 256 MB transient at pack=4096/D=4096/hc_mult=4 and saves it
    # for the linear's backward, costing ~5 GB cumulatively across 4 layers ×
    # 2 (attn+ffn) × 2 (forward + recompute backward).
    #
    # We keep the *output* in fp32 (Sinkhorn input) but skip materialising the
    # full upcasted activation:
    #   * mean-of-squares reduces with ``dtype=fp32`` accumulator, so the only
    #     allocations are the bf16 squared tensor (transient, half the size)
    #     and a tiny ``[B, S, 1]`` fp32 scalar
    #   * the gate linear runs in bf16 (cuBLAS internally accumulates in fp32
    #     anyway), then the tiny ``[B, S, mix_dim]`` output is upcast to fp32
    #     before Sinkhorn — mix_dim is ``(2 + hc_mult) * hc_mult`` which is 24
    #     for hc_mult=4, so the upcast is <1 MB.
    x_flat = x.flatten(2)
    sq = x_flat * x_flat
    rsqrt = torch.rsqrt(sq.mean(-1, keepdim=True, dtype=torch.float32) + norm_eps)
    mixes = torch.nn.functional.linear(x_flat, hc_fn.to(x_flat.dtype)).float() * rsqrt

    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)

    # ``pre`` is fp32 (Sinkhorn output). Multiplying fp32 × bf16 in PyTorch's
    # TensorIterator path casts elementwise on read, so we don't need to
    # materialise a fp32 copy of ``x_flat`` here — the auto-promoted product
    # is fp32 of the same shape but the bf16 input stays bf16 in memory.
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=-2)
    return y.to(dtype), post, comb


def _unshard_hc_params(
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Eager-mode helper: replace any DTensor-wrapped HC parameter with its
    full local tensor.

    Why this lives outside :func:`hc_pre`: FSDP shards every
    :class:`nn.Parameter` into a :class:`DTensor` and only unshards them under
    the pre-forward hook of the *enclosing* :class:`nn.Module`. The HC params
    live on the wrapper but are consumed by ``hc_pre``, so we materialise the
    local view here (cheap, ~100KB per layer) *before* the compile boundary
    instead of inside the compiled graph (which would graph-break ×3 per HC
    call: one per parameter).

    Args:
        hc_fn (Tensor): Mixing projection (potentially a DTensor).
        hc_scale (Tensor): Sub-block scales (potentially a DTensor).
        hc_base (Tensor): Per-slot bias (potentially a DTensor).

    Returns:
        tuple[Tensor, Tensor, Tensor]: Plain-tensor counterparts. Non-DTensor
        inputs are returned as-is.
    """
    from torch.distributed.tensor import DTensor as _DTensor

    if isinstance(hc_fn, _DTensor):
        hc_fn = hc_fn.full_tensor()
    if isinstance(hc_scale, _DTensor):
        hc_scale = hc_scale.full_tensor()
    if isinstance(hc_base, _DTensor):
        hc_base = hc_base.full_tensor()
    return hc_fn, hc_scale, hc_base


@maybe_compile
def hc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """Expand the single-stream block output back into ``hc_mult`` streams.

    Port of ``Block.hc_post`` in DeepSeek-V4-Flash ``inference/model.py`` (L683-686):
    ``out[..., h, d] = post[..., h] * x[..., d] + sum_{h'} comb[..., h, h'] * residual[..., h', d]``.

    Args:
        x (Tensor): Inner-block output, shape ``[B, S, hidden_size]``.
        residual (Tensor): HC-expanded residual saved before :func:`hc_pre`,
            shape ``[B, S, hc_mult, hidden_size]``.
        post (Tensor): Post weights from :func:`hc_pre`, shape ``[B, S, hc_mult]``.
        comb (Tensor): Combination matrix from :func:`hc_pre`, shape ``[B, S, hc_mult, hc_mult]``.

    Returns:
        Tensor: Updated HC-expanded streams, shape ``[B, S, hc_mult, hidden_size]``,
            cast back to ``x.dtype``.
    """
    # post * x is broadcast across hidden dim; comb mixes across the residual stream axis.
    #
    # Implementation note. The natural form is::
    #
    #   mixed = torch.matmul(comb.to(residual.dtype), residual)
    #
    # but ``matmul`` (and ``einsum`` with the same contraction) lowers to
    # ``aten.bmm`` at the inductor level, and inductor delegates ``bmm`` to
    # cuBLAS via ``extern_kernels.bmm`` rather than codegening a triton kernel
    # — ``epilogue_fusion=True`` only fuses pointwise tails onto the matmul,
    # not the matmul body itself. Here the bmm has inner dim ``K = hc_mult =
    # 4`` which is below Hopper's wgmma tile floor (K=16), so cuBLAS falls
    # back to a CUDA-core path that's bandwidth-bound and slow — at
    # pack=16384 it cost ~3 ms per call × 86 calls/step ≈ 250 ms/step.
    #
    # To force inductor to codegen a fused triton kernel for the mixing, we
    # express it as broadcast-multiply + reduce-sum — both pointwise / reduce
    # primitives, which are inductor's strongest fusion targets. The
    # ``[B, S, H_out, H_in, D]`` intermediate that the unsqueeze + multiply
    # implies in eager mode **never materialises under compile**: inductor
    # fuses the multiply with the trailing ``.sum(dim=-2)`` into a single
    # triton kernel that does the H_in reduction in registers and writes only
    # the ``[B, S, H_out, D]`` output. The cast, the ``post * x`` broadcast
    # and the final add all join that kernel's epilogue, so the whole
    # ``hc_post`` body compiles to one fused kernel.
    #
    # WARNING: this function MUST be in the active compile cfg (see
    # ``_V4_LAYER_TARGETS`` in ``deepseek_v4.py``). Running it eagerly would
    # materialise the 8 GB ``[B, S, H, H, D]`` 5D tensor at pack=16384,
    # hc_mult=4, hidden=4096. The compile cfg covers it by default; the
    # ``hc_mult=1`` degenerate path in ``V4DecoderLayer.forward`` skips
    # ``hc_post`` entirely so unit tests with that setting don't hit this.
    mixed = (
        comb.to(residual.dtype).unsqueeze(-1)   # [B, S, H_out, H_in, 1]
        * residual.unsqueeze(-3)                # [B, S, 1,     H_in, D]
    ).sum(dim=-2)                                # → [B, S, H_out, D]
    expanded = post.unsqueeze(-1) * x.unsqueeze(-2) + mixed
    return expanded.type_as(x)
