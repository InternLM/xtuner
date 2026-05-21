# Copyright (c) OpenMMLab. All rights reserved.
#
# The structural reference for the Hyper-Connections wrapper (``hc_pre`` / ``hc_post``
# semantics, parameter shapes, init policy and per-block forward order) is
# DeepSeek-V4-Flash's ``inference/model.py::Block`` (MIT-licensed):
#   https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py
# We re-implement the same numerical contract in pure PyTorch on top of
# :func:`xtuner.v1.module.decoder_layer.hc_sinkhorn.hc_split_sinkhorn` so XTuner
# can train without depending on TileLang.
"""Hyper-Connections (HC) decoder wrapper for DeepSeek-V4-Flash.

The HC wrapper keeps ``hc_mult`` copies of the hidden state and replaces the
plain ``x = x + block(norm(x))`` residual with a learned mix:

1. ``hc_pre`` reduces the ``hc_mult`` streams to one weighted stream that the
   inner attention or FFN block consumes.
2. ``hc_post`` re-expands the block output into ``hc_mult`` streams using a
   learned doubly-stochastic combination of the original streams plus the
   block output.

This wrapper is *layout-only*: it does not touch ``input_layernorm`` /
``post_attention_layernorm`` itself. The contract with ``inner`` is therefore
narrow: ``inner`` must expose ``attn_block(x) -> Tensor`` and
``ffn_block(x) -> Tensor`` callables that take and return ``[B, S, hidden_size]``
and that internally apply the norm + sub-block. PR9 (DeepSeekV4 glue) is
responsible for adapting the real :class:`MoEDecoderLayer` to this contract; in
this PR the wrapper is exercised with a small mock block.
"""

from typing import Any, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from xtuner.v1.utils.compile import maybe_compile

from .hc_sinkhorn import hc_split_sinkhorn


@runtime_checkable
class HCInnerBlock(Protocol):
    """Structural contract that :class:`HCDecoderLayer` requires of its inner
    block."""

    def attn_block(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor: ...

    def ffn_block(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor: ...


class HCWrapperConfig(BaseModel):
    """Configuration for :class:`HCDecoderLayer`.

    Mirrors the three HC-related fields of the DeepSeek-V4-Flash config:
    ``hc_mult``, ``hc_eps``, ``hc_sinkhorn_iters``.

    Args:
        hc_mult (int): Number of hyper-connection streams. ``1`` makes the wrapper
            degenerate to a plain pre-norm residual block.
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
    # The mathematically equivalent fused form skips a `[B, S, hc_mult, hc_mult, D]`
    # intermediate (which dominated peak memory at hc_post — see memory profile at
    # step-0/rank1_memory_snapshot.pickle) by writing the second term as a matmul:
    #   sum_j comb[..., i, j] * residual[..., j, d] == (comb @ residual)[..., i, d]
    #
    # Precision: run the matmul in bf16 with cuBLAS's fp32 internal accumulator
    # (same numerical contract as the original element-wise fp32 auto-promote,
    # within bf16 output quantisation). ``comb`` is the small ``[B,S,hc_mult,hc_mult]``
    # fp32 Sinkhorn output (~1 MB at pack=4096); we cast it down to bf16 so the
    # matmul stays bf16×bf16 and avoid materialising the 256 MB fp32 copy of
    # ``residual`` that ``residual.to(comb.dtype)`` would otherwise create (and
    # save for matmul backward). Across 4 layers × 2 (attn+ffn) × 2 (forward +
    # recompute), this is ~4 GB of peak memory savings at pack=4096.
    expanded = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb.to(residual.dtype), residual)
    return expanded.type_as(x)


class HCDecoderLayer(nn.Module):
    """Hyper-Connections wrapper around an attention + FFN inner block.

    The wrapper owns the HC parameters (``hc_attn_*`` and ``hc_ffn_*``) and the
    HC-expanded forward pattern; the ``inner`` module remains responsible for
    its own norms and sub-block math. The contract with ``inner`` is:

    - ``inner.attn_block(x, *args, **kwargs) -> Tensor``: pre-norm + attention
      sub-block, ``[B, S, hidden_size]`` in/out.
    - ``inner.ffn_block(x, *args, **kwargs) -> Tensor``: pre-norm + FFN sub-block,
      ``[B, S, hidden_size]`` in/out.

    PR9 (DeepSeekV4 glue) will adapt :class:`MoEDecoderLayer` to expose these
    callables. ``hc_mult == 1`` short-circuits to a plain pre-norm residual
    block, which makes the wrapper structurally compatible with non-HC models
    and gives a clean degenerate-equivalence anchor for unit tests.

    Args:
        inner (nn.Module): Module exposing ``attn_block`` and ``ffn_block`` as
            described above.
        hc_cfg (HCWrapperConfig): HC hyper-parameters.
        hidden_size (int): Hidden size ``D`` of a single stream.
    """

    def __init__(self, inner: nn.Module, hc_cfg: HCWrapperConfig, hidden_size: int):
        super().__init__()
        self.inner = inner
        self.hc_mult = hc_cfg.hc_mult
        self.hc_eps = hc_cfg.hc_eps
        self.hc_sinkhorn_iters = hc_cfg.hc_sinkhorn_iters
        self.hidden_size = hidden_size

        mix_dim = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * hidden_size

        # V4 stores HC parameters in fp32 even when the rest of the model is bf16
        # (reference: model.py:Block.__init__ uses `with set_dtype(torch.float32)`),
        # because Sinkhorn over 20 iterations is bf16-unstable. We match that.
        fp32 = torch.float32
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix_dim, hc_dim, dtype=fp32))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_dim, dtype=fp32))
        self.hc_attn_scale = nn.Parameter(torch.zeros(3, dtype=fp32))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_dim, hc_dim, dtype=fp32))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_dim, dtype=fp32))
        self.hc_ffn_scale = nn.Parameter(torch.zeros(3, dtype=fp32))

        # Degenerate-safe init: scale[0]=1 (pre) keeps the pre-weight derivative non-zero
        # so training can move it off the all-zero attractor; scale[1]=scale[2]=0 makes
        # post and comb start from constant uniform values rather than random softmax noise.
        with torch.no_grad():
            self.hc_attn_scale[0] = 1.0
            self.hc_ffn_scale[0] = 1.0

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Run one attention-then-FFN HC-wrapped pass.

        Args:
            x (Tensor): HC-expanded hidden states, shape ``[B, S, hc_mult, hidden_size]``.
            *args: Extra positional arguments forwarded to both ``inner.attn_block`` and ``inner.ffn_block``.
            **kwargs: Extra keyword arguments forwarded the same way.

        Returns:
            Tensor: HC-expanded hidden states, shape ``[B, S, hc_mult, hidden_size]``.
        """
        if self.hc_mult == 1:
            # Degenerate path: H=1 carries no mixing information. Bypass the HC math
            # entirely and apply the plain pre-norm residual, matching how non-HC
            # decoder blocks behave. This is the structural anchor used by
            # `test_hc_mult_1_equals_plain_residual`.
            return self._plain_residual_forward(x, *args, **kwargs)

        # nn.Module.__getattr__ erases the `attn_block`/`ffn_block` attributes for the
        # type checker; cast to the structural protocol that documents the contract.
        inner = cast(HCInnerBlock, self.inner)

        # Hoist DTensor.full_tensor() out of the compile region: doing it here
        # (eager) lets `hc_pre` stay a single contiguous compiled graph instead
        # of breaking three times mid-trace for each HC param under FSDP/EP.
        attn_fn, attn_scale, attn_base = _unshard_hc_params(
            self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        ffn_fn, ffn_scale, ffn_base = _unshard_hc_params(
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )

        residual = x
        x_reduced, post_a, comb_a = hc_pre(
            x,
            attn_fn,
            attn_scale,
            attn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        attn_out = inner.attn_block(x_reduced, *args, **kwargs)
        x = hc_post(attn_out, residual, post_a, comb_a)

        residual = x
        x_reduced, post_f, comb_f = hc_pre(
            x,
            ffn_fn,
            ffn_scale,
            ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        ffn_out = inner.ffn_block(x_reduced, *args, **kwargs)
        x = hc_post(ffn_out, residual, post_f, comb_f)
        return x

    def _plain_residual_forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        inner = cast(HCInnerBlock, self.inner)
        # Squeeze the singleton hc axis so the inner sub-blocks see a clean [B, S, D] tensor.
        x_single = x.squeeze(-2)
        x_single = x_single + inner.attn_block(x_single, *args, **kwargs)
        x_single = x_single + inner.ffn_block(x_single, *args, **kwargs)
        return x_single.unsqueeze(-2)
