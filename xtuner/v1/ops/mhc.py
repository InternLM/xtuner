# Copyright (c) OpenMMLab. All rights reserved.
#
# DeepSeek TileKernels mhc (Manifold HyperConnection / V4 HC) backend.
#
# Upstream: https://github.com/deepseek-ai/TileKernels (Apache-2.0).
#
# Each ``mhc_*`` public function wraps one or more TileLang JIT kernels behind a
# ``torch.library.custom_op`` so the call participates in the PyTorch autograd
# graph: ``setup_context`` saves the minimal set of inputs the kernel's matching
# ``_bwd`` kernel needs, and ``register_autograd`` drives the backward pass through
# the upstream gradient kernel.
#
# These wrappers are imported lazily by their callers (only when
# ``XTUNER_USE_MHC_KERNELS=1`` is set) so installations without TileKernels keep
# working out of the box. The kernels JIT-compile on first call (~1–3s each) and
# are cached for the lifetime of the process via ``functools.lru_cache``.
"""Custom-op wrappers for the DeepSeek TileKernels ``mhc`` backend."""

from functools import lru_cache

import torch
from torch import Tensor


__all__ = [
    "mhc_head_compute_mix",
    "mhc_post",
    "mhc_expand",
    "mhc_sinkhorn",
    "mhc_pre_split_mixes",
    "mhc_pre_apply_mix",
    "is_available",
]


def is_available() -> bool:
    """``True`` if ``tile-kernels`` is importable.

    Cheap to call repeatedly.
    """
    try:
        import tile_kernels.mhc  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Kernel JIT caches
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _head_compute_mix_fwd_kernel(hc_mult: int, hc_eps: float, token_block_size: int):
    from tile_kernels.mhc.head_compute_mix_kernel import _mhc_head_compute_mix_fwd

    return _mhc_head_compute_mix_fwd(mhc_mult=hc_mult, mhc_pre_eps=hc_eps, token_block_size=token_block_size)


@lru_cache(maxsize=None)
def _head_compute_mix_bwd_kernel(hc_mult: int, token_block_size: int, num_sms: int):
    from tile_kernels.mhc.head_compute_mix_kernel import _mhc_head_compute_mix_bwd

    return _mhc_head_compute_mix_bwd(mhc_mult=hc_mult, token_block_size=token_block_size, num_sms=num_sms)


@lru_cache(maxsize=None)
def _post_fwd_kernel(hc_mult: int, hidden: int):
    from tile_kernels.mhc.post_kernel import _mhc_post_fwd

    return _mhc_post_fwd(mhc=hc_mult, hidden=hidden)


@lru_cache(maxsize=None)
def _post_bwd_kernel(hc_mult: int, hidden: int):
    from tile_kernels.mhc.post_kernel import _mhc_post_bwd

    return _mhc_post_bwd(mhc=hc_mult, hidden=hidden)


@lru_cache(maxsize=None)
def _expand_fwd_kernel(hidden: int, hc_mult: int):
    from tile_kernels.mhc.expand_kernel import expand_to_mhc_fwd_tl

    return expand_to_mhc_fwd_tl(hidden=hidden, mhc_mult=hc_mult)


@lru_cache(maxsize=None)
def _expand_bwd_kernel(hidden: int, hc_mult: int):
    from tile_kernels.mhc.expand_kernel import expand_to_mhc_bwd_tl

    return expand_to_mhc_bwd_tl(hidden=hidden, mhc_mult=hc_mult)


@lru_cache(maxsize=None)
def _sinkhorn_fwd_kernel(hc_mult: int, token_block_size: int, repeat: int, eps: float):
    # ``hidden_size`` is the upstream parameter name but the actual semantics is
    # ``mhc_mult`` (the ``comb`` matrix is ``[hc_mult, hc_mult]``).
    from tile_kernels.mhc.sinkhorn_kernel import _mhc_sinkhorn_fwd

    return _mhc_sinkhorn_fwd(hidden_size=hc_mult, token_block_size=token_block_size, repeat=repeat, eps=eps)


@lru_cache(maxsize=None)
def _sinkhorn_bwd_kernel(hc_mult: int, token_block_size: int, repeat: int, eps: float):
    from tile_kernels.mhc.sinkhorn_kernel import _mhc_sinkhorn_bwd

    return _mhc_sinkhorn_bwd(hidden_size=hc_mult, token_block_size=token_block_size, repeat=repeat, eps=eps)


@lru_cache(maxsize=None)
def _pre_split_mixes_fwd_kernel(hc_mult: int, post_mult_value: float, pre_eps: float, token_block_size: int):
    from tile_kernels.mhc.pre_split_mixes_kernel import _mhc_pre_split_mixes_fwd

    return _mhc_pre_split_mixes_fwd(
        mhc_mult=hc_mult,
        mhc_post_mult_value=post_mult_value,
        mhc_pre_eps=pre_eps,
        token_block_size=token_block_size,
    )


@lru_cache(maxsize=None)
def _pre_split_mixes_bwd_kernel(hc_mult: int, post_mult_value: float, token_block_size: int, num_sms: int):
    from tile_kernels.mhc.pre_split_mixes_kernel import _mhc_pre_split_mixes_bwd

    return _mhc_pre_split_mixes_bwd(
        mhc_mult=hc_mult,
        mhc_post_mult_value=post_mult_value,
        token_block_size=token_block_size,
        num_sms=num_sms,
    )


@lru_cache(maxsize=None)
def _pre_apply_mix_fwd_kernel(hc_mult: int, hidden: int):
    from tile_kernels.mhc.pre_apply_mix_kernel import _mhc_pre_apply_mix_fwd

    return _mhc_pre_apply_mix_fwd(mhc_mult=hc_mult, hidden=hidden)


@lru_cache(maxsize=None)
def _pre_apply_mix_bwd_kernel(hc_mult: int, hidden: int):
    from tile_kernels.mhc.pre_apply_mix_kernel import _mhc_pre_apply_mix_bwd

    return _mhc_pre_apply_mix_bwd(mhc_mult=hc_mult, hidden=hidden)


# ---------------------------------------------------------------------------
# head_compute_mix: output = sigmoid(input * scale + base) + eps
# ---------------------------------------------------------------------------
# ``hc_eps`` is a kernel-compile constant — we pass it via the op name suffix
# instead of as a tensor arg, but in practice V4 only ever uses one ``hc_eps``
# value (1e-4) per training run so a single registered op is fine. To support
# multiple eps values we would need to register one op per eps. Here we encode
# eps as a function parameter that propagates into the JIT cache key, and the
# custom_op closes over the value at first registration.

_TOKEN_BLOCK_SIZE = 64  # parity-validated default; small enough for V4 packings.


@torch.library.custom_op("xtuner::mhc_head_compute_mix_fwd", mutates_args=())
def _head_compute_mix_fwd_op(input_mix: Tensor, scale: Tensor, base: Tensor, hc_eps: float) -> Tensor:
    # ``input_mix`` arrives as ``[..., hc_mult]`` fp32. Flatten the leading dims.
    hc_mult = input_mix.shape[-1]
    flat = input_mix.contiguous().view(-1, hc_mult)
    out = torch.empty_like(flat)
    kernel = _head_compute_mix_fwd_kernel(hc_mult=hc_mult, hc_eps=hc_eps, token_block_size=_TOKEN_BLOCK_SIZE)
    kernel(flat, scale.contiguous().view(1), base.contiguous(), out)
    return out.view(input_mix.shape)


@_head_compute_mix_fwd_op.register_fake
def _(input_mix: Tensor, scale: Tensor, base: Tensor, hc_eps: float) -> Tensor:
    return torch.empty_like(input_mix)


@torch.library.custom_op("xtuner::mhc_head_compute_mix_bwd", mutates_args=())
def _head_compute_mix_bwd_op(
    grad_out: Tensor, input_mix: Tensor, scale: Tensor, base: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    hc_mult = input_mix.shape[-1]
    flat_grad = grad_out.contiguous().view(-1, hc_mult)
    flat_input = input_mix.contiguous().view(-1, hc_mult)
    num_sms = torch.cuda.get_device_properties(input_mix.device).multi_processor_count

    input_grad = torch.empty_like(flat_input)
    scale_grad_partial = torch.empty((num_sms, 1), dtype=torch.float32, device=input_mix.device)
    base_grad_partial = torch.empty((num_sms, hc_mult), dtype=torch.float32, device=input_mix.device)
    kernel = _head_compute_mix_bwd_kernel(hc_mult=hc_mult, token_block_size=_TOKEN_BLOCK_SIZE, num_sms=num_sms)
    kernel(
        flat_grad,
        flat_input,
        scale.contiguous().view(1),
        base.contiguous(),
        input_grad,
        scale_grad_partial,
        base_grad_partial,
    )
    # The kernel writes per-SM partials; sum them down to the parameter shapes.
    return (
        input_grad.view(input_mix.shape),
        scale_grad_partial.sum(dim=0).view_as(scale),
        base_grad_partial.sum(dim=0).view_as(base),
    )


@_head_compute_mix_bwd_op.register_fake
def _(grad_out: Tensor, input_mix: Tensor, scale: Tensor, base: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return torch.empty_like(input_mix), torch.empty_like(scale), torch.empty_like(base)


def _head_compute_mix_setup_context(ctx, inputs, output) -> None:
    input_mix, scale, base, _hc_eps = inputs
    ctx.save_for_backward(input_mix, scale, base)


def _head_compute_mix_backward(ctx, grad_out):
    input_mix, scale, base = ctx.saved_tensors
    grad_input, grad_scale, grad_base = _head_compute_mix_bwd_op(grad_out.contiguous(), input_mix, scale, base)
    # The 4th input is ``hc_eps`` (float, not differentiable).
    return grad_input, grad_scale, grad_base, None


_head_compute_mix_fwd_op.register_autograd(_head_compute_mix_backward, setup_context=_head_compute_mix_setup_context)


def mhc_head_compute_mix(input_mix: Tensor, scale: Tensor, base: Tensor, hc_eps: float) -> Tensor:
    """``out = sigmoid(input_mix * scale + base) + hc_eps``.

    Args:
        input_mix (Tensor): Pre-activation mix, shape ``[..., hc_mult]`` fp32.
        scale (Tensor): Scalar multiplier, shape ``[1]`` fp32.
        base (Tensor): Per-stream bias, shape ``[hc_mult]`` fp32.
        hc_eps (float): Stabilizer added to the sigmoid output (compile-time constant).

    Returns:
        Tensor: Same shape and dtype as ``input_mix``.
    """
    return _head_compute_mix_fwd_op(input_mix, scale, base, hc_eps)


# ---------------------------------------------------------------------------
# post_kernel: out = post * x + comb^T @ residual
# ---------------------------------------------------------------------------


@torch.library.custom_op("xtuner::mhc_post_fwd", mutates_args=())
def _post_fwd_op(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    *batch, hc_mult, hidden = residual.shape
    n = 1
    for b in batch:
        n *= b
    kernel = _post_fwd_kernel(hc_mult=hc_mult, hidden=hidden)
    out = torch.empty(n, hc_mult, hidden, dtype=residual.dtype, device=residual.device)
    kernel(
        comb.contiguous().view(n, hc_mult, hc_mult),
        residual.contiguous().view(n, hc_mult, hidden),
        post.contiguous().view(n, hc_mult),
        x.contiguous().view(n, hidden),
        out,
    )
    return out.view(*batch, hc_mult, hidden)


@_post_fwd_op.register_fake
def _(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    return torch.empty_like(residual)


@torch.library.custom_op("xtuner::mhc_post_bwd", mutates_args=())
def _post_bwd_op(
    grad_out: Tensor, x: Tensor, residual: Tensor, post: Tensor, comb: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    *batch, hc_mult, hidden = residual.shape
    n = 1
    for b in batch:
        n *= b
    kernel = _post_bwd_kernel(hc_mult=hc_mult, hidden=hidden)
    # ``_mhc_post_bwd`` was JIT-built with ``out_idx=[5, 6, 7, 8]``: the kernel
    # allocates and returns its four gradient outputs as positional results.
    da, db, dc, dd = kernel(
        grad_out.contiguous().view(n, hc_mult, hidden),
        comb.contiguous().view(n, hc_mult, hc_mult),
        residual.contiguous().view(n, hc_mult, hidden),
        post.contiguous().view(n, hc_mult),
        x.contiguous().view(n, hidden),
    )
    return (
        dd.view(*batch, hidden),
        db.view(*batch, hc_mult, hidden),
        dc.view(*batch, hc_mult),
        da.view(*batch, hc_mult, hc_mult),
    )


@_post_bwd_op.register_fake
def _(
    grad_out: Tensor, x: Tensor, residual: Tensor, post: Tensor, comb: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(x),
        torch.empty_like(residual),
        torch.empty_like(post),
        torch.empty_like(comb),
    )


def _post_setup_context(ctx, inputs, output) -> None:
    x, residual, post, comb = inputs
    ctx.save_for_backward(x, residual, post, comb)


def _post_backward(ctx, grad_out):
    x, residual, post, comb = ctx.saved_tensors
    return _post_bwd_op(grad_out.contiguous(), x, residual, post, comb)


_post_fwd_op.register_autograd(_post_backward, setup_context=_post_setup_context)


def mhc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """V4 ``hc_post``: ``out[..., h, d] = post[..., h] * x[..., d] + sum_{h'}
    comb[..., h', h] * residual[..., h', d]``.

    Args:
        x (Tensor): Block output, shape ``[..., hidden]`` bf16.
        residual (Tensor): HC-expanded residual, shape ``[..., hc_mult, hidden]`` bf16.
        post (Tensor): Per-stream weight, shape ``[..., hc_mult]`` fp32.
        comb (Tensor): Doubly-stochastic mix matrix, shape ``[..., hc_mult, hc_mult]`` fp32.

    Returns:
        Tensor: Updated HC streams, shape ``[..., hc_mult, hidden]`` bf16.
    """
    return _post_fwd_op(x, residual, post, comb)


# ---------------------------------------------------------------------------
# expand_kernel: out[n, h, d] = in[n, d]  (broadcast copy)
# Backward: in_grad[n, d] = sum_h out_grad[n, h, d]
# ---------------------------------------------------------------------------


@torch.library.custom_op("xtuner::mhc_expand_fwd", mutates_args=())
def _expand_fwd_op(hidden_states: Tensor, hc_mult: int) -> Tensor:
    *batch, hidden = hidden_states.shape
    n = 1
    for b in batch:
        n *= b
    kernel = _expand_fwd_kernel(hidden=hidden, hc_mult=hc_mult)
    out = torch.empty(n, hc_mult, hidden, dtype=hidden_states.dtype, device=hidden_states.device)
    kernel(hidden_states.contiguous().view(n, hidden), out)
    return out.view(*batch, hc_mult, hidden)


@_expand_fwd_op.register_fake
def _(hidden_states: Tensor, hc_mult: int) -> Tensor:
    *batch, hidden = hidden_states.shape
    return hidden_states.new_empty(*batch, hc_mult, hidden)


@torch.library.custom_op("xtuner::mhc_expand_bwd", mutates_args=())
def _expand_bwd_op(grad_out: Tensor, hidden_states_shape: list[int], hc_mult: int) -> Tensor:
    *batch, _, hidden = grad_out.shape
    n = 1
    for b in batch:
        n *= b
    kernel = _expand_bwd_kernel(hidden=hidden, hc_mult=hc_mult)
    x_grad = torch.empty(n, hidden, dtype=grad_out.dtype, device=grad_out.device)
    kernel(grad_out.contiguous().view(n, hc_mult, hidden), x_grad)
    return x_grad.view(*hidden_states_shape)


@_expand_bwd_op.register_fake
def _(grad_out: Tensor, hidden_states_shape: list[int], hc_mult: int) -> Tensor:
    return grad_out.new_empty(*hidden_states_shape)


def _expand_setup_context(ctx, inputs, output) -> None:
    hidden_states, hc_mult = inputs
    ctx.hidden_states_shape = list(hidden_states.shape)
    ctx.hc_mult = hc_mult


def _expand_backward(ctx, grad_out):
    grad_input = _expand_bwd_op(grad_out.contiguous(), ctx.hidden_states_shape, ctx.hc_mult)
    return grad_input, None


_expand_fwd_op.register_autograd(_expand_backward, setup_context=_expand_setup_context)


def mhc_expand(hidden_states: Tensor, hc_mult: int) -> Tensor:
    """Broadcast ``[..., hidden] → [..., hc_mult, hidden]`` for HC residual
    streams.

    Args:
        hidden_states (Tensor): Input, shape ``[..., hidden]`` bf16.
        hc_mult (int): Number of HC streams (compile-time constant).

    Returns:
        Tensor: Broadcasted, shape ``[..., hc_mult, hidden]`` bf16.
    """
    return _expand_fwd_op(hidden_states, hc_mult)


# ---------------------------------------------------------------------------
# sinkhorn: doubly-stochastic projection of a [hc_mult, hc_mult] matrix
#
# Forward pseudo-code (eps stabilized at each step):
#   x = softmax(x, dim=-1) + eps
#   x = x / (x.sum(-2, keepdim=True) + eps)
#   for _ in range(repeat - 1):
#       x = x / (x.sum(-1, keepdim=True) + eps)
#       x = x / (x.sum(-2, keepdim=True) + eps)
#
# Backward recomputes the forward in fp32 and walks it in reverse — we only
# need to save the raw input.
# ---------------------------------------------------------------------------

_SINKHORN_TOKEN_BLOCK_SIZE = 32  # small block keeps shared-memory under tilelang's allocator budget.


@torch.library.custom_op("xtuner::mhc_sinkhorn_fwd", mutates_args=())
def _sinkhorn_fwd_op(x: Tensor, repeat: int, eps: float) -> Tensor:
    *batch, hc_mult, hc_mult2 = x.shape
    assert hc_mult == hc_mult2, f"sinkhorn expects a square inner matrix; got {hc_mult}×{hc_mult2}"
    n = 1
    for b in batch:
        n *= b
    flat = x.contiguous().view(n, hc_mult, hc_mult)
    out = torch.empty_like(flat)
    kernel = _sinkhorn_fwd_kernel(hc_mult, _SINKHORN_TOKEN_BLOCK_SIZE, repeat, eps)
    kernel(flat, out)
    return out.view(*batch, hc_mult, hc_mult)


@_sinkhorn_fwd_op.register_fake
def _(x: Tensor, repeat: int, eps: float) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("xtuner::mhc_sinkhorn_bwd", mutates_args=())
def _sinkhorn_bwd_op(grad_out: Tensor, x: Tensor, repeat: int, eps: float) -> Tensor:
    *batch, hc_mult, _ = x.shape
    n = 1
    for b in batch:
        n *= b
    flat_grad = grad_out.contiguous().view(n, hc_mult, hc_mult)
    flat_x = x.contiguous().view(n, hc_mult, hc_mult)
    grad_in = torch.empty_like(flat_x)
    kernel = _sinkhorn_bwd_kernel(hc_mult, _SINKHORN_TOKEN_BLOCK_SIZE, repeat, eps)
    kernel(flat_grad, flat_x, grad_in)
    return grad_in.view(*batch, hc_mult, hc_mult)


@_sinkhorn_bwd_op.register_fake
def _(grad_out: Tensor, x: Tensor, repeat: int, eps: float) -> Tensor:
    return torch.empty_like(x)


def _sinkhorn_setup_context(ctx, inputs, output) -> None:
    x, repeat, eps = inputs
    ctx.save_for_backward(x)
    ctx.repeat = repeat
    ctx.eps = eps


def _sinkhorn_backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    grad_in = _sinkhorn_bwd_op(grad_out.contiguous(), x, ctx.repeat, ctx.eps)
    return grad_in, None, None


_sinkhorn_fwd_op.register_autograd(_sinkhorn_backward, setup_context=_sinkhorn_setup_context)


# ---------------------------------------------------------------------------
# pre_split_mixes: mixes [N, mhc_mult * (2 + mhc_mult)] →
#   pre_layer_mix  = sigmoid(mixes[:, :hc]                 * scale[0] + base[:hc])  + pre_eps
#   post_layer_mix = sigmoid(mixes[:, hc:2hc]              * scale[1] + base[hc:2hc]) * post_mult_value
#   comb_res_mix   = mixes[:, 2hc:] * scale[2] + base[2hc:]                   (no nonlinearity; feeds sinkhorn)
# Backward saves (mixes, post_layer_mix, scale, base); base/scale grads are per-SM
# partials that we sum down to the parameter shapes.
# ---------------------------------------------------------------------------

_PRE_SPLIT_TOKEN_BLOCK_SIZE = 64


@torch.library.custom_op("xtuner::mhc_pre_split_mixes_fwd", mutates_args=())
def _pre_split_mixes_fwd_op(
    mixes: Tensor, scale: Tensor, base: Tensor, post_mult_value: float, pre_eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    *batch, mhc_mult3 = mixes.shape
    n = 1
    for b in batch:
        n *= b
    # mhc_mult3 = hc * (2 + hc). Solve for hc via the positive quadratic root.
    # ``mhc_mult3 = hc * (2 + hc)`` → ``hc = sqrt(1 + mhc_mult3) - 1``.
    hc_mult = int(round((1 + mhc_mult3) ** 0.5 - 1))
    # Sanity check: the formula above gives hc_mult such that hc * (2 + hc) = mhc_mult3.
    assert hc_mult * (2 + hc_mult) == mhc_mult3, f"unexpected mhc_mult3={mhc_mult3}; hc={hc_mult}"

    mixes_flat = mixes.contiguous().view(n, mhc_mult3)
    pre_out = torch.empty(n, hc_mult, dtype=mixes.dtype, device=mixes.device)
    post_out = torch.empty(n, hc_mult, dtype=mixes.dtype, device=mixes.device)
    comb_out = torch.empty(n, hc_mult * hc_mult, dtype=mixes.dtype, device=mixes.device)

    kernel = _pre_split_mixes_fwd_kernel(hc_mult, post_mult_value, pre_eps, _PRE_SPLIT_TOKEN_BLOCK_SIZE)
    kernel(mixes_flat, scale.contiguous().view(3), base.contiguous(), pre_out, post_out, comb_out)
    return (
        pre_out.view(*batch, hc_mult),
        post_out.view(*batch, hc_mult),
        comb_out.view(*batch, hc_mult, hc_mult),
    )


@_pre_split_mixes_fwd_op.register_fake
def _(mixes: Tensor, scale: Tensor, base: Tensor, post_mult_value: float, pre_eps: float):
    *batch, mhc_mult3 = mixes.shape
    # ``mhc_mult3 = hc * (2 + hc)`` → ``hc = sqrt(1 + mhc_mult3) - 1``.
    hc_mult = int(round((1 + mhc_mult3) ** 0.5 - 1))
    return (
        mixes.new_empty(*batch, hc_mult),
        mixes.new_empty(*batch, hc_mult),
        mixes.new_empty(*batch, hc_mult, hc_mult),
    )


@torch.library.custom_op("xtuner::mhc_pre_split_mixes_bwd", mutates_args=())
def _pre_split_mixes_bwd_op(
    pre_grad: Tensor,
    post_grad: Tensor,
    comb_grad: Tensor,
    mixes: Tensor,
    post_out: Tensor,
    scale: Tensor,
    base: Tensor,
    post_mult_value: float,
) -> tuple[Tensor, Tensor, Tensor]:
    *batch, mhc_mult3 = mixes.shape
    n = 1
    for b in batch:
        n *= b
    # ``mhc_mult3 = hc * (2 + hc)`` → ``hc = sqrt(1 + mhc_mult3) - 1``.
    hc_mult = int(round((1 + mhc_mult3) ** 0.5 - 1))
    num_sms = torch.cuda.get_device_properties(mixes.device).multi_processor_count

    mixes_flat = mixes.contiguous().view(n, mhc_mult3)
    post_flat = post_out.contiguous().view(n, hc_mult)
    pre_grad_flat = pre_grad.contiguous().view(n, hc_mult)
    post_grad_flat = post_grad.contiguous().view(n, hc_mult)
    comb_grad_flat = comb_grad.contiguous().view(n, hc_mult * hc_mult)

    mixes_grad = torch.empty_like(mixes_flat)
    scale_grad_partial = torch.empty((num_sms, 3), dtype=mixes.dtype, device=mixes.device)
    base_grad_partial = torch.empty((num_sms, mhc_mult3), dtype=mixes.dtype, device=mixes.device)

    kernel = _pre_split_mixes_bwd_kernel(hc_mult, post_mult_value, _PRE_SPLIT_TOKEN_BLOCK_SIZE, num_sms)
    kernel(
        pre_grad_flat,
        post_grad_flat,
        comb_grad_flat,
        mixes_flat,
        post_flat,
        scale.contiguous().view(3),
        base.contiguous(),
        mixes_grad,
        scale_grad_partial,
        base_grad_partial,
    )
    return (
        mixes_grad.view(mixes.shape),
        scale_grad_partial.sum(dim=0).view_as(scale),
        base_grad_partial.sum(dim=0).view_as(base),
    )


@_pre_split_mixes_bwd_op.register_fake
def _(pre_grad, post_grad, comb_grad, mixes, post_out, scale, base, post_mult_value):
    return torch.empty_like(mixes), torch.empty_like(scale), torch.empty_like(base)


def _pre_split_mixes_setup_context(ctx, inputs, output) -> None:
    mixes, scale, base, post_mult_value, _pre_eps = inputs
    _pre_out, post_out, _comb_out = output
    ctx.save_for_backward(mixes, post_out, scale, base)
    ctx.post_mult_value = post_mult_value


def _pre_split_mixes_backward(ctx, pre_grad, post_grad, comb_grad):
    mixes, post_out, scale, base = ctx.saved_tensors
    gm, gs, gb = _pre_split_mixes_bwd_op(
        pre_grad.contiguous(),
        post_grad.contiguous(),
        comb_grad.contiguous(),
        mixes,
        post_out,
        scale,
        base,
        ctx.post_mult_value,
    )
    # 4th / 5th inputs are non-differentiable Python floats.
    return gm, gs, gb, None, None


_pre_split_mixes_fwd_op.register_autograd(_pre_split_mixes_backward, setup_context=_pre_split_mixes_setup_context)


def mhc_pre_split_mixes(
    mixes: Tensor, scale: Tensor, base: Tensor, post_mult_value: float, pre_eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Split fused ``mixes`` projection into ``(pre, post, comb_logits)``.

    Args:
        mixes (Tensor): Output of ``F.linear(flat_normed, hc_fn)``, shape
            ``[..., hc_mult * (2 + hc_mult)]`` fp32.
        scale (Tensor): Sub-block multipliers, shape ``[3]`` fp32.
        base (Tensor): Per-slot bias, shape ``[hc_mult * (2 + hc_mult)]`` fp32.
        post_mult_value (float): Scalar applied to ``sigmoid(...)`` of the post block (typically ``2.0``).
        pre_eps (float): Stabilizer added to ``pre`` after sigmoid.

    Returns:
        tuple[Tensor, Tensor, Tensor]: ``(pre, post, comb_logits)`` — ``pre`` /
        ``post`` have shape ``[..., hc_mult]``; ``comb_logits`` has shape
        ``[..., hc_mult, hc_mult]`` (still pre-softmax; feed into :func:`mhc_sinkhorn`).
    """
    return _pre_split_mixes_fwd_op(mixes, scale, base, post_mult_value, pre_eps)


# ---------------------------------------------------------------------------
# pre_apply_mix: o[N, h] = sum_{hc} mix[N, hc] * x[N, hc, h]
# ---------------------------------------------------------------------------


@torch.library.custom_op("xtuner::mhc_pre_apply_mix_fwd", mutates_args=())
def _pre_apply_mix_fwd_op(x: Tensor, mix: Tensor) -> Tensor:
    *batch, hc_mult, hidden = x.shape
    n = 1
    for b in batch:
        n *= b
    x_flat = x.contiguous().view(n, hc_mult, hidden)
    mix_flat = mix.contiguous().view(n, hc_mult)
    out = torch.empty(n, hidden, dtype=x.dtype, device=x.device)
    kernel = _pre_apply_mix_fwd_kernel(hc_mult, hidden)
    kernel(x_flat, mix_flat, out)
    return out.view(*batch, hidden)


@_pre_apply_mix_fwd_op.register_fake
def _(x: Tensor, mix: Tensor) -> Tensor:
    *batch, _, hidden = x.shape
    return x.new_empty(*batch, hidden)


@torch.library.custom_op("xtuner::mhc_pre_apply_mix_bwd", mutates_args=())
def _pre_apply_mix_bwd_op(o_grad: Tensor, x: Tensor, mix: Tensor) -> tuple[Tensor, Tensor]:
    *batch, hc_mult, hidden = x.shape
    n = 1
    for b in batch:
        n *= b
    x_flat = x.contiguous().view(n, hc_mult, hidden)
    mix_flat = mix.contiguous().view(n, hc_mult)
    o_grad_flat = o_grad.contiguous().view(n, hidden)
    # The TileLang kernel writes ``x_grad`` in-place (accumulate) and returns ``mix_grad``
    # via ``out_idx=[4]`` — preallocate a zero buffer for ``x_grad`` so the kernel's ``+=``
    # starts from a clean slate.
    x_grad = torch.zeros_like(x_flat)
    kernel = _pre_apply_mix_bwd_kernel(hc_mult, hidden)
    mix_grad = kernel(o_grad_flat, x_flat, mix_flat, x_grad)
    return x_grad.view(x.shape), mix_grad.view(mix.shape)


@_pre_apply_mix_bwd_op.register_fake
def _(o_grad: Tensor, x: Tensor, mix: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), torch.empty_like(mix)


def _pre_apply_mix_setup_context(ctx, inputs, output) -> None:
    x, mix = inputs
    ctx.save_for_backward(x, mix)


def _pre_apply_mix_backward(ctx, o_grad):
    x, mix = ctx.saved_tensors
    return _pre_apply_mix_bwd_op(o_grad.contiguous(), x, mix)


_pre_apply_mix_fwd_op.register_autograd(_pre_apply_mix_backward, setup_context=_pre_apply_mix_setup_context)


def mhc_pre_apply_mix(x: Tensor, mix: Tensor) -> Tensor:
    """Weighted reduce over the hc_mult axis: ``o[..., d] = sum_h mix[..., h] * x[..., h, d]``.

    Args:
        x (Tensor): HC-expanded streams, shape ``[..., hc_mult, hidden]`` bf16.
        mix (Tensor): Per-stream weights, shape ``[..., hc_mult]`` fp32.

    Returns:
        Tensor: Reduced single-stream output, shape ``[..., hidden]`` bf16.
    """
    return _pre_apply_mix_fwd_op(x, mix)


def mhc_sinkhorn(x: Tensor, repeat: int, eps: float) -> Tensor:
    """Doubly-stochastic projection of a per-token ``[hc_mult, hc_mult]``
    matrix.

    Replaces the ``softmax + iterative row/col normalize`` loop in
    :func:`xtuner.v1.module.decoder_layer.deepseek_v4.hc_sinkhorn.hc_split_sinkhorn`. ``x`` is the
    pre-softmax logits — ``softmax`` and ``+ eps`` are folded inside the kernel.

    Args:
        x (Tensor): Input logits, shape ``[..., hc_mult, hc_mult]`` fp32.
        repeat (int): Number of Sinkhorn rounds (compile-time constant).
        eps (float): Stabilizer (compile-time constant).

    Returns:
        Tensor: Doubly-stochastic ``[..., hc_mult, hc_mult]`` fp32.
    """
    return _sinkhorn_fwd_op(x, repeat, eps)
