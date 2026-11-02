from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


def native_swiglu(fused_x: torch.Tensor, split_dim=-1) -> torch.Tensor:
    x1, x2 = torch.chunk(fused_x, 2, dim=split_dim)
    return F.silu(x1) * x2


def npu_swiglu(fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_swiglu(fused_x, dim=split_dim)


def native_clipped_swiglu(fused_x: torch.Tensor, split_dim=-1, alpha=1.702, limit=7) -> torch.Tensor:
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


def native_gelu(x: torch.Tensor, approximate: str | None = None) -> torch.Tensor:
    if approximate is not None:
        return F.gelu(x, approximate=approximate)
    return F.gelu(x)


def npu_gelu(x: torch.Tensor, approximate: str | None = None) -> torch.Tensor:
    import torch_npu

    if approximate is not None:
        return torch_npu.npu_gelu(x, approximate=approximate)
    return torch_npu.npu_gelu(x)


def npu_clipped_swiglu(fused_x: torch.Tensor, split_dim=-1, alpha=1.702, limit=7) -> torch.Tensor:
    raise NotImplementedError


def native_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


act_fn_type_map_cuda = {
    "swiglu": native_swiglu,
    "clipped_swiglu": native_clipped_swiglu,
    "gelu": native_gelu,
    "gelu_pytorch_tanh": partial(native_gelu, approximate="tanh"),
    "silu": native_silu,
}
act_fn_type_map_npu = {
    "swiglu": npu_swiglu,
    "clipped_swiglu": npu_clipped_swiglu,
    "gelu": npu_gelu,
    "gelu_pytorch_tanh": partial(npu_gelu, approximate="tanh"),
    "silu": native_silu,
}


def get_act_fn(act_type):
    from xtuner.v1.utils.device import get_device

    device = get_device()
    if device == "npu":
        return act_fn_type_map_npu[act_type]
    else:
        return act_fn_type_map_cuda[act_type]


# Persistent-grid cap, at the aiv-mode physical block count (48): binaries
# compiled with TRITON_ALL_BLOCKS_PARALLEL=1 replay correctly past the physical
# count, while a stale cache entry compiled without the flag still silently
# drops blocks past the aicore-mode clamp of 24 -- cap the persistent grid so
# every launch stays inside the regime its binary was compiled for.
_GATED_ACT_MAX_PROGS = 48

try:
    _NPU_AVAILABLE = torch.npu.is_available()  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - CPU-only environments
    _NPU_AVAILABLE = False


def _fused_device_ok(t: torch.Tensor) -> bool:
    """Per-tensor device gate for the fused kernels.

    ``_NPU_AVAILABLE`` alone is a machine-level property: a CPU-resident tensor on an NPU
    box would pass it and reach the triton NPU backend with a host pointer -- a hard launch
    failure instead of the eager fallback. Same convention as xtuner1022's act_fn.

    Args:
        t (torch.Tensor): Candidate operand.

    Returns:
        bool: Whether ``t`` lives on an NPU device.
    """
    return t.device.type == "npu"


@triton.jit
def _silu_mul_fwd_kernel(
    y_ptr,
    g_ptr,
    o_ptr,
    R,
    C,
    YS,
    GS,
    OS,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """out = y * silu(g), one pass, fp32 compute with a single bf16 round at the store.

    Serves the zero-centered gated-RMSNorm tail: the eager chain casts both
    operands to fp32, runs aclnnSilu and aclnnMul in fp32 and rounds once at
    the output cast -- the kernel reproduces exactly those rounding points (no
    bf16 intermediate), keeping the swap numerically transparent.
    """
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    c_tiles = C // BC
    r_off = tl.arange(0, BR)
    c_off = tl.arange(0, BC)
    for t in range(pid, tl.cdiv(R, BR) * c_tiles, nprogs):
        rt = t // c_tiles
        ct = t % c_tiles
        rows = rt * BR + r_off
        rmask = rows < R
        cols = ct * BC + c_off
        r64 = rows[:, None].to(tl.int64)
        y = tl.load(y_ptr + r64 * YS + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        g = tl.load(g_ptr + r64 * GS + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        s = g * tl.sigmoid(g)
        tl.store(o_ptr + r64 * OS + cols[None, :], (s * y).to(o_ptr.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _silu_mul_bwd_kernel(
    y_ptr,
    g_ptr,
    dout_ptr,
    dy_ptr,
    dg_ptr,
    R,
    C,
    YS,
    GS,
    DYS,
    DGS,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """Backward of the fused silu gate ``y * silu(g)``, fp32 compute, bf16 grad stores.

    Mirrors the eager autograd chain's rounding: the fp32 ``dout * y`` product
    feeds SiluBackward unrounded (the fp32 chain materializes no bf16
    intermediate), ``dy = dout * s`` and ``dg = ds * sig * (1 + g * (1 - sig))``
    are computed in fp32 and rounded to bf16 only at the store -- the two
    CastBackward rounding points of the eager chain. Measured on NPU: the
    forward is structurally bitwise (``g * tl.sigmoid(g)`` equals aclnnSilu and
    ``tl.sigmoid`` equals aclnnSigmoid bitwise, 33.5M elements, 0 mismatches);
    the residual ``dg`` mismatches vs eager (~6e-6 of elements, <=1 bf16 ulp)
    come from aclnnSiluBackward's internal fp32 arithmetic order, not from a
    tl-vs-aclnn sigmoid ulp gap.
    """
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    c_tiles = C // BC
    r_off = tl.arange(0, BR)
    c_off = tl.arange(0, BC)
    for t in range(pid, tl.cdiv(R, BR) * c_tiles, nprogs):
        rt = t // c_tiles
        ct = t % c_tiles
        rows = rt * BR + r_off
        rmask = rows < R
        cols = ct * BC + c_off
        r64 = rows[:, None].to(tl.int64)
        y = tl.load(y_ptr + r64 * YS + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        g = tl.load(g_ptr + r64 * GS + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + r64 * C + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        s = g * sig
        ds = dout * y
        dg = ds * sig * (1.0 + g * (1.0 - sig))
        dy = dout * s
        tl.store(dy_ptr + r64 * DYS + cols[None, :], dy.to(dy_ptr.dtype.element_ty), mask=rmask[:, None])
        tl.store(dg_ptr + r64 * DGS + cols[None, :], dg.to(dg_ptr.dtype.element_ty), mask=rmask[:, None])


def _silu_mul_bc(c: int) -> int | None:
    """Largest tile width in {512, 256, 128, 64} (all pow2) dividing ``c``.

    Args:
        c (int): Last-dim width of the silu-gate operands.

    Returns:
        int | None: The tile width, or ``None`` when ``c`` is not divisible by
        any supported pow2 width (the caller falls back to eager).
    """
    for bc in (512, 256, 128, 64):
        if c % bc == 0:
            return bc
    return None


def _silu_mul_ok(y: torch.Tensor, g: torch.Tensor) -> bool:
    """Gate for the fused silu gate: NPU, bf16, contiguous, same shape.

    Args:
        y (torch.Tensor): Normed input ``[..., C]``.
        g (torch.Tensor): Gate, same shape as ``y``.

    Returns:
        bool: Whether :class:`_SiluMulFn` can serve the pair.
    """
    if not (
        _NPU_AVAILABLE
        and _fused_device_ok(y)
        and _fused_device_ok(g)
        and y.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
        and y.shape == g.shape
        and y.is_contiguous()
        and g.is_contiguous()
        and y.dim() >= 1
        and y.numel() > 0
    ):
        return False
    return _silu_mul_bc(y.shape[-1]) is not None


# Compiled-kernel direct-launch cache. The triton JIT run path costs ~55us of
# host time per call on this stack (arg binding, specialization hashing,
# backend lookup). Once a kernel is compiled for a specialization, re-launching
# it through ``CompiledKernel.__getitem__``'s runner costs ~30us and is bitwise
# identical (same binary, same args). The key mirrors the JIT's own
# specialization surface (int ==1/%16 divisibility, pointer 16B alignment,
# dtype, device) so a cache hit always replays the exact binary the JIT path
# would have chosen; any other shape falls through to the JIT path, compiles,
# and joins the cache.
_DIRECT_LAUNCH_CACHE: dict[tuple, Any] = {}


def _int_spec(v: int) -> tuple[bool, bool]:
    return (v == 1, v % 16 == 0)


def _direct_launch(jit_fn: Any, tag: str, nprogs: int, *args: Any, **constexprs: Any) -> None:
    key_parts: list[Any] = [tag, nprogs]
    for a in args:
        if isinstance(a, torch.Tensor):
            key_parts.extend((a.device.index, a.dtype, a.data_ptr() % 16 == 0))
        else:
            key_parts.extend(_int_spec(a))
    key_parts.extend(sorted(constexprs.items()))
    runner = _DIRECT_LAUNCH_CACHE.get(tuple(key_parts))
    if runner is None:
        compiled = jit_fn[(nprogs,)](*args, **constexprs)
        _DIRECT_LAUNCH_CACHE[tuple(key_parts)] = compiled[(nprogs, 1, 1)]
        return
    runner(*args)


class _SiluMulFn(torch.autograd.Function):
    """Fused ``y * silu(g)`` for the zero-centered gated-RMSNorm tail.

    Large-shape branch of :func:`fused_silu_mul` (``numel >=
    _SILU_MUL_TRITON_MIN_NUMEL``; smaller shapes keep the plain bf16 tail,
    which is measured fastest there). One triton kernel per direction
    replaces the eager fp32 chain (Cast + Silu + Cast + Mul + Cast forward;
    CastBackward x3 + MulBackward + SiluBackward backward). Rounding points
    reproduce the eager fp32 chain (no bf16 intermediate; single bf16 round
    per output); the A/B harness in tests/ops/test_silu_mul_fuse_npu.py
    asserts equality on NPU. Double backward is supported: under
    ``create_graph=True`` the backward switches to an autograd-trackable
    eager formulation with the same rounding points, which the engine can
    differentiate again (arbitrary order).
    """

    @staticmethod
    def forward(ctx, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        c = y.shape[-1]
        rows = y.numel() // c
        out = torch.empty_like(y)
        bc = _silu_mul_bc(c)
        assert bc is not None, "unreachable: fused_silu_mul gates on _silu_mul_ok first"
        tiles = ((rows + 7) // 8) * (c // bc)
        _direct_launch(
            _silu_mul_fwd_kernel,
            "silu_mul_fwd",
            min(_GATED_ACT_MAX_PROGS, tiles),
            y,
            g,
            out,
            rows,
            c,
            c,
            c,
            c,
            BR=8,
            BC=bc,
            num_warps=8,
        )
        ctx.save_for_backward(y, g)
        ctx.bc = bc
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        y, g = ctx.saved_tensors
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None
        if torch.is_grad_enabled():
            # create_graph=True reaches here: an autograd-trackable eager
            # formulation with the kernel's op order (fp32 compute, bf16
            # store) so the engine can differentiate the backward again.
            y32 = y.float()
            g32 = g.float()
            u = torch.sigmoid(g32)
            dout32 = grad_out.float()
            grad_y = (dout32 * (g32 * u)).to(y.dtype) if ctx.needs_input_grad[0] else None
            grad_g = ((dout32 * y32) * u * (1.0 + g32 * (1.0 - u))).to(g.dtype) if ctx.needs_input_grad[1] else None
            return grad_y, grad_g
        c = y.shape[-1]
        rows = y.numel() // c
        grad_y = torch.empty_like(y)
        grad_g = torch.empty_like(g)
        tiles = ((rows + 7) // 8) * (c // ctx.bc)
        _direct_launch(
            _silu_mul_bwd_kernel,
            "silu_mul_bwd",
            min(_GATED_ACT_MAX_PROGS, tiles),
            y,
            g,
            grad_out.contiguous(),
            grad_y,
            grad_g,
            rows,
            c,
            c,
            c,
            c,
            c,
            BR=8,
            BC=ctx.bc,
            num_warps=8,
        )
        return (
            grad_y if ctx.needs_input_grad[0] else None,
            grad_g if ctx.needs_input_grad[1] else None,
        )


# Shape-class crossover between the plain bf16 tail and the triton pair,
# measured with an ABBA round-robin on NPU (min of rotated passes). Below
# ~16M elements the plain tail ``y * F.silu(g)`` wins outright -- 2 native
# aclnn enqueues (~20us host; fwd 20-74us, fwd+bwd 206-245us) against
# ~56-83us / 281-414us for any fused route, where the autograd-Function and
# launch overhead dominate. At 33.5M elements the triton pair wins (fwd
# 163us vs 236us, fwd+bwd 555us vs 681us): its single pass moves 3x the
# operand bytes (read y/g/dout, write each grad) against the tail's 4-op
# forward chain. Training pays forward+backward every step, so the threshold
# is set at the 33.5M-element per-rank gated-norm shape class (2**25),
# conservatively inside the unmeasured 16.8M-33.5M span on the
# no-regression side.
_SILU_MUL_TRITON_MIN_NUMEL = 1 << 25


def fused_silu_mul(y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """NPU silu-gate tail ``y * silu(g)`` with a shape-dispatched fused route.

    NPU bf16 operands at or above ``_SILU_MUL_TRITON_MIN_NUMEL`` that pass
    the :func:`_silu_mul_ok` gate run the triton kernel pair of
    :class:`_SiluMulFn` through a compiled-kernel direct-launch cache: fp32
    compute with a single bf16 round per output, bitwise equal to the eager
    fp32 chain ``(y.float() * F.silu(g.float())).to(y.dtype)`` on forward and
    ``dy`` (``dg`` within 1 bf16 ulp, from aclnnSiluBackward's internal fp32
    op order). Everything else -- shapes below the threshold, or any operand
    failing the gate -- runs the plain bf16 tail
    ``y * F.silu(g.to(y.dtype))`` unchanged: measured fastest at small shapes
    (2 native enqueues) and bitwise identical to the expression it replaces.
    The element-count check short-circuits before the operand gate, so
    small-shape calls pay only a ``numel`` comparison on the hot path.
    Relative to that tail, the triton route's outputs differ on ~27.5% of
    elements by exactly 1 bf16 ulp (forward/``dy``; ~25.7% for ``dg``): the
    tail rounds silu's bf16 output a second time in the mul, while the kernel
    keeps fp32 until the store, landing on the correctly rounded value.
    Double backward (``create_graph=True`` twice) is supported on the triton
    route; the small-shape tail is plain native autograd.

    Args:
        y (torch.Tensor): Normed input ``[..., C]`` (bf16).
        g (torch.Tensor): Gate, same shape as ``y`` (bf16).

    Returns:
        torch.Tensor: ``y * silu(g)`` (bf16), same shape.
    """
    if y.numel() >= _SILU_MUL_TRITON_MIN_NUMEL and _silu_mul_ok(y, g):
        return _SiluMulFn.apply(y, g)
    if g.dtype != y.dtype:
        g = g.to(y.dtype)
    return y * F.silu(g)
