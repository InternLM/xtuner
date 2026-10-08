# Copyright (c) OpenMMLab. All rights reserved.
"""Fused SwiGLU elementwise kernels (NPU/Triton).

Replaces the two-pass ``silu(gate) * up`` forward (Swish + Mul) and the two-pass
backward (SiluGrad + Mul-grad) with one memory pass each. Reads gate/up from
separate tensors, so no concat and no weight restructuring is required.
Elementwise math only: matches the fp32-upcast convention used by the reference
silu backward.

NPU/BiSheng constraints (measured, see OPERATOR_OPT_PLAN.md):
- coreDim <= 65535, and a 2D grid is flattened, so oversized launches need a
  grid-stride loop kernel variant.
- BLOCK>=16384 overflows UB for 3x fp32 buffers. The single-shot bwd (5 live fp32
  values) fits at BLOCK 8192, but its grid-stride variant overflows UB there and
  runs at 4096.
- Even UNUSED extra scalar args slow the single-shot kernel by ~60% (BiSheng
  codegen), so the single-shot variants take only the args they use.
"""

import os

import torch
import triton
import triton.language as tl


def fused_swiglu_enabled() -> bool:
    """Whether the fused swiglu path is enabled.

    Read live (not at import) so env injected after xtuner import -- phase-file style
    overrides, tests -- still applies. Opt-in: default OFF; the cluster flips it via
    XTUNER_NPU_FUSED_SWIGLU=1 in the launch script.
    """
    return os.environ.get("XTUNER_NPU_FUSED_SWIGLU", "0") == "1"


@triton.jit
def _swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = u * g * tl.sigmoid(g)
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_fwd_kernel_gs(gate_ptr, up_ptr, out_ptr, n_elements, n_blocks, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nprog = tl.num_programs(0).to(tl.int64)
    for i in range(pid, n_blocks, nprog):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = u * g * tl.sigmoid(g)
        tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(gate_ptr, up_ptr, dout_ptr, dgate_ptr, dup_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(dout_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    # d/dg silu = sig + g * sig * (1 - sig) = sig * (1 + g * (1 - sig))
    tl.store(dgate_ptr + offs, (dout * u * sig * (1.0 + g * (1.0 - sig))).to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(dup_ptr + offs, (dout * silu).to(dup_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel_gs(gate_ptr, up_ptr, dout_ptr, dgate_ptr, dup_ptr, n_elements, n_blocks, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nprog = tl.num_programs(0).to(tl.int64)
    for i in range(pid, n_blocks, nprog):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        silu = g * sig
        tl.store(
            dgate_ptr + offs, (dout * u * sig * (1.0 + g * (1.0 - sig))).to(dgate_ptr.dtype.element_ty), mask=mask
        )
        tl.store(dup_ptr + offs, (dout * silu).to(dup_ptr.dtype.element_ty), mask=mask)


def _grid(n: int, block: int, cap: int = 65535) -> tuple[int, int, int, bool]:
    """(BLOCK, n_blocks, grid, single): single=True -> one-shot kernel, else grid-stride."""
    n_blocks = triton.cdiv(n, block)
    return block, n_blocks, min(n_blocks, cap), n_blocks <= cap


class SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(gate)
        n = gate.numel()
        if n > 0:  # a 0-element launch fails (coreDim must be > 0); empty out is the answer
            BLOCK, n_blocks, grid, single = _grid(n, block=8192)  # 3 fp32 buffers fit UB at 8192
            if single:
                _swiglu_fwd_kernel[(grid,)](gate, up, out, n, BLOCK=BLOCK)
            else:
                _swiglu_fwd_kernel_gs[(grid,)](gate, up, out, n, n_blocks, BLOCK=BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        gate, up = ctx.saved_tensors
        # The kernels index dout by flat data_ptr; a strided grad (e.g. from a
        # transposed consumer) would silently corrupt dgate/dup. contiguous() is a
        # no-op for the dense grads the MLP wiring produces.
        dout = dout.contiguous()
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        n = gate.numel()
        if n > 0:  # a 0-element launch fails (coreDim must be > 0); grads stay empty
            # bwd has 5 live fp32 values: the single-shot kernel fits UB at BLOCK 8192,
            # but its grid-stride variant overflows UB -> looped fallback runs at 4096.
            BLOCK, n_blocks, grid, single = _grid(n, block=8192)
            if single:
                _swiglu_bwd_kernel[(grid,)](gate, up, dout, dgate, dup, n, BLOCK=BLOCK)
            else:
                BLOCK, n_blocks, grid, _ = _grid(n, block=4096)
                _swiglu_bwd_kernel_gs[(grid,)](gate, up, dout, dgate, dup, n, n_blocks, BLOCK=BLOCK)
        return dgate, dup


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """out = silu(gate) * up for arbitrary leading dims.

    Both inputs must be contiguous: the kernels index flat data_ptr offsets with no
    stride handling, so a non-contiguous view would silently read the wrong values.
    No runtime fallback: a Triton compile/launch failure raises rather than degrading
    to the eager chain (opt-in perf knob; disable via XTUNER_NPU_FUSED_SWIGLU=0).
    """
    # Explicit raises, not asserts: asserts vanish under `python -O`, and both
    # violations below read out-of-bounds/strided instead of failing loudly.
    if gate.shape != up.shape:
        raise ValueError(f"fused_swiglu gate/up shape mismatch: {gate.shape} vs {up.shape}")
    if not (gate.is_contiguous() and up.is_contiguous()):
        raise ValueError("fused_swiglu requires contiguous gate/up (kernels index flat data_ptr offsets)")
    return SwiGLUFn.apply(gate, up)
