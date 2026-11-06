"""NPU fused gated RMSNorm for GatedDeltaNet.

One fused RMSNorm pass (fp32 gamma/rstd, bf16 output) plus a bf16 silu/mul
replaces the ~13-pass fp32 decomposition; native NpuRmsNormBackward0
autograd carries the gradients. XTUNER_NPU_FUSED_GATED_NORM=1 (default)
selects the fused branch inside ``forward``; the module's own one-shot
``npu_rms_norm`` self-test downgrades the whole class to the fp32 python
fallback when the fused op fails.
"""

import os

import torch
import torch.nn.functional as F
import torch_npu  # type: ignore[import-untyped]
from torch import nn
from torch.distributed.tensor import DTensor


_FUSED_GATED_NORM = bool(int(os.environ.get("XTUNER_NPU_FUSED_GATED_NORM", "1")))


def _npu_rms_norm_selftest() -> bool:
    """Probe the ``npu_rms_norm`` op the fused branch rests on."""
    try:
        dev = f"npu:{os.environ.get('LOCAL_RANK', '0')}"
        x = torch.randn(64, 128, device=dev).to(torch.bfloat16).requires_grad_()
        w = torch.ones(128, device=dev, dtype=torch.float32).requires_grad_()
        y = torch_npu.npu_rms_norm(x, w, 1e-6)[0]
        y.float().sum().backward()
        assert x.grad is not None and w.grad is not None
        return True
    except Exception as exc:
        print(
            f"[rms_norm_gated] npu_rms_norm self-test failed ({exc}); python fallback",
            flush=True,
        )
        return False


_selftest_done = False
_selftest_ok = False


def gated_fused_ok() -> bool:
    """Run the one-shot self-test once and cache the verdict.

    A failure pins the python fp32 fallback for the process lifetime by
    clearing ``_FUSED_GATED_NORM``; the env knob and this verdict must both
    allow the fused branch in ``forward``.
    """
    global _selftest_done, _selftest_ok, _FUSED_GATED_NORM
    if not _selftest_done:
        _selftest_done = True
        _selftest_ok = _npu_rms_norm_selftest()
        if not _selftest_ok:
            _FUSED_GATED_NORM = False
    return _selftest_ok


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, activation="silu", **kwargs):
        super().__init__()
        if activation not in ("silu", "swish"):
            raise ValueError("unsupported gated norm activation")
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.bias = None
        self.eps = eps

    def forward(self, x, g):
        if _FUSED_GATED_NORM and gated_fused_ok():
            # One fused RMSNorm pass (fp32 gamma/rstd, bf16 output) plus a
            # bf16 silu/mul replaces the ~13-pass fp32 decomposition; native
            # NpuRmsNormBackward0 autograd carries the gradients.
            y = torch_npu.npu_rms_norm(x, local(self.weight), epsilon=self.eps)[0]
            return y * F.silu(g.to(x.dtype))
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * local(self.weight).float() * F.silu(g.float())).to(x.dtype)
