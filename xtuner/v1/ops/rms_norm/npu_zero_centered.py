"""NPU fused zero-centered RMSNorm.

``(1 + w)`` folds into the fp32 gamma of the fused ``npu_rms_norm`` op; the
output rounds once to bf16, matching the fp32 reference decomposition.
XTUNER_NPU_FUSED_ZC_NORM=1 (default) selects it on NPU; a one-shot
``npu_rms_norm`` self-test falls back to the native torch expression.
"""

import os

import torch


def npu_zero_centered_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    import torch_npu  # type: ignore[import-untyped]

    # (1 + w) folds into the fp32 gamma of the fused RMSNorm op; the output
    # rounds once to bf16, matching the fp32 reference decomposition.
    return torch_npu.npu_rms_norm(x, weight.float() + 1.0, epsilon=epsilon)[0]


def _rmsnorm_selftest() -> bool:
    import torch_npu  # type: ignore[import-untyped]

    try:
        dev = f"npu:{os.environ.get('LOCAL_RANK', '0')}"
        x = torch.randn(64, 128, device=dev).to(torch.bfloat16).requires_grad_()
        w = torch.ones(128, device=dev, dtype=torch.float32).requires_grad_()
        y = torch_npu.npu_rms_norm(x, w + 1.0, 1e-6)[0]
        y.float().sum().backward()
        assert x.grad is not None and w.grad is not None
        return True
    except Exception as exc:
        print(
            f"[rms_norm] npu_rms_norm self-test failed ({exc}); python fallback",
            flush=True,
        )
        return False


_selftest_done = False
_selftest_ok = False


def rmsnorm_fused_ok() -> bool:
    """Run the npu_rms_norm self-test once and cache the verdict."""
    global _selftest_done, _selftest_ok
    if not _selftest_done:
        _selftest_done = True
        _selftest_ok = _rmsnorm_selftest()
    return _selftest_ok
