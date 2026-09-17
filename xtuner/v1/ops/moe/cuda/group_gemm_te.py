"""XTuner adapter over the externally installed ``te_grouped_gemm`` package.

Kernel work lives in :mod:`te_grouped_gemm`.  This module keeps the FSDP
``[E,N,K]`` unbind, UltraEP replica weight / FP32 grad views, host
``tokens_per_expert_cpu`` → ``m_splits``, and Dynamo custom ops.

This extra is UltraEP-only and is not part of ``.[all]``. Install it the
same way as AdaptiveGEMM::

    pip install --no-build-isolation --no-deps git+https://github.com/ShilohYu/TEGroupedGEMM.git@8661ef18465241439b9c61470cf2445844d52dfc

Environment variables:

``XTUNER_GROUP_GEMM=te|triton|triton_dual|cutlass``
    Default grouped GEMM is Triton. This adapter is selected only when
    ``XTUNER_GROUP_GEMM=te`` (UltraEP production path).
``XTUNER_TE_GEMM_BACKEND=auto|cublas|cutlass|torch``
    Forwarded to the ``te_grouped_gemm`` package (also accepts
    ``TE_GROUPED_GEMM_BACKEND``).
``XTUNER_TE_GEMM_EXTENSION`` / ``TE_GROUPED_GEMM_EXTENSION``
    Optional ``.so`` override used by the package loader.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List

import torch

from xtuner.v1.utils import get_logger


logger = get_logger()

_INSTALL_HINT = (
    "te_grouped_gemm is not installed. Install it like AdaptiveGEMM:\n"
    "  pip install --no-build-isolation --no-deps "
    "git+https://github.com/ShilohYu/TEGroupedGEMM.git@8661ef18465241439b9c61470cf2445844d52dfc\n"
    "or from a local checkout: pip install --no-build-isolation --no-deps -e /path/to/TEGroupedGEMM"
)

try:
    from te_grouped_gemm import gemm as _te_pkg
    from te_grouped_gemm import general_grouped_gemm, selected_backend

    TE_GROUPED_GEMM_INSTALLED = True
    _IMPORT_ERROR: BaseException | None = None
except ImportError as exc:
    TE_GROUPED_GEMM_INSTALLED = False
    _IMPORT_ERROR = exc
    _te_pkg = None
    general_grouped_gemm = None  # type: ignore[assignment]
    selected_backend = None  # type: ignore[assignment]


_LAYOUT_CODES = {"TN": 0, "NN": 1, "NT": 2}


def _require_package() -> None:
    if not TE_GROUPED_GEMM_INSTALLED:
        raise ImportError(_INSTALL_HINT) from _IMPORT_ERROR


def _replica_as_list(replica_weight: torch.Tensor | Sequence[torch.Tensor] | None) -> list[torch.Tensor]:
    if replica_weight is None:
        return []
    if isinstance(replica_weight, torch.Tensor):
        if replica_weight.ndim != 3:
            raise ValueError("replica_weight tensor must have shape [R,N,K]")
        return list(replica_weight.unbind(0))
    return list(replica_weight)


def _physical_weights(
    master_weight: torch.Tensor, replica_weight: torch.Tensor | Sequence[torch.Tensor] | None
) -> list[torch.Tensor]:
    replicas = _replica_as_list(replica_weight)
    if replicas and replicas[0].shape != master_weight.shape[1:]:
        raise ValueError("replica_weight must have shape [R,N,K] or a list of [N,K] matching master weights")
    return list(master_weight.unbind(0)) + replicas


def _counts_list(m_splits: torch.Tensor | list[int], groups: int) -> list[int]:
    if isinstance(m_splits, torch.Tensor):
        if m_splits.ndim != 1:
            raise ValueError("m_splits must be a one-dimensional tensor")
        counts = [int(v) for v in m_splits.detach().to(device="cpu").tolist()]
    else:
        counts = [int(v) for v in m_splits]
    if len(counts) != groups:
        raise ValueError(f"expected {groups} token counts, got {len(counts)}")
    if any(v < 0 for v in counts):
        raise ValueError(f"token counts must be non-negative, got {counts}")
    return counts


@torch.library.custom_op("xtuner_te::grouped_gemm", mutates_args=())
def _native_grouped_op(
    weights: List[torch.Tensor], tokens: torch.Tensor, counts: torch.Tensor, layout: int
) -> torch.Tensor:
    layout_name = "TN" if layout == 0 else "NN"
    return _grouped_tn_nn(weights, tokens.contiguous(), counts, layout_name, grad=layout == 1)


@_native_grouped_op.register_fake
def _native_grouped_fake(
    weights: List[torch.Tensor], tokens: torch.Tensor, counts: torch.Tensor, layout: int
) -> torch.Tensor:
    out_features = weights[0].shape[0] if layout == 0 else weights[0].shape[1]
    return torch.empty((tokens.shape[0], out_features), device=tokens.device, dtype=tokens.dtype)


@torch.library.custom_op("xtuner_te::grouped_wgrad", mutates_args={"out"})
def _native_wgrad_op(
    input_act: torch.Tensor,
    grad_output: torch.Tensor,
    counts: torch.Tensor,
    out: List[torch.Tensor],
) -> None:
    _grouped_nt(input_act, grad_output, counts, out)


@_native_wgrad_op.register_fake
def _native_wgrad_fake(
    input_act: torch.Tensor,
    grad_output: torch.Tensor,
    counts: torch.Tensor,
    out: List[torch.Tensor],
) -> None:
    return None


def _grouped_tn_nn(
    weights: List[torch.Tensor],
    tokens: torch.Tensor,
    counts: torch.Tensor,
    layout: str,
    grad: bool,
) -> torch.Tensor:
    _require_package()
    m_splits = _counts_list(counts, len(weights))
    out_features = weights[0].shape[0] if layout == "TN" else weights[0].shape[1]
    packed = tokens.new_empty(tokens.shape[0], out_features)
    out, _, _ = general_grouped_gemm(
        weights,
        list(tokens.split(m_splits)),
        [packed],
        layout=layout,
        m_splits=m_splits,
        grad=grad,
        single_output=True,
    )
    return out[0]


def _grouped_nt(
    input_act: torch.Tensor,
    grad_output: torch.Tensor,
    counts: torch.Tensor,
    out: List[torch.Tensor],
) -> None:
    _require_package()
    m_splits = _counts_list(counts, len(out))
    general_grouped_gemm(
        list(input_act.split(m_splits)),
        list(grad_output.split(m_splits)),
        out,
        layout="NT",
        m_splits=m_splits,
        grad=True,
    )


class TEGroupedGemm(torch.autograd.Function):
    """Autograd node mirroring TE's ``_GroupedLinear``."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        master_weight: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        tokens_per_expert_cpu: torch.Tensor | None = None,
        replica_weight: torch.Tensor | Sequence[torch.Tensor] | None = None,
        replica_grad: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if master_weight.ndim != 3:
            raise ValueError("master_weight must have shape [E,N,K]")
        if x.ndim != 2:
            raise ValueError("x must have shape [M,K]")
        if x.shape[1] != master_weight.shape[2]:
            raise ValueError("x and master_weight feature dimensions disagree")
        weights = _physical_weights(master_weight, replica_weight)
        counts = tokens_per_expert if tokens_per_expert_cpu is None else tokens_per_expert_cpu
        if counts.ndim != 1:
            raise ValueError("token counts must be one-dimensional")
        out = _native_grouped_op(weights, x, counts, _LAYOUT_CODES["TN"])
        ctx.save_for_backward(x, master_weight, counts)
        ctx.replica_weight = replica_weight
        ctx.replica_grad = replica_grad
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, master_weight, counts = ctx.saved_tensors
        weights = _physical_weights(master_weight, ctx.replica_weight)
        dx = _native_grouped_op(weights, grad_output, counts, _LAYOUT_CODES["NN"])
        master_dw = torch.empty_like(master_weight)
        wgrad_list = list(master_dw.unbind(0))
        if ctx.replica_grad is not None:
            wgrad_list.extend(_replica_as_list(ctx.replica_grad))
        _native_wgrad_op(x.contiguous(), grad_output.contiguous(), counts, wgrad_list)
        return dx, master_dw, None, None, None, None


def te_grouped_gemm(
    x: torch.Tensor,
    master_weight: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    tokens_per_expert_cpu: torch.Tensor | None = None,
    replica_weight: torch.Tensor | Sequence[torch.Tensor] | None = None,
    replica_grad: torch.Tensor | Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute one TE-style grouped GEMM over master and optional replica
    weights."""

    _require_package()
    return TEGroupedGemm.apply(
        x, master_weight, tokens_per_expert, tokens_per_expert_cpu, replica_weight, replica_grad
    )


def _load_native():
    _require_package()
    return _te_pkg._load_native()


def _require_native(backend: str) -> None:
    if backend in {"cublas", "cutlass"}:
        _require_package()
        native = _te_pkg._load_native()
        if native is None or not hasattr(native, "te_general_grouped_gemm"):
            raise RuntimeError(
                f"XTUNER_TE_GEMM_BACKEND resolved to {backend!r} but "
                "te_grouped_gemm._C is not loaded; pip install TEGroupedGEMM"
            )


__all__ = [
    "TEGroupedGemm",
    "general_grouped_gemm",
    "selected_backend",
    "te_grouped_gemm",
]
