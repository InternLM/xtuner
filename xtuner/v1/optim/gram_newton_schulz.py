# Copyright (c) OpenMMLab. All rights reserved.
"""Gram Newton-Schulz orthogonalization for XTuner's distributed Muon.

The algorithm and CuTeDSL-backed GEMM interface are adapted from
https://github.com/Dao-AILab/gram-newton-schulz by Jack Zhang, Noah Amsel,
Berlin Chen, and Tri Dao (MIT license).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor


SYMMETRIC_KERNEL_TILE_SIZE = 256
POLAR_EXPRESS_SAFETY_FACTOR = 1.05
_UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
POLAR_EXPRESS_COEFFICIENTS = tuple(
    (
        a / POLAR_EXPRESS_SAFETY_FACTOR,
        b / POLAR_EXPRESS_SAFETY_FACTOR**3,
        c / POLAR_EXPRESS_SAFETY_FACTOR**5,
    )
    for a, b, c in _UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS
)

_TORCH_BACKEND = SimpleNamespace(
    sym_mm=lambda A, B: A @ B,
    sym_baddbmm=lambda A, B, C, alpha=1.0, beta=1.0: torch.baddbmm(C, A, B, alpha=alpha, beta=beta),
    mm=lambda A, B: A @ B,
    mm_add=lambda A, B, C, beta: torch.baddbmm(C, A, B, beta=beta),
)


def _make_kernel_backend() -> SimpleNamespace:
    try:
        from quack.gemm_interface import gemm, gemm_add, gemm_symmetric
    except ImportError as error:
        raise ImportError(
            "CuTeDSL Gram Newton-Schulz requires quack-kernels and nvidia-cutlass-dsl. "
            "Install XTuner with `pip install -e '.[gram-ns]' --no-build-isolation`."
        ) from error

    return SimpleNamespace(
        sym_mm=gemm_symmetric,
        sym_baddbmm=lambda A, B, C, alpha=1.0, beta=1.0: gemm_symmetric(A, B, C=C, alpha=alpha, beta=beta),
        mm=lambda A, B: gemm(A, B, tuned=False),
        mm_add=lambda A, B, C, beta: gemm_add(A, B, C=C, beta=beta, tuned=False),
    )


def _make_gram_newton_schulz(
    ops: SimpleNamespace,
    coefficients: Sequence[tuple[float, float, float]],
    restart_iterations: Sequence[int],
    epsilon: float,
    compile_kwargs: dict[str, Any] | None,
) -> Callable[[Tensor], Tensor]:
    """Build a closure with fixed operators so ``torch.compile`` sees a stable
    graph."""
    coefficients = tuple(coefficients)
    restart_set = frozenset(restart_iterations)

    def gram_newton_schulz(X: Tensor) -> Tensor:
        tall = X.size(-2) > X.size(-1)
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)
        X = X.to(torch.float16)

        R = ops.sym_mm(X.mT, X) if tall else ops.sym_mm(X, X.mT)
        identity = (
            torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(R.size(0), -1, -1).contiguous()
        )
        Q = None

        for iteration, (a, b, c) in enumerate(coefficients):
            if iteration in restart_set and iteration != 0:
                X = ops.mm(X, Q) if tall else ops.mm(Q, X)
                R = ops.sym_mm(X.mT, X) if tall else ops.sym_mm(X, X.mT)
                Q = None

            Z = ops.sym_baddbmm(R, R, C=R, alpha=c, beta=b)
            if iteration == 0 or iteration in restart_set:
                Q = Z + a * identity
            else:
                Q = ops.sym_baddbmm(Q, Z, C=Q, beta=a)

            if iteration < len(coefficients) - 1 and iteration + 1 not in restart_set:
                RZ = ops.sym_baddbmm(R, Z, C=R, beta=a)
                R = ops.sym_baddbmm(Z, RZ, C=RZ, beta=a)

        return ops.mm(X, Q) if tall else ops.mm(Q, X)

    if compile_kwargs is not None:
        gram_newton_schulz = torch.compile(gram_newton_schulz, **compile_kwargs)
    return gram_newton_schulz


def _make_standard_newton_schulz(
    ops: SimpleNamespace,
    coefficients: Sequence[tuple[float, float, float]],
    epsilon: float,
    compile_kwargs: dict[str, Any] | None,
) -> Callable[[Tensor], Tensor]:
    """Build the recommended standard NS fallback with the same FP16 policy."""
    coefficients = tuple(coefficients)

    def standard_newton_schulz(X: Tensor) -> Tensor:
        tall = X.size(-2) > X.size(-1)
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)
        X = X.to(torch.float16)

        for a, b, c in coefficients:
            A = ops.sym_mm(X.mT, X) if tall else ops.sym_mm(X, X.mT)
            B = ops.sym_baddbmm(A, A, C=A, alpha=c, beta=b)
            X = ops.mm_add(X, B, C=X, beta=a) if tall else ops.mm_add(B, X, C=X, beta=a)
        return X

    if compile_kwargs is not None:
        standard_newton_schulz = torch.compile(standard_newton_schulz, **compile_kwargs)
    return standard_newton_schulz


class GramNewtonSchulz:
    """XTuner-compatible adaptive Gram Newton-Schulz callable.

    Rectangular matrices whose aspect ratio is greater than
    ``min_aspect_ratio`` use Gram NS. Square and low-aspect-ratio matrices use
    standard NS. Both paths use FP32 normalization followed by FP16 internal
    computation and the recommended safety-scaled Polar Express coefficients.

    Args:
        epsilon (float): Normalization epsilon used by both algorithms.
        use_kernels (bool): Whether to use CuTeDSL kernels for supported matrix shapes.
        restart_iterations (tuple[int, ...]): Iterations before which Gram NS rebuilds the Gram matrix.
        min_aspect_ratio (float): Gram NS is used only above this matrix aspect ratio.
        compile_kwargs (dict[str, Any] | None): Keyword arguments passed to ``torch.compile``. ``None`` disables
            compilation.
    """

    def __init__(
        self,
        *,
        epsilon: float = 1e-7,
        use_kernels: bool = True,
        restart_iterations: tuple[int, ...] = (2,),
        min_aspect_ratio: float = 2.0,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if min_aspect_ratio < 1.0:
            raise ValueError(f"min_aspect_ratio must be at least 1, got {min_aspect_ratio}")
        if any(iteration <= 0 or iteration >= len(POLAR_EXPRESS_COEFFICIENTS) for iteration in restart_iterations):
            raise ValueError(
                f"restart_iterations must be in [1, {len(POLAR_EXPRESS_COEFFICIENTS) - 1}], got {restart_iterations}"
            )

        self.epsilon = epsilon
        self.restart_iterations = tuple(restart_iterations)
        self.min_aspect_ratio = min_aspect_ratio
        self._compiled = compile_kwargs is not None

        kernel_backend = _make_kernel_backend() if use_kernels else None
        self._kernel_backend = kernel_backend
        self._gram_kernel: Callable[[Tensor], Tensor] | None = None
        self._standard_kernel: Callable[[Tensor], Tensor] | None = None
        self._gram_torch = _make_gram_newton_schulz(
            _TORCH_BACKEND,
            POLAR_EXPRESS_COEFFICIENTS,
            self.restart_iterations,
            epsilon,
            compile_kwargs,
        )
        self._standard_torch = _make_standard_newton_schulz(
            _TORCH_BACKEND, POLAR_EXPRESS_COEFFICIENTS, epsilon, compile_kwargs
        )
        if kernel_backend is not None:
            self._gram_kernel = _make_gram_newton_schulz(
                kernel_backend,
                POLAR_EXPRESS_COEFFICIENTS,
                self.restart_iterations,
                epsilon,
                compile_kwargs,
            )
            self._standard_kernel = _make_standard_newton_schulz(
                kernel_backend, POLAR_EXPRESS_COEFFICIENTS, epsilon, compile_kwargs
            )

    def _selection(self, rows: int, cols: int) -> tuple[str, str]:
        aspect_ratio = max(rows, cols) / min(rows, cols)
        algorithm = "gram" if rows != cols and aspect_ratio > self.min_aspect_ratio else "standard_fallback"
        backend = (
            "cutedsl"
            if self._kernel_backend is not None and min(rows, cols) >= SYMMETRIC_KERNEL_TILE_SIZE
            else "torch"
        )
        return algorithm, backend

    def _run_selected(self, X: Tensor) -> Tensor:
        rows, cols = X.shape[-2:]
        algorithm, backend = self._selection(rows, cols)
        if backend == "cutedsl":
            kernel = self._gram_kernel if algorithm == "gram" else self._standard_kernel
            assert kernel is not None
            output = kernel(X)
        else:
            torch_implementation = self._gram_torch if algorithm == "gram" else self._standard_torch
            output = torch_implementation(X)

        # reduce-overhead compilation may reuse CUDA-graph output buffers. Keep
        # each optimizer callback result independent from the next invocation.
        if self._compiled:
            output = output.clone()
        return output

    def __call__(self, G: Tensor, epsilon: float | Tensor = 1e-7, num_experts: int = 1) -> Tensor:
        """Orthogonalize XTuner's concatenated regular or MoE matrices.

        Args:
            G (Tensor): Concatenated 2D update matrix.
            epsilon (float | Tensor): Compatibility argument for Muon's orthogonalization callback. The configured
                instance epsilon is used.
            num_experts (int): Number of equal row-wise expert matrices concatenated in ``G``.

        Returns:
            Tensor: Orthogonalized update with the original shape and dtype.
        """
        if G.ndim != 2:
            raise ValueError(f"Gram NS expects a 2D tensor, got shape {tuple(G.shape)}")
        if num_experts <= 0 or G.size(0) % num_experts:
            raise ValueError(f"Invalid num_experts={num_experts} for input shape {tuple(G.shape)}")

        original_shape = G.shape
        original_dtype = G.dtype
        X = G.view(num_experts, G.size(0) // num_experts, G.size(1))
        output = self._run_selected(X)
        return output.to(original_dtype).view(original_shape)
