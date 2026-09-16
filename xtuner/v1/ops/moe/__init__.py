import os
import warnings

from .protocol import (
    GroupGemmProtocol,
    MoePermuteProtocol,
    MoeUnpermuteProtocol,
    cpu_group_gemm,
    cpu_permute,
    cpu_unpermute,
)


# ``triton_dual`` is an explicit spelling for the Triton implementation when
# UltraEP's master/replica dual-base kernel is required.  It intentionally
# shares the same callable as ``triton``: the grouped-linear arguments decide
# whether the ordinary or dual-base launch is used, while this name makes the
# experiment configuration and run metadata unambiguous.
_GROUP_GEMM_BACKENDS = ("te", "triton", "triton_dual", "cutlass")


def selected_group_gemm_backend() -> str:
    """Return the grouped-GEMM backend selected by ``XTUNER_GROUP_GEMM``.

    Values: ``triton`` (default), ``te``, ``triton_dual``, ``cutlass``.
    ``get_group_gemm`` only applies this on CUDA; CPU and NPU keep their device
    kernels.

    If ``XTUNER_GROUP_GEMM`` is unset, ``triton`` is selected unless the legacy
    ``XTUNER_USE_CUTLASS_GROUP_GEMM=1`` alias is set.
    """

    explicit = os.environ.get("XTUNER_GROUP_GEMM")
    if explicit is not None:
        backend = explicit.lower()
    elif os.environ.get("XTUNER_USE_CUTLASS_GROUP_GEMM") == "1":
        warnings.warn(
            "XTUNER_USE_CUTLASS_GROUP_GEMM is deprecated; use XTUNER_GROUP_GEMM=cutlass instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = "cutlass"
    else:
        backend = "triton"
    if backend not in _GROUP_GEMM_BACKENDS:
        raise ValueError(
            f"Unsupported XTUNER_GROUP_GEMM={backend!r}; "
            "expected te, triton, triton_dual, or cutlass"
        )
    return backend


def get_group_gemm() -> GroupGemmProtocol:
    """Return the grouped-GEMM implementation for this process.

    Device first, then one CUDA backend from ``XTUNER_GROUP_GEMM``:

    * ``triton`` (default): native grouped GEMM.
    * ``te``: standalone ``te_grouped_gemm`` package
      (``pip install`` TEGroupedGEMM).
    * ``triton``: native grouped GEMM.  With UltraEP replica buffers it uses
      the dual-base kernel, so master and replica weights stay in separate
      allocations while sharing one persistent launch.
    * ``triton_dual``: explicit alias for the Triton path above.  Use this in
      UltraEP experiments to make dual-base dispatch visible in metadata.
    * ``cutlass``: the ``grouped_gemm`` CUTLASS extra.

    Replica weights on ``cutlass`` / CPU / NPU raise immediately.

    Returns:
        (GroupGemmProtocol): Grouped GEMM over master and optional replica weights.
    """

    from xtuner.v1.utils import get_device

    backend = selected_group_gemm_backend()
    device = get_device()
    if device == "cpu":
        return cpu_group_gemm
    elif device == "cuda":
        if backend == "te":
            from .cuda.group_gemm_te import te_grouped_gemm

            return te_grouped_gemm
        if backend == "cutlass":
            from .cuda import cutlass_group_gemm as cuda_group_gemm
            from .cuda import cutlass_import_exception

            if cuda_group_gemm is None:
                raise ImportError("cutlass group gemm is unavailable") from cutlass_import_exception
            print("---------------------------Using cutlass group gemm-------------------------")
            return cuda_group_gemm
        from .cuda import triton_group_gemm as cuda_group_gemm

        return cuda_group_gemm
    elif device == "npu":
        from .npu import npu_group_gemm

        return npu_group_gemm
    else:
        raise NotImplementedError


def get_token_permute() -> MoePermuteProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return cpu_permute

    elif device == "cuda":
        from .cuda import cuda_token_permute

        return cuda_token_permute
    elif device == "npu":
        from .npu import npu_token_permute

        return npu_token_permute
    else:
        raise NotImplementedError


def get_token_unpermute() -> MoeUnpermuteProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return cpu_unpermute
    elif device == "cuda":
        from .cuda import cuda_token_unpermute

        return cuda_token_unpermute
    elif device == "npu":
        from .npu import npu_token_unpermute

        return npu_token_unpermute
    else:
        raise NotImplementedError


group_gemm = get_group_gemm()
permute = get_token_permute()
unpermute = get_token_unpermute()
