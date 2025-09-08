from .group_gemm import triton_group_gemm


try:
    from .group_gemm_cutlass import cutlass_group_gemm

    cutlass_import_exception = None
except ImportError as e:
    cutlass_group_gemm = None  # type: ignore[assignment]
    cutlass_import_exception = e


try:
    import grouped_gemm

    from .permute_unpermute import cuda_token_permute, cuda_token_unpermute
except ImportError:
    from .permute_unpermute import cuda_token_permute_torch as cuda_token_permute
    from .permute_unpermute import cuda_token_unpermute_torch as cuda_token_unpermute
