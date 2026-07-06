# Copyright (c) OpenMMLab. All rights reserved.


def sparse_mla_fwd_interface(*args, **kwargs):
    from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface as _impl

    return _impl(*args, **kwargs)


def sparse_mla_bwd(*args, **kwargs):
    from .tilelang_sparse_mla_bwd import sparse_mla_bwd as _impl

    return _impl(*args, **kwargs)


def indexer_fwd_interface(*args, **kwargs):
    from .tilelang_indexer_fwd import indexer_fwd_interface as _impl

    return _impl(*args, **kwargs)


__all__ = ["indexer_fwd_interface", "sparse_mla_bwd", "sparse_mla_fwd_interface"]
