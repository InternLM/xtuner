# Copyright (c) OpenMMLab. All rights reserved.

"""Recipe-level default for GLM-5.2 Indexer query chunking."""

DEFAULT_INDEXER_TOPK_QUERY_CHUNK_SIZE = 8192


def resolve_indexer_topk_query_chunk_size(raw_value: str | None, backend: str) -> int | None:
    """Resolve the recipe setting without changing low-level API defaults."""

    backend = backend.strip().lower()
    if raw_value is None:
        return DEFAULT_INDEXER_TOPK_QUERY_CHUNK_SIZE if backend in ("tilelang", "cudnn_dsa") else None

    value = raw_value.strip().lower()
    if value in ("", "0", "none", "null"):
        return None
    try:
        chunk_size = int(value)
    except ValueError as exc:
        raise ValueError("INDEXER_TOPK_QUERY_CHUNK_SIZE must be a positive integer or 0/none") from exc
    if chunk_size <= 0:
        raise ValueError("INDEXER_TOPK_QUERY_CHUNK_SIZE must be a positive integer or 0/none")
    return chunk_size
