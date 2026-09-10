"""Tests for the GLM-5.2 recipe-level selector chunk default."""

import pytest

from xtuner.v1.model.moe.glm52.indexer_chunk import (
    DEFAULT_INDEXER_TOPK_QUERY_CHUNK_SIZE,
    resolve_indexer_topk_query_chunk_size,
)


@pytest.mark.parametrize(
    ("raw_value", "backend", "expected"),
    [
        (None, "tilelang", DEFAULT_INDEXER_TOPK_QUERY_CHUNK_SIZE),
        (None, "cudnn_dsa", DEFAULT_INDEXER_TOPK_QUERY_CHUNK_SIZE),
        (None, "torch", None),
        ("1024", "tilelang", 1024),
        ("0", "tilelang", None),
    ],
)
def test_resolve_query_chunk_size(raw_value, backend, expected):
    assert resolve_indexer_topk_query_chunk_size(raw_value, backend) == expected


@pytest.mark.parametrize("raw_value", ["-1", "not-an-integer"])
def test_resolve_query_chunk_size_rejects_invalid_values(raw_value):
    with pytest.raises(ValueError, match="INDEXER_TOPK_QUERY_CHUNK_SIZE"):
        resolve_indexer_topk_query_chunk_size(raw_value, "tilelang")
