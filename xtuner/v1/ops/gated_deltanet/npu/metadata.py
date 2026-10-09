"""Precomputed metadata used by NPU GatedDeltaNet operators."""

import os
from functools import lru_cache

import torch

from xtuner.v1.data_proto.sequence_context import GatedDeltaNetMetadata


def _prepare_chunk_indices_list(cu_seqlens: tuple[int, ...], block_size: int) -> list[int]:
    indices = []
    for sequence_id, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        num_chunks = (end - start + block_size - 1) // block_size
        for chunk_id in range(num_chunks):
            indices.append(sequence_id)
            indices.append(chunk_id)
    return indices


def get_npu_causal_conv1d_block_sizes(total_tokens: int) -> tuple[int, int]:
    """Return the forward and backward block sizes selected by causal-conv."""
    from .causal_conv1d.causal_conv1d_triton_ascend import get_num_cores

    num_cores = int(get_num_cores())
    tiles = 1 << (((max(16, total_tokens) + num_cores - 1) // num_cores) - 1).bit_length()
    return min(32, tiles), min(4, tiles)


def get_npu_delta_rule_block_sizes(num_heads: int, chunk_size: int) -> set[int]:
    """Return every chunk-index block size selected by the NPU delta-rule."""
    cumsum_base = max(1, (1 << 17) // (num_heads * chunk_size))
    cumsum_base = ((cumsum_base + chunk_size - 1) // chunk_size) * chunk_size
    cumsum_block_size = 1 << (cumsum_base - 1).bit_length()

    block_sizes = {chunk_size, cumsum_block_size, 608 * 2}
    block_sizes.update(size for size in (32, 64, 128) if size <= chunk_size)
    return block_sizes


@lru_cache(maxsize=8)
def _prepare_npu_metadata(
    cu_seqlens: tuple[int, ...],
    device: str,
    total_tokens: int,
    block_sizes: tuple[int, ...],
    list_block_sizes: tuple[int, ...],
) -> GatedDeltaNetMetadata:
    if not cu_seqlens or cu_seqlens[0] != 0 or cu_seqlens[-1] != total_tokens:
        raise ValueError("cu_seqlens must start at zero and end at the total token count")

    flat_indices_by_size = {
        block_size: _prepare_chunk_indices_list(cu_seqlens, block_size) for block_size in block_sizes
    }
    chunk_indices = {
        str(block_size): torch.tensor(flat_indices_by_size[block_size], device=device, dtype=torch.int64).reshape(
            -1, 2
        )
        for block_size in block_sizes
    }
    chunk_indices_list = {str(block_size): flat_indices_by_size[block_size] for block_size in list_block_sizes}
    return GatedDeltaNetMetadata(
        cu_seqlens_int64=torch.tensor(cu_seqlens, device=device, dtype=torch.int64),
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
    )


def prepare_npu_metadata(
    *,
    cu_seqlens: list[int],
    device: torch.device | str,
    total_tokens: int,
    block_sizes: set[int],
    list_block_sizes: set[int] | None = None,
) -> GatedDeltaNetMetadata:
    """Prepare the requested NPU metadata, sharing tensors by block size."""
    return _prepare_npu_metadata(
        tuple(cu_seqlens),
        str(device),
        total_tokens,
        tuple(sorted(block_sizes)),
        tuple(sorted(list_block_sizes or set())),
    )


def prepare_npu_gated_deltanet_metadata(
    *,
    cu_seqlens: list[int],
    device: torch.device | str,
    total_tokens: int,
    num_heads: int,
) -> GatedDeltaNetMetadata:
    """Prepare the union of NPU causal-conv and delta-rule metadata."""
    chunk_size = int(os.environ.get("CHUNK_SIZE", "64"))
    causal_fwd_block_size, causal_bwd_block_size = get_npu_causal_conv1d_block_sizes(total_tokens)
    block_sizes = get_npu_delta_rule_block_sizes(num_heads, chunk_size)
    block_sizes.update((causal_fwd_block_size, causal_bwd_block_size))
    return prepare_npu_metadata(
        cu_seqlens=cu_seqlens,
        device=device,
        total_tokens=total_tokens,
        block_sizes=block_sizes,
        list_block_sizes={chunk_size},
    )
