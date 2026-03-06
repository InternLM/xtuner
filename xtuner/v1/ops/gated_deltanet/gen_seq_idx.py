# Copyright (c) OpenMMLab. All rights reserved.
"""Triton kernel for filling sequence indices in GatedDeltaNet."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _fill_seq_idx_kernel(
    out_ptr,
    cu_seq_lens_ptr,
    num_seqs: tl.constexpr,
):
    """Fill output tensor with sequence indices based on cu_seq_lens.
    
    For each position in the sequence, determine which sequence it belongs to
    and write the sequence index to the output.
    
    Args:
        out_ptr: Pointer to output tensor, shape (1, seq_len)
        cu_seq_lens_ptr: Pointer to cumulative sequence lengths, shape (num_seqs + 1,)
        num_seqs: Number of sequences (derived from cu_seq_lens.shape[0] - 1)
    """
    # Get the position in the sequence
    pos = tl.program_id(0)
    
    # Load cu_seq_lens into local memory for efficient access
    # cu_seq_lens has shape (num_seqs + 1,)
    # We need to find i such that cu_seq_lens[i] <= pos < cu_seq_lens[i+1]
    
    # Since num_seqs is usually small (batch size), we do a linear search
    # Note: Triton doesn't support break, so we use tl.where for conditional assignment
    seq_idx = 0
    for i in range(num_seqs):
        start = tl.load(cu_seq_lens_ptr + i)
        end = tl.load(cu_seq_lens_ptr + i + 1)
        in_range = (pos >= start) & (pos < end)
        # Only update seq_idx if current position is in range
        seq_idx = tl.where(in_range, i, seq_idx)
    
    # Write the sequence index to output
    # Output shape is (1, seq_len), so we write at position pos
    tl.store(out_ptr + pos, seq_idx)


@torch.library.custom_op("gated_deltanet::gen_seq_idx", mutates_args={})
def gen_seq_idx(
    seq_len: int,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Fill output tensor with sequence indices.
    
    Args:
        seq_len: Length of the packed sequence (total length of all sequences)
        cu_seq_lens: Cumulative sequence lengths tensor of shape (num_seqs + 1,)
                    e.g., [0, 3, 8] for sequences of length 3 and 5
    """
    out = torch.empty((1, seq_len), dtype=torch.int32, device=cu_seq_lens.device)
    seq_len = out.shape[1]
    num_seqs = cu_seq_lens.shape[0] - 1
    
    # Ensure cu_seq_lens is contiguous and on the same device
    cu_seq_lens = cu_seq_lens.contiguous()
    
    # Launch kernel with one thread per position
    grid = (seq_len,)
    with torch.cuda.device(out.device.index):
        wrap_triton(_fill_seq_idx_kernel)[grid](
            out,
            cu_seq_lens,
            num_seqs=num_seqs,
        )
    return out

@gen_seq_idx.register_fake
def gen_seq_idx_fake(
    seq_len: int,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation of gen_seq_idx for non-CUDA devices."""
    # Create an output tensor of shape (1, seq_len)
    out = torch.empty((1, seq_len), dtype=torch.int32, device=cu_seq_lens.device)  
    return out
