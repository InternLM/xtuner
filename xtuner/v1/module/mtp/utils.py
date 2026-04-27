"""Utility functions for Multi-Token Prediction (MTP)."""

import torch

from xtuner.v1.data_proto import SequenceContext


def roll_packed_tensor(
    tensor: torch.Tensor,
    cu_seq_lens: torch.IntTensor,
    shifts: int = -1,
    dim: int = -1,
    fill_value: float | int = 0,
) -> torch.Tensor:
    """Roll a packed tensor along the specified dimension.

    This function respects sequence boundaries in packed sequences, shifting each
    sequence independently without crossing boundaries.

    Args:
        tensor (torch.Tensor): Input packed tensor to roll.
        cu_seq_lens (torch.IntTensor): Cumulative sequence lengths defining packed
            sequence boundaries. Shape [num_sequences + 1].
        shifts (int): Number of positions to shift. Use -1 for left shift (default).
            Only negative shifts are supported.
        dim (int): Dimension along which to roll. The ``cu_seq_lens`` boundaries
            are applied on this dimension. Default is -1 (last dimension).
        fill_value (float | int): Value used to fill boundary positions after rolling.
            Defaults to 0. Use the loss ignore index (e.g., -100) when rolling label
            tensors to ensure boundary positions are excluded from loss computation.

    Returns:
        torch.Tensor: Rolled tensor with boundary positions filled with ``fill_value``.

    Example:
        For packed sequences [1,2,3] and [4,5,6] with shifts=-1, dim=-1:
        >>> tensor = torch.tensor([[1, 2, 3, 4, 5, 6]])
        >>> cu_seq_lens = torch.tensor([0, 3, 6], dtype=torch.int32)
        >>> rolled = roll_packed_tensor(tensor, cu_seq_lens, shifts=-1, dim=-1)
        >>> rolled  # [[2, 3, 0, 5, 6, 0]]

        For a 3D tensor with dim=-2 (e.g., inputs_embeds of shape [1, seq_len, hidden]):
        >>> tensor = torch.arange(12).reshape(1, 6, 2)
        >>> cu_seq_lens = torch.tensor([0, 3, 6], dtype=torch.int32)
        >>> rolled = roll_packed_tensor(tensor, cu_seq_lens, shifts=-1, dim=-2)
        >>> rolled[0, 2]  # tensor([0, 0])  (boundary filled with fill_value=0)
    """
    assert shifts <= 0, "Only negative shift is supported"

    # Normalize dim to a positive index
    dim = dim % tensor.dim()

    rolled_tensor = tensor.clone()

    # Roll each packed sequence independently within its boundaries
    for i in range(len(cu_seq_lens) - 1):
        start_idx = cu_seq_lens[i].item()
        end_idx = cu_seq_lens[i + 1].item()

        # Extract sequence slice along the specified dimension
        seq_slice = tensor.narrow(dim, start_idx, end_idx - start_idx)  # type: ignore[arg-type]
        rolled_seq = torch.roll(seq_slice, shifts=shifts, dims=dim)

        # Fill the last |shifts| positions along dim to avoid information
        # leakage across sequences.  For shifts=-1 the last 1 position is
        # filled; for shifts=-2 the last 2 positions are filled, etc.
        seq_len = end_idx - start_idx
        fill_len = min(-shifts, seq_len)
        fill_start = seq_len - fill_len
        fill_slice = rolled_seq.narrow(dim, fill_start, fill_len)  # type: ignore[arg-type]
        fill_slice.fill_(fill_value)

        # Write back to the rolled tensor
        rolled_tensor.narrow(dim, start_idx, end_idx - start_idx).copy_(rolled_seq)  # type: ignore[arg-type]

    return rolled_tensor


def roll_sequence_context(
    seq_ctx: SequenceContext,
    shifts: int = -1,
) -> SequenceContext:
    """Roll the sequence context to get future tokens for MTP prediction.

    This function respects sequence boundaries in packed sequences, shifting each
    sequence independently without crossing boundaries. Returns a new
    ``SequenceContext`` — the original is never modified.

    Args:
        seq_ctx (SequenceContext): Input sequence context with packed sequences.
        shifts (int): Number of positions to shift. Use -1 for left shift (default).
            Only -1 is currently supported.

    Returns:
        SequenceContext: A new sequence context with shifted input_ids (and/or
            inputs_embeds). Positions at sequence boundaries are zeroed to prevent
            information leakage.

    Example:
        For packed sequences [1,2,3] and [4,5,6] with shifts=-1:
        Original input_ids:  [1, 2, 3, 4, 5, 6]
        Rolled input_ids:    [2, 3, 0, 5, 6, 0]
    """
    sp_mesh = seq_ctx.sequence_parallel_mesh
    is_sp = sp_mesh is not None and sp_mesh.size() > 1

    overrides: dict = {}

    raw_input_ids = seq_ctx.raw_input_ids
    if raw_input_ids is not None:
        rolled = roll_packed_tensor(tensor=raw_input_ids, cu_seq_lens=seq_ctx.cu_seq_lens_q, shifts=shifts, dim=-1)
        overrides["raw_input_ids"] = rolled
        if is_sp:
            s = seq_ctx._shard_start
            overrides["input_ids"] = rolled[:, s : s + seq_ctx._shard_size]
        else:
            overrides["input_ids"] = rolled

    raw_inputs_embeds = seq_ctx.raw_inputs_embeds
    if raw_inputs_embeds is not None:
        rolled_e = roll_packed_tensor(
            tensor=raw_inputs_embeds, cu_seq_lens=seq_ctx.cu_seq_lens_q, shifts=shifts, dim=-2
        )
        overrides["raw_inputs_embeds"] = rolled_e
        if is_sp:
            s = seq_ctx._shard_start
            overrides["inputs_embeds"] = rolled_e[:, s : s + seq_ctx._shard_size]
        else:
            overrides["inputs_embeds"] = rolled_e

    return seq_ctx.copy(**overrides)
