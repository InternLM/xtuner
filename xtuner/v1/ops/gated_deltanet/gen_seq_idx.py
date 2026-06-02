import torch


def gen_seq_idx(seq_len: int, cu_seq_lens: torch.Tensor) -> torch.Tensor:
    """Fill output tensor with sequence indices.

    Args:
        seq_len: Length of the packed sequence (total length of all sequences)
        cu_seq_lens: Cumulative sequence lengths tensor of shape (num_seqs + 1,)
                    e.g., [0, 3, 8] for sequences of length 3 and 5
    """
    pos = torch.arange(seq_len, device=cu_seq_lens.device, dtype=cu_seq_lens.dtype)
    # right=True gives insertion point to keep sorted order; -1 converts to seq index
    seq_idx = torch.searchsorted(cu_seq_lens, pos, right=True) - 1
    return seq_idx.unsqueeze(0).to(torch.int32)
