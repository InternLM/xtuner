# Copyright (c) OpenMMLab. All rights reserved.
import torch


@torch.library.custom_op("xpuyu::fill_paged_kv_cache", mutates_args=("key_cache", "value_cache"))
def fill_paged_kv_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    max_length_q: int,
    max_length_k: int,
    block_table: torch.Tensor,
) -> None:
    bs = block_table.size(0)
    from lmdeploy.pytorch.kernels import fill_kv_cache

    fill_kv_cache(
        key_states.transpose(1, 2)[:, : cu_seq_lens_k[bs]],
        value_states.transpose(1, 2)[:, : cu_seq_lens_k[bs]],
        key_cache,
        value_cache,
        cu_seq_lens_q[:bs],  # q_start_loc
        cu_seq_lens_q[1 : bs + 1] - cu_seq_lens_q[:bs],  # q_seq_length
        kv_seq_length=cu_seq_lens_k[1 : bs + 1] - cu_seq_lens_k[:bs],
        max_q_seq_length=max_length_q,
        block_offsets=block_table,
    )  # type: ignore

    return


@fill_paged_kv_cache.register_fake
def fill_paged_kv_cache_fake(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    max_length_q: int,
    max_length_k: int,
    block_table: torch.Tensor,
) -> None:
    return
