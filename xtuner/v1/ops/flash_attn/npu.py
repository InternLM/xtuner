import torch
import torch_npu


def npu_flash_varlen_attn(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    # print(f"rank:{torch.distributed.get_rank()}, cu_seqlens_k:{cu_seqlens_q.type}")
    if not causal:
        fa_out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[1],
            pse=None,
            atten_mask=None,
            scale=q.shape[-1] ** -0.5,
            keep_prob=1 - dropout_p,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].tolist()),
        )[0]
    else:
        fa_out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[1],
            pse=None,
            atten_mask=torch.triu(torch.ones([2048, 2048], dtype=torch.bool, device=q.device), diagonal=1),
            scale=q.shape[-1] ** -0.5,
            keep_prob=1 - dropout_p,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].tolist()),
            sparse_mode=3,
        )[0]
    return fa_out
