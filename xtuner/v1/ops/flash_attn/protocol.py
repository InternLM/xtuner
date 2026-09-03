from typing import Protocol


class FlashAttnVarlenProtocol(Protocol):
    def __call__(
        self,
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
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
        cu_seqlens_q_list: list[int] | None = None,
        cu_seqlens_k_list: list[int] | None = None,
    ): ...


def cpu_flash_varlen_attn(
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
    cu_seqlens_q_list: list[int] | None = None,
    cu_seqlens_k_list: list[int] | None = None,
):
    raise NotImplementedError("CPU Flash Attention is not implemented yet.")
