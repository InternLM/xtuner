from typing import Protocol, Tuple

import torch


class ApplyRotaryEmbProtocol(Protocol):
    def __call__(self, query_states, key_states, cos, sin) -> Tuple[torch.Tensor, torch.Tensor]: ...


def get_apply_rotary_emb() -> ApplyRotaryEmbProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "npu":

        def npu_apply_rotary_pos_emb(query_states, key_states, cos, sin) -> Tuple[torch.Tensor, torch.Tensor]:
            import torch_npu

            query_states = torch_npu.npu_rotary_mul(query_states, cos.unsqueeze(1), sin.unsqueeze(1))
            key_states = torch_npu.npu_rotary_mul(key_states, cos.unsqueeze(1), sin.unsqueeze(1))
            return query_states, key_states

        return npu_apply_rotary_pos_emb
    else:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        return apply_rotary_pos_emb


apply_rotary_pos_emb = get_apply_rotary_emb()
