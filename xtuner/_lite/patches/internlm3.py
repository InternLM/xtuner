# Copyright (c) OpenMMLab. All rights reserved.
from xtuner._lite.chat import HybridChatTemplate
from xtuner._lite.modelings.internlm3.modeling_internlm3 import (
    InternLM3Attention,
    InternLM3DecoderLayer,
    InternLM3ForCausalLM,
    InternLM3RotaryEmbedding,
)

from .llama import CUDAPatchedLlamaForCausalLM


class CUDAPatchedInternLM3ForCausalLM(CUDAPatchedLlamaForCausalLM):
    rotary_emb_cls = InternLM3RotaryEmbedding
    attn_cls = InternLM3Attention
    layer_cls = InternLM3DecoderLayer
    causal_cls = InternLM3ForCausalLM

    chat_template = HybridChatTemplate(
        system="<|im_start|>system\n{system}<|im_end|>\n",
        user="<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        assistant="{assistant}<|im_end|>",
        stop_words=["<|im_end|>"],
    )

    def __init__(self, model, fsdp_config=None):
        super().__init__(model, fsdp_config)

        if fsdp_config.max_length is not None:
            self.patched_model.config.rope_scaling = {"rope_type": "default"}
            ori_max_len = self.patched_model.config.max_position_embeddings
            self.patched_model.config.max_position_embeddings = max(
                fsdp_config.max_length, ori_max_len
            )
            self.patched_model.model.rotary_emb = InternLM3RotaryEmbedding(
                self.patched_model.config
            ).to(self.device_type)


class MLUPatchedInternLM3ForCausalLM(CUDAPatchedInternLM3ForCausalLM):
    device_type = "mlu"


class MuxiPatchedInternLM3ForCausalLM(CUDAPatchedInternLM3ForCausalLM):
    device_type = "muxi"
