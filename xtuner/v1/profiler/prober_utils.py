# This file is separated from prober.py to avoid circular import.
import types
from pathlib import Path

import torch
import torch.nn as nn

from xtuner.v1.model.moe.moe import DenseDecoderLayer, LMHead, MoEBlock, MoEDecoderLayer
from xtuner.v1.module.attention.gated_deltanet import FusedRMSNormGated, GatedDeltaNet, has_fused_rms_norm_gated
from xtuner.v1.module.attention.mha import MultiHeadAttention
from xtuner.v1.module.attention.mla import MultiLatentAttention
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEGate, MoEMLP
from xtuner.v1.module.linear.linear import _Linear
from xtuner.v1.module.rms_norm.rms_norm import RMSNorm
from xtuner.v1.module.rope.rope import FourierEmbedding, Qwen3VLTextRotaryEmbedding, RotaryEmbedding
from xtuner.v1.profiler.prober import ProberList


def register_prober_list(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            wrapped = ProberList.wrap_embedding_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, (RotaryEmbedding, Qwen3VLTextRotaryEmbedding, FourierEmbedding)):
            wrapped = ProberList.wrap_rotary_emb_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, (DenseDecoderLayer, MoEDecoderLayer)):
            wrapped = ProberList.wrap_decoder_layer_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, RMSNorm):
            wrapped = ProberList.wrap_rms_norm_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif has_fused_rms_norm_gated and isinstance(module, FusedRMSNormGated):
            wrapped = ProberList.wrap_fused_rms_norm_gated_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, MoEMLP):
            wrapped = ProberList.wrap_moe_mlp_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, _Linear):
            wrapped = ProberList.wrap_linear_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, (MultiHeadAttention, MultiLatentAttention)):
            wrapped = ProberList.wrap_attention_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, GatedDeltaNet):
            wrapped = ProberList.wrap_attention_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
            module.causal_conv1d_fn = ProberList.wrap_causal_conv1d_fn(module.causal_conv1d_fn, name)
            module.chunk_gated_delta_rule = ProberList.wrap_chunk_gated_delta_rule(module.chunk_gated_delta_rule, name)
        elif isinstance(module, MoEGate):
            wrapped = ProberList.wrap_moe_gate_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, MoEBlock):
            wrapped = ProberList.wrap_moe_block_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, LMHead):
            wrapped = ProberList.wrap_lm_head_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore


def setup_prober_list(exp_dir: Path, profile_step: list[int] | None, model: nn.Module, prober_list: list[str]):
    if len(prober_list) == 0:
        return
    ProberList.setup(exp_dir, profile_step, prober_list)
    register_prober_list(model)
    torch._dynamo.reset()  # clear compile cache so next forward recompiles with prober wrappers active
