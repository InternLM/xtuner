# This file is separated from prober.py to avoid circular import.
import types
from pathlib import Path

import torch.nn as nn

from xtuner.v1.loss.moe_loss import BalancingLoss, ZLoss
from xtuner.v1.model.moe.moe import DenseDecoderLayer, LMHead, MoEBlock, MoEDecoderLayer
from xtuner.v1.module.attention.mha import MultiHeadAttention
from xtuner.v1.module.attention.mla import MultiLatentAttention
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEGate
from xtuner.v1.module.rms_norm.rms_norm import RMSNorm
from xtuner.v1.profiler.prober import ProberList


def register_prober_list(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            wrapped = ProberList.wrap_embedding_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, (DenseDecoderLayer, MoEDecoderLayer)):
            wrapped = ProberList.wrap_decoder_layer_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, RMSNorm):
            wrapped = ProberList.wrap_rms_norm_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, (MultiHeadAttention, MultiLatentAttention)):
            wrapped = ProberList.wrap_attention_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, MoEGate):
            wrapped = ProberList.wrap_moe_gate_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, MoEBlock):
            wrapped = ProberList.wrap_moe_block_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, LMHead):
            wrapped = ProberList.wrap_lm_head_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, BalancingLoss):
            wrapped = ProberList.wrap_balancing_loss_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore
        elif isinstance(module, ZLoss):
            wrapped = ProberList.wrap_z_loss_forward(module.forward, name)
            module.forward = types.MethodType(wrapped, module)  # type: ignore


def setup_prober_list(exp_dir: Path, profile_step: list[int] | None, model: nn.Module, prober_list: list[str]):
    if len(prober_list) == 0:
        return
    ProberList.setup(exp_dir, profile_step, prober_list)
    register_prober_list(model)
