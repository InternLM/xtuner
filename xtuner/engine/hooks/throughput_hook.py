# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Optional, Union

import torch
from mmengine import print_log
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper
from torch.utils._pytree import tree_flatten

from xtuner.parallel.sequence import get_sequence_parallel_world_size

DATA_BATCH = Optional[Union[dict, tuple, list]]


class ThroughputHook(Hook):

    # priority must be higher than LoggerHook (50) and lower than
    # IterTimerHook (60)
    priority = 55

    def __init__(self,
                 use_activation_checkpointing=None,
                 hidden_size=None,
                 num_layers=None,
                 vocab_size=None,
                 mlp_ratio=None,
                 is_casual=None):
        self.use_activation_checkpointing = use_activation_checkpointing
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.mlp_ratio = mlp_ratio
        self.is_casual = is_casual

    @staticmethod
    def _guess_is_casual_attn(model):
        for module in model.modules():
            if hasattr(module, 'is_causal'):
                return module.is_causal
        print_log(
            'It\'s impossible to speculate whether casual attention was used, '
            'and FLOPs will be calculated as `casual = True`.', 'current')
        return True

    @staticmethod
    def _get_batch_size_and_sequence_len(data_batch):
        data_list, _ = tree_flatten(data_batch)
        for data in data_list:
            if isinstance(data, torch.Tensor):
                return data.size(0), data.size(1)
        raise RuntimeError('No tensor found in the batch')

    @staticmethod
    def _guess_use_activation_checkpointing(model):
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                return module.gradient_checkpointing
        return False

    def before_run(self, runner) -> None:
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        self.use_activation_checkpointing = \
            (self.use_activation_checkpointing or
             self._guess_use_activation_checkpointing(model))
        self.hidden_size = self.hidden_size or model.config.hidden_size
        self.num_layers = self.num_layers or model.config.num_hidden_layers
        self.vocab_size = self.vocab_size or model.config.vocab_size
        self.mlp_ratio = self.mlp_ratio or (model.config.intermediate_size /
                                            model.config.hidden_size)
        self.mlp_ratio *= 1.5  # has gate_proj
        self.is_casual = self.is_casual if self.is_casual is not None \
            else self._guess_is_casual_attn(model)

        use_varlen_attn = getattr(model, 'use_varlen_attn', False)
        if use_varlen_attn:
            print_log(
                'Using variable-length Flash Attention causes an inflation'
                ' in the FLOPs calculation.',
                'current',
                level=logging.WARNING)

        return

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Calc flops based on the paper of Megatron
        https://deepakn94.github.io/assets/papers/megatron-sc21.pdf."""

        batch_size, sequence_len = self._get_batch_size_and_sequence_len(
            data_batch)
        sequence_parallel_size = get_sequence_parallel_world_size()
        sequence_len /= sequence_parallel_size

        message_hub = runner.message_hub
        iter_time = message_hub.get_scalar('train/time').current()

        # We consider a language model with 𝑙 transformer layers,
        # hidden size h, sequence length s, vocabulary size V, and
        # training batch size B.
        # A $A_{mxk}$ x $X_{kxn}$ matrix multiplication requires 2𝑚 ×𝑘 ×𝑛 FLOPs
        # (factor of 2 needed to account for multiplies and adds).

        # Attention Layer:
        # qkv_proj + o_proj: 8B * s * h^2
        # attn: 2B * s^2 * h (casual=False) and 2B * s^2 * h / 2 (casual=True)

        # MLP Layer:
        # up_proj + down_proj + gate_proj: 4B * s * h^2 * mlp_ratio
        # (In Llama mlp_ratio = intermediate_size / hidden_size * 1.5
        # (has gate_proj))

        # The backward pass requires double the number of FLOPs since we
        # need to calculate the gradients with respect to both input and
        # weight tensors. In addition, we are using activation recomputation,
        # which requires an additional forward pass before the backward pass.

        # While sequence parallel will affect the FLOPs calculation in attn.
        # Suppose the sequence length in one GPU is s and the sequence
        # parallel world size is `sp_size`, which means the total
        # sequence length in the attention calculation is
        # `s * sp_size` and the number of attention heads decrease to
        # `num_heads / sp_size`. Hence, the FLOPs in attn calculation is:
        # 2B * (s * sp_size)^2 * (h / sp_size) (casual=False) and
        # 2B * (s * sp_size)^2 * (h / sp_size) / 2 (casual=True)

        flops_qkvo_proj = 8 * batch_size * sequence_len * self.hidden_size**2
        flops_attn = 4 * batch_size * sequence_len**2 * self.hidden_size * \
            sequence_parallel_size / (int(self.is_casual) + 1)
        flops_mlp = 4 * self.mlp_ratio * batch_size * sequence_len * \
            self.hidden_size**2
        flops_wo_head = (3 + int(self.use_activation_checkpointing)) * (
            flops_qkvo_proj + flops_attn + flops_mlp) * self.num_layers
        flops_head = 3 * 2 * batch_size * sequence_len * self.hidden_size * \
            self.vocab_size
        flops_per_iteration = flops_wo_head + flops_head

        avg_tflops_per_gpu = flops_per_iteration / 1e12 / (iter_time + 1e-12)
        tokens_per_sec_per_gpu = batch_size * sequence_len / (
            iter_time + 1e-12)

        message_hub.update_scalar('train/tflops', avg_tflops_per_gpu)
        message_hub.update_scalar('train/tokens_per_sec',
                                  tokens_per_sec_per_gpu)
