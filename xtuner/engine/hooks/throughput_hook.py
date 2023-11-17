# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper
from torch.utils._pytree import tree_flatten

DATA_BATCH = Optional[Union[dict, tuple, list]]


class ThroughputHook(Hook):
    priority = 'BELOW_NORMAL'

    def __init__(self,
                 use_activation_checkpointing=None,
                 hidden_size=None,
                 num_layers=None,
                 vocab_size=None,
                 mlp_ratio=None):
        self.use_activation_checkpointing = use_activation_checkpointing
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.mlp_ratio = mlp_ratio

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
        return

    def _get_batch_size_and_sequence_len(self, data_batch):
        data_list, _ = tree_flatten(data_batch)
        for data in data_list:
            if isinstance(data, torch.Tensor):
                return data.size(0), data.size(1)
        raise RuntimeError('No tensor found in the batch')

    def _guess_use_activation_checkpointing(self, model):
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                return module.gradient_checkpointing
        return False

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Calc flops based on the paper of Megatron
        https://deepakn94.github.io/assets/papers/megatron-sc21.pdf."""

        batch_size, sequence_len = self._get_batch_size_and_sequence_len(
            data_batch)

        message_hub = runner.message_hub
        iter_time = message_hub.get_scalar('train/time').current()

        flops_per_iteration = (
            (3 + int(self.use_activation_checkpointing)) *
            ((8 + self.mlp_ratio * 4) * batch_size * sequence_len *
             self.hidden_size**2 +
             4 * batch_size * sequence_len**2 * self.hidden_size)
        ) * self.num_layers + \
            6 * batch_size * sequence_len * self.hidden_size * self.vocab_size

        avg_tflops_per_gpu = flops_per_iteration / 1e12 / (iter_time + 1e-12)
        tokens_per_sec_per_gpu = batch_size * sequence_len / (
            iter_time + 1e-12)

        message_hub.update_scalar('train/tflops', avg_tflops_per_gpu)
        message_hub.update_scalar('train/tokens_per_sec',
                                  tokens_per_sec_per_gpu)
