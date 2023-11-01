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
                 tp_size=1,
                 pp_size=1,
                 use_activation_checkpointing=None):
        self.mp_world_size = tp_size * pp_size
        self.use_activation_checkpointing = use_activation_checkpointing

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

        if is_model_wrapper(runner.model):
            model_numel = runner.model.module.model_numel
        else:
            model_numel = runner.model.model_numel
        batch_size, sequence_len = self._get_batch_size_and_sequence_len(
            data_batch)

        message_hub = runner.message_hub
        iter_time = message_hub.get_scalar('train/time').current()
        if self.use_activation_checkpointing is None:
            self.use_activation_checkpointing = \
                self._guess_use_activation_checkpointing(runner.model)

        flops = batch_size * sequence_len * model_numel * 2 * (
            3 + int(self.use_activation_checkpointing))
        avg_tflops_per_gpu = flops / 1e12 / (iter_time +
                                             1e-12) / self.mp_world_size

        message_hub.update_scalar('train/tflops', avg_tflops_per_gpu)
