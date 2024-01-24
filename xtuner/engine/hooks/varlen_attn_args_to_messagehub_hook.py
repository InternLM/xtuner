# Copyright (c) OpenMMLab. All rights reserved.
from typing import Mapping, Optional, Sequence, Union

import torch
import torch.distributed as dist
from mmengine import MessageHub
from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


class VarlenAttnArgsToMessageHubHook(Hook):

    args = ('cumulative_len', 'indexes', 'max_seqlen')

    def cast_data(self, data):
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, torch.Tensor):
            return data.cuda()
        else:
            return data

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: dict = None) -> None:
        rank = dist.get_rank()
        message_hub = MessageHub.get_instance('varlen_attn_args')

        assert 'data' in data_batch.keys()
        data = data_batch['data']

        for arg in self.args:
            assert arg in data
            message_hub.update_info(f'{arg}_rank_{rank}',
                                    self.cast_data(data.pop(arg)))

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        rank = dist.get_rank()
        message_hub = MessageHub.get_instance('varlen_attn_args')

        for arg in self.args:
            message_hub.update_info(f'{arg}_rank_{rank}', None)
