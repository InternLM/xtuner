# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.distributed as dist
from mmengine import MessageHub
from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


class VarlenAttnArgsToMessageHubHook(Hook):

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: dict = None) -> None:
        rank = dist.get_rank()
        message_hub = MessageHub.get_instance('varlen_attn_args')

        assert 'data' in data_batch.keys()
        data = data_batch['data']

        cumulative_len = data.pop('cumulative_len')
        assert len(cumulative_len) == 1
        cumulative_len = cumulative_len[0].cuda()
        message_hub.update_info(f'cumulative_len_rank_{rank}', cumulative_len)

        max_seqlen = data.pop('max_seqlen')
        message_hub.update_info(f'max_seqlen_rank_{rank}', max_seqlen)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        rank = dist.get_rank()
        message_hub = MessageHub.get_instance('varlen_attn_args')
        message_hub.update_info(f'cumulative_len_rank_{rank}', None)
        message_hub.update_info(f'max_seqlen_rank_{rank}', None)
