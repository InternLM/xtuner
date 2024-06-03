# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmengine import MessageHub
from mmengine.dist import get_rank
from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


class VarlenAttnArgsToMessageHubHook(Hook):

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: dict = None) -> None:
        rank = get_rank()
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
        rank = get_rank()
        message_hub = MessageHub.get_instance('varlen_attn_args')
        message_hub.update_info(f'cumulative_len_rank_{rank}', None)
        message_hub.update_info(f'max_seqlen_rank_{rank}', None)

    def before_val_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
        """
        rank = get_rank()
        message_hub = MessageHub.get_instance('varlen_attn_args')

        assert 'data' in data_batch.keys()
        data = data_batch['data']

        cumulative_len = data.pop('cumulative_len')
        assert len(cumulative_len) == 1
        cumulative_len = cumulative_len[0].cuda()
        message_hub.update_info(f'cumulative_len_rank_{rank}', cumulative_len)

        max_seqlen = data.pop('max_seqlen')
        message_hub.update_info(f'max_seqlen_rank_{rank}', max_seqlen)

    def after_val_iter(self,
                       runner,
                       batch_idx,
                       data_batch=None,
                       outputs=None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        rank = get_rank()
        message_hub = MessageHub.get_instance('varlen_attn_args')
        message_hub.update_info(f'cumulative_len_rank_{rank}', None)
        message_hub.update_info(f'max_seqlen_rank_{rank}', None)
