"""Finetuning dataset."""
import random
from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler

from xtuner.rlhf.dataset.base import (InfiniteDataset,
                                      MultiSourceInBatchDatset,
                                      MultiSourceInDataDatset)


@dataclass
class Message:
    message: List[dict]
    sys_prompt: str = 'default'
    rm_prompt: str = 'default'
    token_ids: List[int] = None
    mes_type: str = 'prompt'


class MessageIter():
    """Create sequences from dataset.

    Args:
        sample_strategy (str) ["in_batch", "in_data"]:
            "in_batch": sample data by ratio for every single training batch
            "in_data": merge all data by ratio and then sample training batch
    """

    def __init__(self,
                 message_datasets: list[str] = None,
                 message_type: str = 'prompt',
                 tokenizer=None,
                 max_len: int = 4096,
                 samples_each_epoch: int = 0,
                 random_seed: int = 110,
                 sample_strategy: str = 'in_batch',
                 **kwargs):
        assert message_type in ['prompt', 'pretrain']
        assert sample_strategy in [
            'in_batch', 'in_data'
        ], ("`sample_strategy` should in ['in_batch', 'in_data'],"
            f' but got {sample_strategy}')
        if (message_datasets is None) or (samples_each_epoch == 0):
            logger.warning(f'message_datasets: {message_datasets}'
                           f' samples_each_epoch: {samples_each_epoch}.')
            self.message_datasets = None
            self.samples_each_epoch = 0
            return None
        assert message_datasets is not None
        self.message_type = message_type
        self.sample_strategy = sample_strategy
        self.tokenizer = tokenizer
        assert self.tokenizer.chat_template is not None, (
            'Make sure tokenizer has chat_template.')
        # message data
        self.message_datasets = message_datasets
        self.samples_each_epoch = samples_each_epoch
        self.max_len = max_len

        self.random_seed = random_seed
        self.rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if self.sample_strategy == 'in_batch':
            self._init_in_batch()
        elif self.sample_strategy == 'in_data':
            self._init_in_data()
        else:
            raise NotImplementedError(
                "sample_strategy should in ['in_batch', 'in_data'],"
                f' but got {sample_strategy}')
        logger.info(f'[MES_ITER] {self.message_type} dataset initialized, '
                    f'random seed {self.random_seed}, '
                    f'{self.samples_each_epoch} per epoch.\n')

        self.epoch_index = 0

    def _init_in_data(self):
        logger.info(f'Init {self.message_type} in data dataset ...')
        self.message_dataset = MultiSourceInDataDatset(
            task_groups=self.message_datasets, tokenizer=self.tokenizer)

        logger.info(f'Init {self.message_type} in data sampler ...')
        assert hasattr(self.message_dataset, 'all_dataset')
        mes_sampler = RandomSampler(self.message_dataset.all_dataset)
        self.mes_dataloader = iter(
            DataLoader(
                self.message_dataset.all_dataset,
                collate_fn=lambda x: x,
                sampler=mes_sampler,
                batch_size=self.samples_each_epoch))

    def yield_in_data(self):
        logger.info('yielding data from '
                    f'{self.message_type} in_data sampler ...')
        mes_sequence = []

        mes_batch_messages = next(self.mes_dataloader)
        for index, message in enumerate(mes_batch_messages):
            if message is None:
                continue
            sequence = self._postprocess_sequence(message)
            if sequence is not None:
                mes_sequence.append(sequence)
                if len(mes_sequence) == self.samples_each_epoch:
                    break
        # TODO, len(mes_sequence) < self.samples_each_epoch,
        # tmp: random sample from chosen data
        if len(mes_sequence) < self.samples_each_epoch:
            missed = self.samples_each_epoch - len(mes_sequence)
            logger.warning(
                f'[MES_ITER] {self.message_type} {missed} dirty data ...')
            for i in range(missed):
                mes_sequence.append(mes_sequence[i])

        assert len(
            mes_sequence
        ) == self.samples_each_epoch, \
            f'{len(mes_sequence)} == {self.samples_each_epoch}'

        assert len(mes_sequence) == self.samples_each_epoch
        logger.info(f'[Epoch {self.epoch_index}] '
                    f'sample {len(mes_sequence)} {self.message_type}')
        return mes_sequence

    def _init_in_batch(self):
        logger.info(f'Init {self.message_type} in batch dataset ...')
        self.message_dataset = MultiSourceInBatchDatset(
            task_groups=self.message_datasets, tokenizer=self.tokenizer)

        logger.info(f'Init {self.message_type} in batch sampler ...')
        samples_cnts = []
        for task in self.message_dataset._task_group:
            task['target_num_each_epoch'] = int(
                task['prob'] * self.samples_each_epoch + 0.5) + 1
            inner_dataset = InfiniteDataset(task['dataset'], self.rng)
            task['iterator'] = iter(inner_dataset)
            samples_cnts.append(task['target_num_each_epoch'])
            logger.info(
                f"[MES_ITER] {task['filepath']}: task prob: {task['prob']}"
                f' original number of messages: {len(inner_dataset.data)}'
                f" target_num_each_epoch: {task['target_num_each_epoch']}")
        assert sum(samples_cnts) >= self.samples_each_epoch

    def yield_in_batch(self):
        logger.info('yield data from '
                    f'{self.message_type} in_batch sampler ...')
        mes_sequence = []

        # epoch_rng only use in this epoch.
        epoch_rng = np.random.RandomState(self.epoch_index)
        # prepare epoch data
        mes_batch_messages = []
        for task in self.message_dataset._task_group:
            messages = []
            for _ in range(task['target_num_each_epoch']):
                messages.append(next(task['iterator']))
            logger.info(f'[MES_ITER] sample {len(messages)} '
                        f"{self.message_type} from {task['filepath']}")
            epoch_rng.shuffle(messages)
            mes_batch_messages.extend(messages)
        epoch_rng.shuffle(mes_batch_messages)
        for index, message in enumerate(mes_batch_messages):
            sequence = self._postprocess_sequence(message)
            if sequence is not None:
                mes_sequence.append(sequence)
                if len(mes_sequence) == self.samples_each_epoch:
                    break
        # TODO, len(mes_sequence) < self.samples_each_epoch,
        # tmp: random sample from chosen data
        if len(mes_sequence) < self.samples_each_epoch:
            missed = self.samples_each_epoch - len(mes_sequence)
            logger.warning(
                f'[MES_ITER] {self.message_type} {missed} dirty data ...')
            for i in range(missed):
                mes_sequence.append(mes_sequence[i])

        assert len(mes_sequence) == self.samples_each_epoch
        logger.info(f'[Epoch {self.epoch_index}] sample '
                    f'{len(mes_sequence)} {self.message_type}')

        return mes_sequence

    def __iter__(self):
        while True:
            if self.sample_strategy == 'in_batch':
                yield self.yield_in_batch()
            elif self.sample_strategy == 'in_data':
                yield self.yield_in_data()

            self.epoch_index += 1

    def _postprocess_sequence(self, message):
        """Post process sequence: tokenization & truncation."""
        message_data = message['data']
        new_meaasage_data = []
        if self.message_type == 'prompt':
            for _ in reversed(range(len(message_data))):
                if message_data[_]['role'] == 'user':
                    new_meaasage_data = message_data[:_ + 1]
                    break
            assert new_meaasage_data[-1]['role'] == 'user', \
                f'prompt data last role must user, {new_meaasage_data}'
            token_ids = self.tokenizer.apply_chat_template(
                new_meaasage_data,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt')
            if (token_ids.shape[-1] <= 4) or (token_ids.shape[-1] >
                                              self.max_len):
                # TODO truncation??
                logger.warning(
                    f'[MES_ITER] {self.message_type} message {message} '
                    'is too short or long, skipped.')
                return None
        elif self.message_type == 'pretrain':
            for _ in reversed(range(len(message_data))):
                if message_data[_]['role'] == 'assistant':
                    new_meaasage_data = message_data[:_ + 1]
                    break
            assert new_meaasage_data[-1]['role'] == 'assistant', \
                f'pretrain data last role must assistant, {new_meaasage_data}'
            token_ids = self.tokenizer.apply_chat_template(
                new_meaasage_data,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors='pt')

            if token_ids.shape[-1] <= 4 or token_ids.shape[-1] > self.max_len:
                # TODO truncation??
                logger.warning(
                    f'[MES_ITER] {self.message_type} message {message} '
                    'is too short or long, skipped.')
                return None
        return Message(
            message=new_meaasage_data,
            token_ids=token_ids,
            sys_prompt=message['sys_prompt'],
            rm_prompt=message['rm_prompt'],
            mes_type=self.message_type)
