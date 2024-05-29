"""Finetuning dataset."""
import random
from dataclasses import dataclass
from typing import List

import numpy as np
from torch.utils.data import DataLoader, IterableDataset, RandomSampler

from .base import InfiniteDataset, MultiSourceDatset


@dataclass
class Message:
    message: List[dict]
    sys_meta: str = 'default'
    rm_meta: str = 'default'
    token_ids: List[int] = None
    mes_type: str = 'ppo'


class TxtMessageDataset(IterableDataset):
    """Create sequences from dataset.

    Args:
        sample_strategy (str) ["in_batch", "in_data"]:
        "in_batch":
        sample data by ratio for every single training batch
        "in_data":
        merge all data by ratio first and then sample training batch
    """

    def __init__(self,
                 ppo_datas: list[str] = None,
                 pt_datas: list[str] = None,
                 tokenizer=None,
                 max_seq_len: int = 4096,
                 num_samples_each_epoch: int = 64,
                 pt_data_samples: int = 0,
                 random_seed: int = 110,
                 sample_strategy: str = 'in_batch',
                 ratio_within_datas: bool = True,
                 **kwargs):

        assert sample_strategy in [
            'in_batch', 'in_data'
        ], f"sample_strategy should in ['in_batch', 'in_data'], but got {sample_strategy}"  # noqa: E501
        self.sample_strategy = sample_strategy
        assert ppo_datas is not None, '[Data error] Specify your data task config'  # noqa: E501
        self.tokenizer = tokenizer
        assert self.tokenizer.chat_template is not None, 'Make sure tokenizer has chat_template.'  # noqa: E501

        self.ppo_message_dataset = MultiSourceDatset(
            task_groups=ppo_datas,
            sub_dataset_type='file',
            tokenizer=self.tokenizer,
            ratio_within_datas=ratio_within_datas)
        if pt_data_samples is not None and pt_data_samples != 0:
            assert pt_datas is not None, f'[PT DATA error] samples num {pt_data_samples}, while pt_datas is None'  # noqa: E501
            self.pt_message_dataset = MultiSourceDatset(
                task_groups=pt_datas,
                sub_dataset_type='file',
                tokenizer=self.tokenizer,
                ratio_within_datas=ratio_within_datas)
            self.pt_data_per_epoch = pt_data_samples
            self.ppo_data_per_epoch = num_samples_each_epoch - self.pt_data_per_epoch  # noqa: E501
        else:
            self.pt_message_dataset = None
            self.pt_data_per_epoch = 0
            self.ppo_data_per_epoch = num_samples_each_epoch

        self.max_seq_len = max_seq_len
        self.num_samples_each_epoch = num_samples_each_epoch

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
                f"sample_strategy should in ['in_batch', 'in_data'], but got {sample_strategy}"  # noqa: E501
            )

        self.epoch_index = 0

    def _init_in_data(self):
        print(
            '========================= Init in data sampler ========================='  # noqa: E501
        )
        if self.pt_data_per_epoch != 0:
            assert hasattr(self.pt_message_dataset, 'all_dataset')
            pt_sampler = RandomSampler(self.pt_message_dataset.all_dataset)
            self.pt_dataloader = iter(
                DataLoader(
                    self.pt_message_dataset.all_dataset,
                    collate_fn=lambda x: x,
                    sampler=pt_sampler,
                    batch_size=self.pt_data_per_epoch))
            print(
                f'[PT data] pretrain data per epoch: {self.pt_data_per_epoch}')

        assert hasattr(self.ppo_message_dataset, 'all_dataset')
        prompt_sampler = RandomSampler(self.ppo_message_dataset.all_dataset)
        self.prompt_dataloader = iter(
            DataLoader(
                self.ppo_message_dataset.all_dataset,
                collate_fn=lambda x: x,
                sampler=prompt_sampler,
                batch_size=self.ppo_data_per_epoch))

        print(f'[PPO data] ppo data per epoch: {self.ppo_data_per_epoch}')
        print(
            f'[Txt] Training dataset initialized, random seed {self.random_seed}.\n'  # noqa: E501
        )

    def yield_in_data(self):
        print(
            '========================= yield data from data sampler ========================='  # noqa: E501
        )
        batch_sequence = []
        ppo_sequence, pt_sequence = [], []
        if self.pt_data_per_epoch != 0:
            pt_batch_messages = next(self.pt_dataloader)
            for index, message in enumerate(pt_batch_messages):
                sequence = self._postprocess_sequence(message, mes_type='pt')
                if sequence is not None:
                    assert sequence.mes_type == 'pt', f'Data type should be pt, but get {sequence.mes_type}'  # noqa: E501
                    pt_sequence.append(sequence)
                    if len(pt_sequence) == self.pt_data_per_epoch:
                        break
        assert len(
            pt_sequence
        ) == self.pt_data_per_epoch, f'{len(pt_sequence)} != {self.pt_data_per_epoch}'  # noqa: E501

        ppo_batch_messages = next(self.prompt_dataloader)
        for index, message in enumerate(ppo_batch_messages):
            sequence = self._postprocess_sequence(message, mes_type='ppo')
            if sequence is not None:
                assert sequence.mes_type == 'ppo', f'Data type should be ppo. but get {sequence.mes_type}'  # noqa: E501
                ppo_sequence.append(sequence)
                if len(ppo_sequence) == self.ppo_data_per_epoch:
                    break
        if len(ppo_sequence) < self.ppo_data_per_epoch:
            missed = self.ppo_data_per_epoch - len(ppo_sequence)
            print(
                f'[Warning] {missed} dirty data, use {missed} data from sampled data...'  # noqa: E501
            )
            for i in range(missed):
                ppo_sequence.append(ppo_sequence[i])

        assert len(
            ppo_sequence
        ) == self.ppo_data_per_epoch, f'{len(ppo_sequence)} == {self.ppo_data_per_epoch}'  # noqa: E501

        print(
            f'prepare TxtMessageDataset done: {len(ppo_sequence)} ppo & {len(pt_sequence)} pretrain, for epoch {self.epoch_index}.'  # noqa: E501
        )
        batch_sequence = ppo_sequence + pt_sequence
        assert len(
            batch_sequence
        ) == self.num_samples_each_epoch, '[Epoch {self.epoch_index}] Wrong data len'  # noqa: E501
        return batch_sequence

    def _init_in_batch(self):
        print(
            '========================= Init in batch sampler ========================='  # noqa: E501
        )
        samples_cnts = []
        pt_data_len = 0
        if self.pt_data_per_epoch != 0:
            for task in self.pt_message_dataset._task_group:
                task['target_num_each_epoch'] = int(
                    task['prob'] * self.pt_data_per_epoch + 0.5) + 1
                inner_dataset = InfiniteDataset(task['dataset'], self.rng)
                task['iterator'] = iter(inner_dataset)
                samples_cnts.append(task['target_num_each_epoch'])
                print(
                    f"[PT data] {task['filepath']}: task prob: {task['prob']}, "  # noqa: E501
                    f'ori number of messages: {len(inner_dataset.data)}, '
                    f"target_num_each_epoch: {task['target_num_each_epoch']}"
                )  # noqa: E501
            pt_data_len = sum(samples_cnts)
            assert pt_data_len >= self.pt_data_per_epoch, f'Make sure there are enough pretrain data, {pt_data_len} >= {self.pt_data_per_epoch}'  # noqa: E501
            print(
                f'[PT data] pretrain data per epoch: {self.pt_data_per_epoch}, sampled {pt_data_len}'  # noqa: E501
            )
        for task in self.ppo_message_dataset._task_group:
            task['target_num_each_epoch'] = int(
                task['prob'] * self.ppo_data_per_epoch + 0.5) + 1
            inner_dataset = InfiniteDataset(task['dataset'], self.rng)
            task['iterator'] = iter(inner_dataset)
            samples_cnts.append(task['target_num_each_epoch'])
            print(f"{task['filepath']}: task prob: {task['prob']}, "
                  f'ori number of messages: {len(inner_dataset.data)}, '
                  f"target_num_each_epoch: {task['target_num_each_epoch']}")
        assert (
            sum(samples_cnts) - pt_data_len
        ) >= self.ppo_data_per_epoch, 'Make sure there are enough ppo datas'
        print(
            f'[PPO data] ppo data per epoch: {self.ppo_data_per_epoch}, sampled: {sum(samples_cnts) - pt_data_len}'  # noqa: E501
        )

        if sum(samples_cnts) <= self.num_samples_each_epoch:
            print(
                f'[Txt loader] Warning!!! sample nums {sum(samples_cnts)} <= samples {self.num_samples_each_epoch}'  # noqa: E501
            )
        print(
            f'[Txt] Training dataset initialized, random seed {self.random_seed}.\n'  # noqa: E501
        )

    def yield_in_batch(self):
        print(
            '========================= yield data from batch sampler ========================='  # noqa: E501
        )
        batch_sequence = []
        ppo_sequence, pt_sequence = [], []

        # epoch_rng only use in this epoch.
        epoch_rng = np.random.RandomState(self.epoch_index)
        # prepare epoch data
        if self.pt_data_per_epoch != 0:
            pt_batch_messages = []
            for task in self.pt_message_dataset._task_group:
                messages = []
                for _ in range(task['target_num_each_epoch']):
                    messages.append(next(task['iterator']))
                print(
                    f"[PT] prepare {len(messages)} data from {task['filepath']}"  # noqa: E501
                )
                epoch_rng.shuffle(messages)
                pt_batch_messages.extend(messages)
            epoch_rng.shuffle(pt_batch_messages)
            for index, message in enumerate(pt_batch_messages):
                sequence = self._postprocess_sequence(message, mes_type='pt')
                if sequence is not None:
                    assert sequence.mes_type == 'pt', f'Data type should be pt, but get {sequence.mes_type}'  # noqa: E501
                    pt_sequence.append(sequence)
                    if len(pt_sequence) == self.pt_data_per_epoch:
                        break
        assert len(
            pt_sequence
        ) == self.pt_data_per_epoch, f'{len(pt_sequence)} != {self.pt_data_per_epoch}'  # noqa: E501

        ppo_batch_messages = []
        for task in self.ppo_message_dataset._task_group:
            messages = []
            for _ in range(task['target_num_each_epoch']):
                messages.append(next(task['iterator']))
            print(
                f"[PPO] prepare {len(messages)} data from {task['filepath']}")
            epoch_rng.shuffle(messages)
            ppo_batch_messages.extend(messages)
        epoch_rng.shuffle(ppo_batch_messages)
        for index, message in enumerate(ppo_batch_messages):
            sequence = self._postprocess_sequence(message, mes_type='ppo')
            if sequence is not None:
                assert sequence.mes_type == 'ppo', f'Data type should be ppo. but get {sequence.mes_type}'  # noqa: E501
                ppo_sequence.append(sequence)
                if len(ppo_sequence) == self.ppo_data_per_epoch:
                    break
        assert len(
            ppo_sequence
        ) == self.ppo_data_per_epoch, f'{len(ppo_sequence)} == {self.ppo_data_per_epoch}'  # noqa: E501

        print(
            f'prepare TxtMessageDataset done: {len(ppo_sequence)} ppo & {len(pt_sequence)} pretrain, for epoch {self.epoch_index}.'  # noqa: E501
        )
        batch_sequence = ppo_sequence + pt_sequence
        assert len(
            batch_sequence
        ) == self.num_samples_each_epoch, '[Epoch {self.epoch_index}] Wrong data len'  # noqa: E501
        return batch_sequence

    def __iter__(self):
        while True:
            if self.sample_strategy == 'in_batch':
                yield self.yield_in_batch()
            elif self.sample_strategy == 'in_data':
                yield self.yield_in_data()

            self.epoch_index += 1

    def _postprocess_sequence(self, message, mes_type='ppo'):
        """Post process sequence: tokenization & truncation."""
        message_data = message['data']
        new_meaasage_data = []
        if mes_type == 'ppo':
            for _ in reversed(range(len(message_data))):
                if message_data[_]['role'] == 'user':
                    new_meaasage_data = message_data[:_ + 1]
                    break
            assert new_meaasage_data[-1][
                'role'] == 'user', f'ppo data last role must user, {new_meaasage_data}'  # noqa: E501
            token_ids = self.tokenizer.apply_chat_template(
                new_meaasage_data,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt')
        elif mes_type == 'pt':
            for _ in reversed(range(len(message_data))):
                if message_data[_]['role'] == 'assistant':
                    new_meaasage_data = message_data[:_ + 1]
                    break
            assert new_meaasage_data[-1][
                'role'] == 'assistant', f'pretrain data last role must assistant, {new_meaasage_data}'  # noqa: E501
            token_ids = self.tokenizer.apply_chat_template(
                new_meaasage_data,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors='pt')
        if token_ids.shape[-1] <= 4 or token_ids.shape[-1] > self.max_seq_len:
            print(
                f'[TXT Loader] Warning, {mes_type} message {message} is too short or long, skipped...'  # noqa: E501
            )
            return None
        return Message(
            message=new_meaasage_data,
            token_ids=token_ids,
            sys_meta=message['sys_meta'],
            rm_meta=message['rm_meta'],
            mes_type=mes_type)
