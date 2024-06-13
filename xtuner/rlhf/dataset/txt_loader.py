""" Finetuning dataset. """
import random
from typing import List
import numpy as np
from dataclasses import dataclass
from torch.utils.data import IterableDataset, DataLoader, RandomSampler
from .base import MultiSourceDatset, InfiniteDataset


@dataclass
class Message:
    message: List[dict]
    sys_meta: str = "default"
    rm_meta: str = "default"
    token_ids: List[int] = None
    mes_type: str = "prompt"


class TxtMessageDataset(IterableDataset):
    """ Create sequences from dataset.
    Args:
        sample_strategy (str) ["in_batch", "in_data"]: "in_batch": sample data by ratio for every single training batch
                                                   "in_data": merge all data by ratio first and then sample training batch
    """
    def __init__(self,
                 prompt_datasets: list[str] = None,
                 pretrain_datasets: list[str] = None,
                 tokenizer=None,
                 max_prompt_len: int = 4096,
                 max_pretrain_len: int = 4096,
                 prompt_samples_each_epoch: int = 64,
                 pretrain_samples_each_epoch: int = 0,
                 random_seed: int = 110,
                 sample_strategy: str = "in_batch",
                 ratio_within_datasets: bool = True,
                 **kwargs
                 ):
        assert sample_strategy in ["in_batch", "in_data"], f"sample_strategy should in ['in_batch', 'in_data'], but got {sample_strategy}"
        self.sample_strategy = sample_strategy
        assert prompt_datasets is not None, "[Data error] Specify your data task config"
        self.tokenizer = tokenizer
        assert self.tokenizer.chat_template is not None, "Make sure tokenizer has chat_template."

        self.prompt_message_dataset = MultiSourceDatset(task_groups=prompt_datasets,
                                                    sub_dataset_type="file",
                                                    tokenizer=self.tokenizer,
                                                    ratio_within_datasets=ratio_within_datasets
                                                    )
        if pretrain_samples_each_epoch is not None and pretrain_samples_each_epoch > 0:
            assert pretrain_datasets is not None, f"[PT DATA error] samples num {pretrain_samples_each_epoch}, while pretrain_datasets is None"
            self.pt_message_dataset = MultiSourceDatset(task_groups=pretrain_datasets,
                                                        sub_dataset_type="file",
                                                        tokenizer=self.tokenizer,
                                                        ratio_within_datasets=ratio_within_datasets
                                                        )
            self.pretrain_samples_each_epoch = pretrain_samples_each_epoch
        else:
            self.pt_message_dataset = None
            self.pretrain_samples_each_epoch = 0
        self.prompt_samples_each_epoch = prompt_samples_each_epoch

        self.max_prompt_len = max_prompt_len
        self.max_pretrain_len = max_pretrain_len
        self.num_samples_each_epoch = self.pretrain_samples_each_epoch + self.prompt_samples_each_epoch
        
        self.random_seed = random_seed
        self.rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if self.sample_strategy == "in_batch":
            self._init_in_batch()
        elif self.sample_strategy == "in_data":
            self._init_in_data()
        else:
            raise NotImplementedError(f"sample_strategy should in ['in_batch', 'in_data'], but got {sample_strategy}")

        self.epoch_index = 0

    def _init_in_data(self):
        print(f"========================= Init in data sampler =========================")
        if self.pretrain_samples_each_epoch != 0:
            assert hasattr(self.pt_message_dataset, "all_dataset")
            pt_sampler = RandomSampler(self.pt_message_dataset.all_dataset)
            self.pt_dataloader = iter(DataLoader(
                self.pt_message_dataset.all_dataset, collate_fn=lambda x: x, sampler=pt_sampler, batch_size=self.pretrain_samples_each_epoch
            ))
            print(f"[PT data] pretrain data per epoch: {self.pretrain_samples_each_epoch}")

        assert hasattr(self.prompt_message_dataset, "all_dataset")
        prompt_sampler = RandomSampler(self.prompt_message_dataset.all_dataset)
        self.prompt_dataloader = iter(DataLoader(
            self.prompt_message_dataset.all_dataset, collate_fn=lambda x: x, sampler=prompt_sampler, batch_size=self.prompt_samples_each_epoch
        ))

        print(f"[Prompt data] prompt data per epoch: {self.prompt_samples_each_epoch}")
        print(f"[Txt] Training dataset initialized, random seed {self.random_seed}.\n")
    
    def yield_in_data(self):
        print(f"========================= yield data from data sampler =========================")
        batch_sequence = []
        prompt_sequence, pretrain_sequence = [], []
        if self.pretrain_samples_each_epoch != 0:
            pretrain_batch_messages = next(self.pt_dataloader)
            for index, message in enumerate(pretrain_batch_messages):
                sequence = self._postprocess_sequence(message, mes_type="pretrain")
                if sequence is not None:
                    assert sequence.mes_type == 'pretrain', f"Data type should be pretrain, but get {sequence.mes_type}"
                    pretrain_sequence.append(sequence)
                    if len(pretrain_sequence) == self.pretrain_samples_each_epoch:
                        break
        assert len(pretrain_sequence) == self.pretrain_samples_each_epoch, f"{len(pretrain_sequence)} != {self.pretrain_samples_each_epoch}"

        prompt_batch_messages = next(self.prompt_dataloader)
        for index, message in enumerate(prompt_batch_messages):
            if message is None:
                continue
            sequence = self._postprocess_sequence(message, mes_type="prompt")
            if sequence is not None:
                assert sequence.mes_type == 'prompt', f"Data type should be prompt. but get {sequence.mes_type}"
                prompt_sequence.append(sequence)
                if len(prompt_sequence) == self.prompt_samples_each_epoch:
                    break
        # TODO, len(prompt_sequence) < self.prompt_samples_each_epoch, random sample from chosen data
        if len(prompt_sequence) < self.prompt_samples_each_epoch:
            missed = self.prompt_samples_each_epoch - len(prompt_sequence)
            print(f"[Warning] {missed} dirty data, use {missed} data from sampled data...")
            for i in range(missed):
                prompt_sequence.append(prompt_sequence[i])

        assert len(prompt_sequence) == self.prompt_samples_each_epoch, f"{len(prompt_sequence)} == {self.prompt_samples_each_epoch}"

        print(f"prepare TxtMessageDataset done: {len(prompt_sequence)} prompt & {len(pretrain_sequence)} pretrain, for epoch {self.epoch_index}.")
        batch_sequence = prompt_sequence + pretrain_sequence
        assert len(batch_sequence) == self.num_samples_each_epoch, "[Epoch {self.epoch_index}] Wrong data len"
        return batch_sequence

    def _init_in_batch(self):
        print(f"========================= Init in batch sampler =========================")
        samples_cnts = []
        pt_data_len = 0
        if self.pretrain_samples_each_epoch != 0:
            for task in self.pt_message_dataset._task_group:
                task["target_num_each_epoch"] = int(task["prob"] * self.pretrain_samples_each_epoch + 0.5) + 1
                inner_dataset = InfiniteDataset(task["dataset"], self.rng)
                task["iterator"] = iter(inner_dataset)
                samples_cnts.append(task["target_num_each_epoch"])
                print(f"[Pretrain data] {task['filepath']}: task prob: {task['prob']}, "
                        f"ori number of messages: {len(inner_dataset.data)}, "
                        f"target_num_each_epoch: {task['target_num_each_epoch']}")
            pt_data_len = sum(samples_cnts)
            # TODO
            assert pt_data_len >= self.pretrain_samples_each_epoch, f"Make sure there are enough pretrain datas, {pt_data_len} >= {self.pretrain_samples_each_epoch}"
            print(f"[PT data] pretrain data per epoch: {self.pretrain_samples_each_epoch}, sampled {pt_data_len}")

        for task in self.prompt_message_dataset._task_group:
            task["target_num_each_epoch"] = int(task["prob"] * self.prompt_samples_each_epoch + 0.5) + 1
            inner_dataset = InfiniteDataset(task["dataset"], self.rng)
            task["iterator"] = iter(inner_dataset)
            samples_cnts.append(task["target_num_each_epoch"])
            print(f"{task['filepath']}: task prob: {task['prob']}, "
                    f"ori number of messages: {len(inner_dataset.data)}, "
                    f"target_num_each_epoch: {task['target_num_each_epoch']}")
        assert (sum(samples_cnts) - pt_data_len) >= self.prompt_samples_each_epoch, "Make sure there are enough prompt datas"
        print(f"[Prompt data] prompt data per epoch: {self.prompt_samples_each_epoch}, sampled: {sum(samples_cnts) - pt_data_len}")

        assert sum(samples_cnts) >= self.num_samples_each_epoch, "[Dataset init] sample num error"
        # if sum(samples_cnts) <= self.num_samples_each_epoch:
        #     print(f"[Txt loader] Warning!!! sample nums {sum(samples_cnts)} <= samples {self.num_samples_each_epoch}")
        print(f"[Txt] Training dataset initialized, random seed {self.random_seed}.\n")
    
    def yield_in_batch(self):
        print(f"========================= yield data from batch sampler =========================")
        batch_sequence = []
        prompt_sequence, pretrain_sequence = [], []

        # epoch_rng only use in this epoch.
        epoch_rng = np.random.RandomState(self.epoch_index)
        # prepare epoch data
        # print(f"prepare TxtMessageDataset for epoch {self.epoch_index}...")
        if self.pretrain_samples_each_epoch != 0 :
            pretrain_batch_messages = []
            for task in self.pt_message_dataset._task_group:
                messages = []
                for _ in range(task["target_num_each_epoch"]):
                    messages.append(next(task["iterator"]))
                print(f"[Pretrain] prepare {len(messages)} data from {task['filepath']}")
                epoch_rng.shuffle(messages)
                pretrain_batch_messages.extend(messages)
                # if len(pretrain_batch_messages) == self.pretrain_samples_each_epoch:
                #     break
            epoch_rng.shuffle(pretrain_batch_messages)
            for index, message in enumerate(pretrain_batch_messages):
                sequence = self._postprocess_sequence(message, mes_type="pretrain")
                if sequence is not None:
                    assert sequence.mes_type == 'pretrain', f"Data type should be pretrain, but get {sequence.mes_type}"
                    pretrain_sequence.append(sequence)
                    if len(pretrain_sequence) == self.pretrain_samples_each_epoch:
                        break
        assert len(pretrain_sequence) == self.pretrain_samples_each_epoch, f"{len(pretrain_sequence)} != {self.pretrain_samples_each_epoch}"

        prompt_batch_messages = []
        for task in self.prompt_message_dataset._task_group:
            messages = []
            for _ in range(task["target_num_each_epoch"]):
                messages.append(next(task["iterator"]))
            print(f"[Prompt] prepare {len(messages)} data from {task['filepath']}")
            epoch_rng.shuffle(messages)
            prompt_batch_messages.extend(messages)
        epoch_rng.shuffle(prompt_batch_messages)
        for index, message in enumerate(prompt_batch_messages):
            sequence = self._postprocess_sequence(message, mes_type="prompt")
            if sequence is not None:
                assert sequence.mes_type == 'prompt', f"Data type should be prompt. but get {sequence.mes_type}"
                prompt_sequence.append(sequence)
                if len(prompt_sequence) == self.prompt_samples_each_epoch:
                    break
        assert len(prompt_sequence) == self.prompt_samples_each_epoch, f"{len(prompt_sequence)} == {self.prompt_samples_each_epoch}"

        print(f"prepare TxtMessageDataset done: {len(prompt_sequence)} prompt & {len(pretrain_sequence)} pretrain, for epoch {self.epoch_index}.")
        batch_sequence = prompt_sequence + pretrain_sequence
        assert len(batch_sequence) == self.num_samples_each_epoch, "[Epoch {self.epoch_index}] Wrong data len"
        return batch_sequence

    def __iter__(self):
        while True:
            if self.sample_strategy == "in_batch":
                yield self.yield_in_batch()
            elif self.sample_strategy == "in_data":
                yield self.yield_in_data()

            self.epoch_index += 1

    def _postprocess_sequence(self, message, mes_type=None):
        """Post process sequence: tokenization & truncation."""
        message_data = message['data']
        new_meaasage_data = []
        if mes_type == "prompt":
            for _ in reversed(range(len(message_data))):
                if message_data[_]["role"] == "user":
                    new_meaasage_data = message_data[:_ + 1]
                    break
            assert new_meaasage_data[-1]["role"] == "user", f"prompt data last role must user, {new_meaasage_data}"
            token_ids = self.tokenizer.apply_chat_template(new_meaasage_data, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            if token_ids.shape[-1] <= 4 or token_ids.shape[-1] > self.max_prompt_len:
                # TODO truncation??
                # raise RuntimeError(f"token_ids is too long: {token_ids.shape[-1]}")
                print(f"[TXT Loader] Warning, {mes_type} message {message} is too short or long, skipped...")
                return None
        elif mes_type == "pretrain":
            for _ in reversed(range(len(message_data))):
                if message_data[_]["role"] == "assistant":
                    new_meaasage_data = message_data[:_ + 1]
                    break
            assert new_meaasage_data[-1]["role"] == "assistant", f"pretrain data last role must assistant, {new_meaasage_data}"
            token_ids = self.tokenizer.apply_chat_template(new_meaasage_data, tokenize=True, add_generation_prompt=False, return_tensors="pt")

            if token_ids.shape[-1] <= 4 or token_ids.shape[-1] > self.max_pretrain_len:
                # TODO truncation??
                # raise RuntimeError(f"token_ids is too long: {token_ids.shape[-1]}")
                print(f"[TXT Loader] Warning, {mes_type} message {message} is too short or long, skipped...")
                return None
        return Message(message=new_meaasage_data,
                       token_ids=token_ids,
                       sys_meta=message['sys_meta'],
                       rm_meta=message['rm_meta'],
                       mes_type=mes_type)
