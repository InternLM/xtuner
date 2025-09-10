from math import exp
import random
from xtuner.v1.datasets.packing import HardPackDataset
from xtuner.v1.datasets import JsonlDataset
from itertools import chain
import numpy as np

import pytest



class _FakeJsonlDataset:
    def __init__(self, sample_length: int, random_length: bool = False):
        self.sample_length = sample_length
        self.random_length = random_length

        if self.random_length:
            length = random.randint(1, self.sample_length) if self.sample_length else random.randint(1, 512)
        else:
            length = self.sample_length if self.sample_length else 512

        self.input_ids_list = [[random.randint(0, 10000) for _ in range(length)] for _ in range(1000)]
        self.labels_list = self.input_ids_list.copy()
        self.num_tokens = np.array(list(len(i) for i in self.input_ids_list))

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": self.input_ids_list[index],
            "labels": self.labels_list[index],
            "num_tokens": self.num_tokens[index],
        }

    def __len__(self) -> int:
        return len(self.input_ids_list)


@pytest.mark.parametrize(
    "random_length,global_pack",
    [
        (True, True),
        (False, False),
    ]
)
def test_hard_pack_dataset(random_length, global_pack):
    pack_max_length = 256
    datasets = []
    for _ in range(5):
        datasets.append(_FakeJsonlDataset(1024, random_length=random_length))

    hard_pack_dataset = HardPackDataset(datasets, pack_max_length=pack_max_length, seed=10, global_pack=global_pack)

    input_ids_list = []
    for pack_sample in hard_pack_dataset:
        for sample in pack_sample:
            input_ids_list.append(sample["input_ids"])

    total_input_ids = list(chain(*input_ids_list))

    random_gen = random.Random(10)
    if global_pack:
        expected_sample = [[i["input_ids"] for i in d] for d in datasets]
        expected_sample = list(chain.from_iterable(expected_sample))
        random_gen.shuffle(expected_sample)
        expected_inputs_ids = list(chain.from_iterable(expected_sample))
    else:
        expected_sample = [[i["input_ids"] for i in d] for d in datasets]
        for sample in expected_sample:
            random_gen.shuffle(sample)
        expected_inputs_ids = list(chain.from_iterable(chain.from_iterable(expected_sample)))

    expected_length = len(expected_inputs_ids) // pack_max_length * pack_max_length
    assert len(total_input_ids) == expected_length

    for i in range(expected_length):
        try:
            assert total_input_ids[i] == expected_inputs_ids[i]
        except Exception as e:
            __import__('ipdb').set_trace()
