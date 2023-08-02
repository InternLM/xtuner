import copy
from itertools import chain

from mmchat.utils import IGNORE_INDEX


def encode_fn(example, tokenizer, max_length, input_with_labels=True):
    input_encode = tokenizer(
        f"{tokenizer.bos_token}{example['input']}", add_special_tokens=False)
    if input_with_labels:
        output_encode = tokenizer(
            f"{example['output']}{tokenizer.eos_token}",
            add_special_tokens=False)
        input_ids = input_encode['input_ids'] + output_encode['input_ids']
        labels = [IGNORE_INDEX] * len(
            input_encode['input_ids']) + copy.deepcopy(
                output_encode['input_ids'])
    else:
        input_ids = input_encode['input_ids']
        labels = [IGNORE_INDEX] * len(input_encode['input_ids'])
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}


class Concatenator:
    # https://github.com/facebookresearch/llama-recipes/blob/main/ft_datasets/utils.py

    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.residual = {'input_ids': [], 'labels': []}

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k]))
            for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(
            concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i:i + self.chunk_size]
                    for i in range(0, chunk_num *
                                   self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size):]
                for k, v in concatenated_samples.items()
            }
        else:
            result = {k: [v] for k, v in concatenated_samples.items()}
            self.residual = {k: [] for k in concatenated_samples.keys()}

        return result
