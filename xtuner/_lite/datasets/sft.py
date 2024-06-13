from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.utils.import_utils import is_flash_attn_2_available

from xtuner._lite import get_logger
from xtuner._lite.chat import ChatMessages
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from .base import BaseTrainDataset

logger = get_logger()


class FinetuneDataset(BaseTrainDataset):
    """Dataset for Text Data Training.

    The required input dataset format for each piece of data is as follows：
    ```
    {
        "messages" : [
            {"role" : "user", "content" : "Hello"},
            {"role" : "assistant", "content" : "Hello!"},
            {"role" : "user", "content" : "Who are you?"},
            {"role" : "assistant", "content" : "I'm an AI assistant."}
        ]
    }
    ```
    For a more detailed introduction on dataset format, you can refer to the
    XTuner documentation.

    For common dataset formats, we also provide corresponding conversion
    scripts, refer to the XTuner documentation.

    Note:
       If your dataset is not in the aforementioned format, and it's
       inconvenient to convert to that format, you can inherit `TextDataset`
       and override the `tokenize_dataset` method if you want to use the
       high-performance data processing features of XTuner.


    Args:
        tokenizer (_TokenizerType):
            The original data is a dict composed of strings, which needs
            to be tokenized before it can be trained. The tokenizer must be
            consistent with the one used in the model being trained.
        chat_template (dict | ChatTemplate):
            Different models correspond to different chat template. If it is
            a dict, it will be considered in the style of MMEngine config and
            will be built accordingly. Ultimately, a ChatTemplate instance
            will be obtained.
        sample_ratio (float):
            Control over-sampling or under-sampling of the dataset, 2 means
            over-sampling the dataset to double its original size,
            0.5 means under-sampling the dataset to half its original size.
            Default is 1.
        max_length (int):
            The maximum length of a single data in the dataset. When
            `pack_to_max_length` is False, data exceeding the `max_length`
            will be truncated, only preserving the first `max_length` tokens.
            Default is 2048.
        pack_to_max_length (bool):
            If True, multiple data will be packed to form a new dataset. Each
            data in the packed dataset has a length of `max_length`. Default
            is True.
        num_proc (int):
            The number of threads used when processing dataset. Default is 8.
        cache_dir (str):
            The cache path of the dataset. When the path points to an existing
            cached dataset, it will directly load the cached dataset, and the
            settings of `data_dir` and `data_files` will become invalid. When
            the path does not point to a cached dataset, the processed data
            will be cached to this path. Default is None.
        data_files (List[str]):
            The paths of several json files. When `data_dir` is None, it will
            directly load files according to the addresses in `data_files`;
            when `data_dir` is not None, the file paths in `data_files` should
            be relative paths under `data_dir`. When `cache_dir` is not None,
            and `cache_dir` points to a cached dataset, the settings of
            `data_files` become invalid. Default is None.
        data_dir (str):
            When `data_files` is None, it will load all json files in
            `data_dir` and its several tiers of sub-directories; when
            `data_files` is not None, only json files in `data_files` are
            loaded. When `cache_dir` is not None, and `cache_dir` points
            to a  cached dataset, the setting of `data_dir` become invalid.
            Default is None.
    """

    def tokenize_dataset(self, dataset: List[dict]) -> List[dict]:
        """Tokenize the dataset and convert it into a trainable format.

        In the `tokenize_dataset` method, you need to define how to convert
        the raw data to the correct prompts (taking note of chat templates)
        and tokenize them; you need to define the label for each token, with
        the labels of the part that doesn't need to calculate loss set to -100.

        The labels don't need to be offset, it will be offset when the model
        calculates loss, meaning the label of each token should be itself or
        -100.


         Args:
             dataset (List[dict]):  The untokenized dataset.

         Note:
             The input must be a native Python list of dict, not
             `datasets.Dataset`, otherwise multithreaded data filtering will be
             slow.

         Returns:
             List[dict]:
                 Each dict in the list must contain three keys: `input_ids`,
                 `labels` and `num_tokens`.
                 `input_ids` and `labels` are lists of int, and they should
                 have equal lengths.
                 `num_tokens` is an integer, the length of `input_ids`.
        """

        def openai_to_training(item: dict) -> Dict:

            data = ChatMessages.from_dict(item)
            data = data.tokenize(self.tokenizer, self.chat_template)

            return data

        dataset = self.multi_thread_map(openai_to_training, dataset,
                                        'Tokenize Dataset')

        return dataset

    @classmethod
    def dataloader_collate_fn(cls, instances):

        pad_index = DEFAULT_PAD_TOKEN_INDEX

        input_ids = []
        labels = []
        position_ids = []
        chunk_sizes = []

        for data in instances:
            input_ids.append(torch.LongTensor(data['input_ids']))
            labels.append(torch.LongTensor(data['labels']))
            if 'position_ids' in data:
                position_ids.append(torch.IntTensor(data['position_ids']))
            else:
                position_ids = torch.arange(0, len(data['input_ids']))

            if 'chunk_sizes' in data:
                chunk_sizes.extend(data['chunk_sizes'])

        if len(instances) > 1:
            if is_flash_attn_2_available():
                input_ids = torch.cat(input_ids, dim=0)
                labels = torch.cat(labels, dim=0)
                position_ids = torch.cat(position_ids, dim=0)
                attention_mask = torch.ones_like(input_ids, dtype=bool)
            else:
                input_ids = pad_sequence(
                    input_ids, batch_first=True, padding_value=pad_index)
                labels = pad_sequence(
                    labels, batch_first=True, padding_value=IGNORE_INDEX)
                position_ids = pad_sequence(
                    position_ids, batch_first=True, padding_value=-1)

                attention_mask = (position_ids >= 0).bool()
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            position_ids = torch.stack(position_ids)
            attention_mask = torch.ones_like(input_ids, dtype=bool)

        # TODO support sp
        data_dict = {
            'input_ids':
            input_ids,
            'attention_mask':
            attention_mask,
            'labels':
            labels,
            'position_ids':
            position_ids,
            'chunk_sizes':
            torch.IntTensor(chunk_sizes) if len(chunk_sizes) else None
        }
        return data_dict
