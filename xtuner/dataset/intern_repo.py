# Copyright (c) OpenMMLab. All rights reserved.
import itertools as it
import json
import mmap
import operator
import os
import threading
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from mmengine import print_log
from torch import distributed as dist
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from xtuner.registry import BUILDER


class JsonlDataset(torch.utils.data.Dataset):
    """

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "input_ids": List[int],
        "labels": List[int]
    }
    ```

    """

    def __init__(self, path: str, min_length=50):
        self.path = path
        self.threadlocal = threading.local()
        resolved_path = Path(path).resolve()
        self.resolved_path = resolved_path
        self.meta = Path(f'{resolved_path}.meta')

        # only build the cache in on the primary worker to prevent
        # overloading nfs
        assert os.path.exists(
            self.meta
        ), f'The cache file:{self.meta} is not found for file:{self.path}'
        try:
            with open(self.meta, 'rb') as f:
                meta = np.load(f)
        except Exception as e:
            print(f'Cannot load file {self.meta}...')
            raise e
        self.offsets = meta[:, 0]
        self.length = meta[:, -1]

        if min_length > 0:
            mask = self.length >= min_length
            self.offsets = self.offsets[mask]
            self.length = self.length[mask]

    def __getitem__(self, idx):
        f = self._get_mmap()
        position = self.offsets[idx]
        f.seek(position)
        item = f.readline().decode('utf-8')
        try:
            item = json.loads(item)
            item['input_ids'] = item['tokens']
            del item['tokens']
            labels = [x if x > 0 else -100 for x in item['input_ids']]
            item['input_ids'] = [abs(x) for x in item['input_ids']]
            item['labels'] = labels
            item['length'] = len(item['input_ids'])  # add a length info
        except Exception as err:
            raise json.decoder.JSONDecodeError(
                doc=self.path,
                pos=position,
                msg=(f'Error while loading JSONL line in file {self.path} '
                     f'at byte {position}. Contents of line:\n{item}\n{err}'),
            )
        return item

    def get_dataset_name(self):
        return str(self.resolved_path)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, 'handles'):
            with open(self.path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.threadlocal.handles = [f, mm]
                if self.path.endswith('.gz') or self.path.endswith(
                        '.bz') or self.path.endswith('.bz2'):
                    raise NotImplementedError(
                        'Compressed files are not supported because .seek() '
                        'would require rereading the entire file, making '
                        'performance too slow.')
        return self.threadlocal.handles[-1]

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != 'threadlocal':
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, 'handles'):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def __len__(self):
        # Virtual length of the dataset depends on the epoch number
        # if the number of documents is not perfectly divisible by the
        # data_subshard_count
        return len(self.offsets)


class PackedDataset(torch.utils.data.Dataset):
    """The class PackedDataset takes in a dataset and aggregates samples of
    different lengths together based on the packed_length.

    Args:
        dataset: The original dataset to pack.
        packed_length: The length of each packed sample. Default is 8192.
    """

    def __init__(self, dataset, packed_length: int = 8192, seed: int = 1024):
        self.dataset = dataset
        self.packed_length = packed_length
        if isinstance(dataset, JsonlDataset):
            self.length = dataset.length
        elif isinstance(dataset, Dataset):
            assert 'length' in dataset.column_names
            self.length = dataset['length']
        else:
            raise NotImplementedError
        self.seed = seed

        rng = np.random.RandomState(self.seed)
        shuffled_indices = np.arange(len(self.length))
        rng.shuffle(shuffled_indices)
        self.shuffled_indices = shuffled_indices.tolist()
        self.shuffled_samples_len = list(
            map(self.length.__getitem__, shuffled_indices))
        self.shuffled_accumulated_samples_len = list(
            it.accumulate(self.shuffled_samples_len, operator.add))
        self.num_tokens = sum(self.length)

    def __len__(self):
        return self.num_tokens // self.packed_length

    def search_sample_index(self, pack_idx: int = 0):
        assert pack_idx >= 0
        length_train = (pack_idx + 1) * self.packed_length
        sample_index = np.searchsorted(
            self.shuffled_accumulated_samples_len, length_train, side='left')
        return sample_index

    def mapping(self, pack_idx: int = 0):
        begin_sample_idx, begin_token_id = 0, 0
        if pack_idx > 0:
            begin_sample_idx = self.search_sample_index(pack_idx - 1)
            # The position where the previous packed data ends
            begin_token_id = self.shuffled_samples_len[begin_sample_idx] - (
                self.shuffled_accumulated_samples_len[begin_sample_idx]
                -  # noqa: W504,W503
                (pack_idx) * self.packed_length)
            if begin_token_id == self.shuffled_samples_len[begin_sample_idx]:
                begin_sample_idx += 1
                begin_token_id = 0

        end_sample_idx = self.search_sample_index(pack_idx)
        end_token_id = self.shuffled_samples_len[end_sample_idx] - (
            self.shuffled_accumulated_samples_len[end_sample_idx]
            -  # noqa: W504,W503
            (pack_idx + 1) * self.packed_length)
        return begin_sample_idx, begin_token_id, end_sample_idx, end_token_id

    def build_pack(self, begin_sample_idx: int, begin_token_id: int,
                   end_sample_idx: int, end_token_id: int):
        pack, cumulative_len, indexes, labels = [], [0], [], []

        while begin_sample_idx < end_sample_idx:
            sample_idx = self.shuffled_indices[begin_sample_idx]
            sample = self.dataset[sample_idx]
            chunk = sample['input_ids'][begin_token_id:]
            pack.extend(chunk)
            _labels = sample['labels'][begin_token_id:]
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            cumulative_len.append(cumulative_len[-1] + len(chunk))
            indexes.extend(list(range(len(chunk))))
            begin_sample_idx = begin_sample_idx + 1
            begin_token_id = 0

        sample_idx = self.shuffled_indices[end_sample_idx]
        sample = self.dataset[sample_idx]
        chunk = sample['input_ids'][begin_token_id:
                                    end_token_id]  # fragment of a sample
        _labels = sample['labels'][begin_token_id:end_token_id]
        pack.extend(chunk)
        assert len(_labels) == len(chunk), (_labels, chunk)
        labels.extend(_labels)
        cumulative_len.append(cumulative_len[-1] + len(chunk))
        indexes.extend(list(range(len(chunk))))

        out = {
            'input_ids': pack,
            'cumulative_len': cumulative_len,
            'indexes': indexes,
            'labels': labels
        }
        return out

    def __getitem__(self, item: int):
        pos_before, token_id_before, pos_after, token_id_after = self.mapping(
            item)
        return self.build_pack(pos_before, token_id_before, pos_after,
                               token_id_after)


def load_intern_repo_dataset(folder, min_length=0):
    assert os.path.exists(folder), f'{folder} does not exist.'
    datasets = []

    triples = [('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train', ['cn'], []), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn', ['chinese_sensitive_v1', 'code2question', 'code_library_10k_import_ds1000', 'long_alpaca', 'self_critique_answer_no_critique', 'self_critique_refine_answer', 'dolly_chat_format_safety_filtered_v1', 'firefly_split_chat_format', 'flan2022_sampling_each256_niv2_zh_chat_format', 'hallucination', 'self_critique_gen_qa', 'stackoverflow_selected_python_chat_format', 'zephyr_ultrachat_200k_filtered', 'know_saraswati_cot', 'pj_characters_x10', 'zephyr_ultrafeedback_clean_filtered', 'emoji_chat_format', 'gmath', 'kaggle_cn', 'kaggle_en', 'lab_info', 'lima_chat_format_safety_filtered_v1', 'merged_math', 'moss_emotion_v2', 'msagent', 'WizardLM', 'char_x10_chat_format', 'coigv03_01_chat_format_safety_filtered_v1', 'government_department_safety_filtered_v1', 'greeting_x10', 'leetcode_filter_chat_refined', 'puyu_stylize', 'data_reflow', 'flan_v2_official_chat_format_512_safety_filtered_v1', 'math_coder_v0_2', 'open_file', 'ruozhibax10', 'safety_response', 'share_gpt_v6_chat_format_safety_filtered_v1', 'slimorca_dedup', 'ultrafeedback_critique', 'chinese_poetry_10x', 'EvolCode_v4x_r2_0', 'gsm8k_chat', 'ministry_of_foreign_affairs_safety_filtered_v1', 'no_robots', 'openai_summary', 'poem_chat_format', 'rolebench_w_sys_filtered', 'state_council_policy_safety_filtered_v1', 'unnatural_instructions_chat_format', 'code_library_ds1000', 'gaokao_essay_safety_filtered_v1', 'long_qlora', 'moss_no_moss_chat_fudan_format_safety_filtered_v1', 'puyu_chat_format_v2', 'self_critique_qa', 'self_critique_refine_critique', 'toolbench_0830'], []), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/chinese_sensitive_v1', [], ['18k_sensitive_refinev2_pos2en1cn_qmark_aug_2x_insertpos_addmeta.bin.meta', 'red_team_chat_format_0808-0822_refine_2x.bin', 'red_team_chat_format_0808-0822_refine_2x.bin.meta', '18k_sensitive_refinev2_pos2en1cn_qmark_aug_2x_insertpos_addmeta.bin', 'sensitive_word_qa_5w.bin', 'sensitive_word_qa_5w.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/code2question', [], ['c_s_format_datum_code.bin', 'c_s_format_datum_code.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/code_library_10k_import_ds1000', [], ['code_library_10k_import_ds1000.bin', 'code_library_10k_import_ds1000.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/long_alpaca', [], ['LongAlpaca-12k.bin', 'LongAlpaca-12k.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/self_critique_answer_no_critique', [], ['base_train_20.bin', 'base_train_20.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/self_critique_refine_answer', [], ['critiques_train.bin.meta', 'critiques_train.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/dolly_chat_format_safety_filtered_v1', [], ['dolly.bin', 'dolly.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/firefly_split_chat_format', [], ['AncientPoem_chat.bin', 'Composition_chat.bin', 'Couplet_chat.bin', 'Cot_chat.bin', 'MusicComment_chat.bin', 'KeywordRecognition_chat.bin', 'Program_chat.bin.meta', 'StoryGeneration_chat.bin.meta', 'ProseGeneration_chat.bin.meta', 'OpenQA_chat.bin', 'StoryGeneration_chat.bin', 'ClassicalChinese_chat.bin', 'TextCorrection_chat.bin', 'ProseGeneration_chat.bin', 'Composition_chat.bin.meta', 'Summary_chat.bin.meta', 'ClassicalChinese_chat.bin.meta', 'Cot_chat.bin.meta', 'JinYongGeneration_chat.bin', 'AncientPoem_chat.bin.meta', 'NER_chat.bin.meta', 'MRC_chat.bin.meta', 'Couplet_chat.bin.meta', 'TextCorrection_chat.bin.meta', 'ProductDesc_chat.bin', 'Summary_chat.bin', 'TextMatching_chat.bin', 'Translation_chat.bin', 'LyricGeneration_chat.bin', 'Dictionary_chat.bin', 'SentimentAnalyze_chat.bin', 'NLI_chat.bin.meta', 'Program_chat.bin', 'NER_chat.bin', 'NLI_chat.bin', 'OpenQA_chat.bin.meta', 'ProductDesc_chat.bin.meta', 'MRC_chat.bin', 'TextMatching_chat.bin.meta', 'Dictionary_chat.bin.meta', 'MusicComment_chat.bin.meta', 'Translation_chat.bin.meta', 'LyricGeneration_chat.bin.meta', 'JinYongGeneration_chat.bin.meta', 'KeywordRecognition_chat.bin.meta', 'SentimentAnalyze_chat.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/flan2022_sampling_each256_niv2_zh_chat_format', [], ['chat_format_niv2_zh.bin', 'chat_format_niv2_zh.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/hallucination', [], ['event1_0_answer_with_doc_16384_zh_0_name.bin', 'event1_0_subjective_zh_0.bin', 'person2_0_subjective_zh_0.bin', 'person1_0_answer_with_doc_16384_zh_0_name.bin', 'person1_1_question_16384_zh_0.bin', 'thing1_0_answerable_16384_zh_0.bin', 'thing1_1_answer_with_doc_16384_zh_0_name.bin', 'location1_0_answer_with_doc_16384_zh_0_name.bin', 'person2_0_faithful_answer_zh_0_name.bin', 'person2_1_annotate_16384_zh_1_answer.bin.meta', 'event1_1_answer_with_doc_16384_zh_0_name.bin.meta', 'person1_0_subjective_zh_0.bin.meta', 'person2_0_faithful_answer_zh_0_name.bin.meta', 'event1_0_faithful_answer_zh_0_name.bin.meta', 'thing1_0_best_question_zh_0.bin.meta', 'event1_0_answerable_16384_zh_0.bin', 'thing1_1_question_16384_zh_0.bin', 'location1_1_question_16384_zh_0.bin', 'person1_1_best_question_zh_0.bin', 'event1_1_subjective_zh_0.bin', 'person2_0_answerable_16384_zh_0.bin', 'person2_1_subjective_zh_0.bin', 'event1_1_best_question_zh_0.bin', 'person1_0_faithful_answer_zh_0_name.bin', 'location1_0_faithful_answer_zh_0_name.bin.meta', 'person1_1_answer_with_doc_16384_zh_0_name.bin.meta', 'person1_1_faithful_answer_zh_0_name.bin.meta', 'person2_0_question_16384_zh_0.bin.meta', 'person2_0_best_question_zh_0.bin.meta', 'thing1_0_answerable_16384_zh_0.bin.meta', 'thing1_1_answer_with_doc_16384_zh_0_name.bin.meta', 'location1_0_answerable_16384_zh_0.bin.meta', 'location1_0_best_question_zh_0.bin.meta', 'location1_1_answerable_16384_zh_0.bin.meta', 'person1_0_question_16384_zh_0.bin.meta', 'thing1_1_answerable_16384_zh_0.bin.meta', 'thing1_1_faithful_answer_zh_0_name.bin.meta', 'location1_0_subjective_zh_0.bin', 'person1_0_best_question_zh_0.bin', 'person1_1_faithful_answer_zh_0_name.bin', 'event1_0_best_question_zh_0.bin', 'event1_0_annotate_16384_zh_1_answer.bin', 'event1_1_annotate_16384_zh_1_answer.bin', 'person1_0_question_16384_zh_0.bin', 'thing1_1_answerable_16384_zh_0.bin', 'person2_0_subjective_zh_0.bin.meta', 'location1_1_best_question_zh_0.bin.meta', 'location1_1_question_16384_zh_0.bin.meta', 'location1_1_subjective_zh_0.bin.meta', 'person2_1_best_question_zh_0.bin.meta', 'person2_1_answer_with_doc_16384_zh_0_name.bin.meta', 'event1_1_answerable_16384_zh_0.bin', 'person2_0_annotate_16384_zh_1_answer.bin', 'person2_1_annotate_16384_zh_1_answer.bin', 'person2_1_faithful_answer_zh_0_name.bin', 'thing1_0_question_16384_zh_0.bin', 'location1_1_subjective_zh_0.bin', 'location1_0_question_16384_zh_0.bin', 'person1_0_annotate_16384_zh_1_answer.bin', 'person1_1_annotate_16384_zh_1_answer.bin', 'person2_0_answer_with_doc_16384_zh_0_name.bin', 'person2_1_answerable_16384_zh_0.bin', 'event1_0_question_16384_zh_0.bin.meta', 'event1_0_subjective_zh_0.bin.meta', 'location1_1_annotate_16384_zh_1_answer.bin.meta', 'thing1_1_best_question_zh_0.bin.meta', 'thing1_1_question_16384_zh_0.bin.meta', 'thing1_0_annotate_16384_zh_1_answer.bin.meta', 'event1_1_annotate_16384_zh_1_answer.bin.meta', 'location1_0_answer_with_doc_16384_zh_0_name.bin.meta', 'person1_1_annotate_16384_zh_1_answer.bin.meta', 'thing1_1_subjective_zh_0.bin.meta', 'event1_0_question_16384_zh_0.bin', 'location1_1_best_question_zh_0.bin', 'person2_1_question_16384_zh_0.bin', 'thing1_0_subjective_zh_0.bin', 'location1_1_faithful_answer_zh_0_name.bin', 'event1_0_answer_with_doc_16384_zh_0_name.bin.meta', 'event1_1_faithful_answer_zh_0_name.bin.meta', 'person1_0_answerable_16384_zh_0.bin.meta', 'person1_0_best_question_zh_0.bin.meta', 'person1_1_subjective_zh_0.bin.meta', 'person2_1_faithful_answer_zh_0_name.bin.meta', 'person2_1_question_16384_zh_0.bin.meta', 'person1_1_answerable_16384_zh_0.bin.meta', 'person1_1_question_16384_zh_0.bin.meta', 'person2_0_answerable_16384_zh_0.bin.meta', 'person2_1_answerable_16384_zh_0.bin.meta', 'event1_1_best_question_zh_0.bin.meta', 'person1_1_answerable_16384_zh_0.bin', 'thing1_1_best_question_zh_0.bin', 'location1_1_answerable_16384_zh_0.bin', 'person2_1_best_question_zh_0.bin', 'thing1_1_subjective_zh_0.bin', 'event1_0_faithful_answer_zh_0_name.bin', 'person2_1_answer_with_doc_16384_zh_0_name.bin', 'thing1_1_faithful_answer_zh_0_name.bin', 'event1_1_answerable_16384_zh_0.bin.meta', 'location1_0_annotate_16384_zh_1_answer.bin.meta', 'thing1_0_faithful_answer_zh_0_name.bin.meta', 'person1_0_answer_with_doc_16384_zh_0_name.bin.meta', 'event1_0_answerable_16384_zh_0.bin.meta', 'thing1_0_answer_with_doc_16384_zh_0_name.bin.meta', 'thing1_1_annotate_16384_zh_1_answer.bin.meta', 'event1_0_annotate_16384_zh_1_answer.bin.meta', 'person1_0_annotate_16384_zh_1_answer.bin.meta', 'location1_1_faithful_answer_zh_0_name.bin.meta', 'person1_0_faithful_answer_zh_0_name.bin.meta', 'event1_1_faithful_answer_zh_0_name.bin', 'location1_0_annotate_16384_zh_1_answer.bin', 'location1_1_annotate_16384_zh_1_answer.bin', 'person1_1_answer_with_doc_16384_zh_0_name.bin', 'person2_0_question_16384_zh_0.bin', 'thing1_0_faithful_answer_zh_0_name.bin', 'event1_1_answer_with_doc_16384_zh_0_name.bin', 'person1_1_subjective_zh_0.bin', 'event1_1_question_16384_zh_0.bin', 'person2_0_best_question_zh_0.bin', 'thing1_0_annotate_16384_zh_1_answer.bin', 'thing1_0_answer_with_doc_16384_zh_0_name.bin', 'thing1_1_annotate_16384_zh_1_answer.bin', 'location1_1_answer_with_doc_16384_zh_0_name.bin', 'thing1_0_best_question_zh_0.bin', 'location1_0_subjective_zh_0.bin.meta', 'person2_0_annotate_16384_zh_1_answer.bin.meta', 'thing1_0_question_16384_zh_0.bin.meta', 'event1_1_question_16384_zh_0.bin.meta', 'person2_1_subjective_zh_0.bin.meta', 'person2_0_answer_with_doc_16384_zh_0_name.bin.meta', 'location1_0_faithful_answer_zh_0_name.bin', 'person1_0_answerable_16384_zh_0.bin', 'person1_0_subjective_zh_0.bin', 'location1_0_answerable_16384_zh_0.bin', 'location1_0_best_question_zh_0.bin', 'event1_0_best_question_zh_0.bin.meta', 'thing1_0_subjective_zh_0.bin.meta', 'person1_1_best_question_zh_0.bin.meta', 'event1_1_subjective_zh_0.bin.meta', 'location1_0_question_16384_zh_0.bin.meta', 'location1_1_answer_with_doc_16384_zh_0_name.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/self_critique_gen_qa', [], ['base_train_30.bin', 'base_train_30.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/stackoverflow_selected_python_chat_format', [], ['stackoverflow_selected_python_chat_format.bin.meta', 'stackoverflow_selected_python_chat_format.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/zephyr_ultrachat_200k_filtered', [], ['zephyr-ultrachat-200k_sft_test.bin.meta', 'zephyr-ultrachat-200k_sft_test.bin', 'zephyr-ultrachat-200k_sft_train.bin', 'zephyr-ultrachat-200k_sft_train.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/know_saraswati_cot', [], ['know_saraswati_cot.bin.meta', 'know_saraswati_cot.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/pj_characters_x10', [], ['pj_characters_x10.bin', 'pj_characters_x10.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/zephyr_ultrafeedback_clean_filtered', [], ['zephyr-ultrafeedback_sft_test_clean.bin', 'zephyr-ultrafeedback_sft_train_clean.bin', 'zephyr-ultrafeedback_sft_train_clean.bin.meta', 'zephyr-ultrafeedback_sft_test_clean.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/emoji_chat_format', [], ['emoji_chat_format.bin.meta', 'emoji_chat_format.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/gmath', [], ['gmath.bin', 'gmath.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/kaggle_cn', [], ['info.bin', 'info.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/kaggle_en', [], ['info.bin', 'info.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/lab_info', [], ['lab_info.bin.meta', 'lab_info.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/lima_chat_format_safety_filtered_v1', [], ['lima_chat_format.bin.meta', 'lima_chat_format.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/merged_math', [], ['merged_data_20231207.bin.meta', 'merged_data_20231207.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/moss_emotion_v2', [], ['moss_emotion.bin.meta', 'moss_emotion.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/msagent', [], ['chatml_train.bin', 'dev.bin.meta', 'train.bin.meta', 'train.bin', 'dev.bin', 'chatml_dev.bin', 'chatml_dev.bin.meta', 'chatml_train.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/WizardLM', [], ['alpaca_evol_instruct_70k.bin.meta', 'alpaca_evol_instruct_70k.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/char_x10_chat_format', [], ['char_x10_chat_format.bin', 'char_x10_chat_format.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/coigv03_01_chat_format_safety_filtered_v1', [], ['exam_chat.bin.meta', 'cmcc_safety_filterd.bin', 'human1_chat_format.bin', 'leetcode_chat_clean_v3.bin', 'human1_chat_format.bin.meta', 'leetcode_chat_clean_v3.bin.meta', 'cmcc_safety_filterd.bin.meta', 'exam_chat.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/government_department_safety_filtered_v1', [], ['zhengfu_qa_v3.bin', 'zhengfu_qa_v3.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/greeting_x10', [], ['greeting_x10.bin.meta', 'greeting_x10.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/leetcode_filter_chat_refined', [], ['leetcode_filter_chat_refined.bin', 'leetcode_filter_chat_refined.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/puyu_stylize', [], ['puyu_stylize_processd2.bin', 'puyu_stylize_processd2.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/data_reflow', [], ['both_bad_aug.bin.meta', 'both_bad_aug.bin', 'gpt4_better_aug.bin', 'both_good_aug.bin.meta', 'both_good_aug.bin', 'gpt4_better_aug.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/flan_v2_official_chat_format_512_safety_filtered_v1', [], ['niv2.bin', 'dialog.bin', 'flan2021.bin', 't0.bin', 'cot.bin', 'cot.bin.meta', 't0.bin.meta', 'dialog.bin.meta', 'flan2021.bin.meta', 'niv2.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/math_coder_v0_2', [], ['math_ci.bin.meta', 'math_ci_cp2.bin.meta', 'gsm8k_ci.bin', 'gsm8k_ci.bin.meta', 'gsm8k_ci_cp1.bin.meta', 'gsm8k_ci_cp2.bin', 'math_ci_cp1.bin', 'math_ci_cp1.bin.meta', 'gsm8k_ci_cp2.bin.meta', 'gsm8k_ci_cp1.bin', 'math_ci.bin', 'math_ci_cp2.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/open_file', [], ['file_open_instruction_chat.bin', 'file_open_instruction_chat.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/ruozhibax10', [], ['ruozhiba_aug.bin.meta', 'ruozhiba_aug.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/safety_response', [], ['safety_response.bin', 'safety_response.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/share_gpt_v6_chat_format_safety_filtered_v1', [], ['share_gpt_v6.bin.meta', 'share_gpt_v6.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/slimorca_dedup', [], ['slim_orca_dedup_filtered.bin.meta', 'slim_orca_dedup_filtered.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/ultrafeedback_critique', [], ['false_qa.bin', 'sharegpt.bin.meta', 'ultrachat.bin.meta', 'truthful_qa.bin', 'sharegpt.bin', 'flan.bin.meta', 'evol_instruct.bin', 'ultrachat.bin', 'evol_instruct.bin.meta', 'false_qa.bin.meta', 'flan.bin', 'truthful_qa.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/chinese_poetry_10x', [], ['chinese-poetry-10x.bin.meta', 'chinese-poetry-10x.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/EvolCode_v4x_r2_0', [], ['polishedcode-v4.x-r2.bin.meta', 'polishedcode-v4.x-r2.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/gsm8k_chat', [], ['train_socratic.bin', 'train_socratic.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/ministry_of_foreign_affairs_safety_filtered_v1', [], ['waijiaobu_qa_v3.bin.meta', 'waijiaobu_qa_v3.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/no_robots', [], ['no_robots_train_filtered.bin', 'no_robots_test_filtered.bin', 'no_robots_test_filtered.bin.meta', 'no_robots_train_filtered.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/openai_summary', [], ['tldr_3_filtered_train.bin.meta', 'tldr_3_filtered_train.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/poem_chat_format', [], ['poem_chat_format.bin.meta', 'poem_chat_format.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/rolebench_w_sys_filtered', [], ['rolebench-eng_role-generalization_general_train.bin', 'rolebench-zh_general_train.bin', 'rolebench-eng_role-generalization_role_specific_train.bin.meta', 'rolebench-eng_instruction-generalization_general_train.bin', 'rolebench-eng_role-generalization_general_train.bin.meta', 'rolebench-eng_instruction-generalization_role_specific_train.bin', 'rolebench-eng_instruction-generalization_role_specific_train.bin.meta', 'rolebench-eng_role-generalization_role_specific_train.bin', 'rolebench-eng_instruction-generalization_general_train.bin.meta', 'rolebench-zh_general_train.bin.meta', 'rolebench-zh_role_specific_train.bin.meta', 'rolebench-zh_role_specific_train.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/state_council_policy_safety_filtered_v1', [], ['guowuyuan_qa_v3.bin.meta', 'guowuyuan_qa_v3.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/unnatural_instructions_chat_format', [], ['full_data.bin', 'full_data.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/code_library_ds1000', [], ['code_library_ds1000.bin.meta', 'code_library_ds1000.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/gaokao_essay_safety_filtered_v1', [], ['gaokao_essay.bin', 'gaokao_essay.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/long_qlora', [], ['LongQLoRA-SFT-Data-39k.bin.meta', 'LongQLoRA-SFT-Data-39k.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/moss_no_moss_chat_fudan_format_safety_filtered_v1', [], ['moss_v1_honest.bin', 'moss_v1_harmless_en.bin', 'moss_v1_harmless_zh_china-related_gpt4_fix_qmark_aug.bin.meta', 'moss_v1_switching.bin', 'moss_v1_awesome_en.bin', 'moss_v1_code.bin.meta', 'moss_v1_rp.bin.meta', 'moss_v1_continue.bin.meta', 'moss_v1_harmless_zh_china-related_gpt4_fix_qmark_aug.bin', 'moss_v1_honest.bin.meta', 'moss_v1_awesome_zh.bin.meta', 'moss_v1_harmless_en.bin.meta', 'moss_v1_advice.bin', 'moss_v1_awesome_en.bin.meta', 'moss_v1_advice.bin.meta', 'moss_v1_continue.bin', 'moss_v1_switching.bin.meta', 'moss_v1_awesome_zh.bin', 'moss_v1_code.bin', 'moss_v1_rp.bin', 'moss_v1_harmless_zh_non-chinarelated.bin', 'moss_v1_writing.bin', 'moss_v1_harmless_zh_non-chinarelated.bin.meta', 'moss_v1_writing.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/puyu_chat_format_v2', [], ['puyu_chat_format_v2.bin', 'puyu_chat_format_v2.bin.meta']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/self_critique_qa', [], ['base_train_50.bin.meta', 'base_train_50.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/self_critique_refine_critique', [], ['helpfulness_train.bin.meta', 'helpfulness_train.bin']), ('/mnt/petrelfs/share_data/zhangwenwei/data/llm/delivery_ft-0.17/v0.17.0rc8_32k/training/chatml_llamav13_32k/train/cn/toolbench_0830', [], ['toolllama_G123_dfs_train.bin', 'toolllama_G123_dfs_eval.bin', 'toolllama_G1_retrieve_random_mix_chatml_train.bin', 'toolllama_G1_retrieve_random_mix_chatml_train.bin.meta', 'toolllama_G1_retrieve_random_mix_train.bin.meta', 'toolllama_G123_dfs_eval.bin.meta', 'toolllama_G123_dfs_train.bin.meta', 'toolllama_G1_retrieve_random_mix_chatml_test.bin', 'toolllama_G123_dfs_chatml_eval.bin', 'toolllama_G123_dfs_chatml_eval.bin.meta', 'toolllama_G1_retrieve_random_mix_train.bin', 'toolllama_G1_retrieve_random_mix_test.bin', 'toolllama_G123_dfs_chatml_train.bin', 'toolllama_G1_retrieve_random_mix_chatml_test.bin.meta', 'toolllama_G1_retrieve_random_mix_test.bin.meta', 'toolllama_G123_dfs_chatml_train.bin.meta'])]

    for root, dirs, files in triples:
        dirs.sort()
        print_log(f'Reading {root}...', logger='current')

        for fn in tqdm(
                sorted(files),
                total=len(files),
                leave=False,
                disable=dist.get_rank() != 0):
            if fn.endswith('.bin'):
                fp = os.path.join(root, fn)
                ds = JsonlDataset(fp, min_length=min_length)

                if len(ds) == 0:
                    continue
                datasets.append(ds)

    return datasets


def build_packed_dataset_rank0(dataset_cfg, packed_length=8192, seed=1024):
    if isinstance(dataset_cfg, dict):
        datasets = BUILDER.build(dataset_cfg)
    else:
        datasets = dataset_cfg

    if not isinstance(datasets, list):
        datasets = [datasets]

    packed_datasets = []

    for dataset in datasets:
        ds = PackedDataset(dataset, packed_length, seed=seed)
        packed_datasets.append(ds)

    dataset = ConcatDataset(datasets=packed_datasets)

    return dataset


def build_packed_dataset(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return build_packed_dataset_rank0(*args, **kwargs)

    if dist.get_rank() == 0:
        dataset = build_packed_dataset_rank0(*args, **kwargs)
        objects = [dataset]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def process_intern_repo_dataset(folder,
                                packed_length=8192,
                                min_length=0,
                                seed=1024):

    assert os.path.exists(folder), f'{folder} does not exist.'
    datasets = []
    if dist.get_rank() == 0:
        triples = [list(os.walk(folder, followlinks=True))]
    else:
        triples = [None]
    dist.broadcast_object_list(triples, src=0)
    triples = triples[0]

    for root, dirs, files in triples:
        dirs.sort()  # Let the folder need to be returned in a fixed order
        if dist.get_rank() == 0:
            print_log(f'Reading {root}...', logger='current')
        num_token_in_folder = 0

        for fn in tqdm(
                sorted(files),
                total=len(files),
                leave=False,
                disable=dist.get_rank() != 0):
            if fn.endswith('.bin'):
                fp = os.path.join(root, fn)
                ds = JsonlDataset(fp, min_length=min_length)

                if len(ds) == 0:
                    continue

                ds = PackedDataset(ds, packed_length, seed=seed)

                num_token_in_folder += len(ds) * packed_length
                datasets.append(ds)

    dataset = ConcatDataset(datasets=datasets)

    return dataset
