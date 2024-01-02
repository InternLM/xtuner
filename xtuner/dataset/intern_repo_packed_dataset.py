# Copyright (c) OpenMMLab. All rights reserved.
import os

from datasets import concatenate_datasets, load_dataset, load_from_disk
from mmengine import print_log
from torch import distributed as dist
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import threading
import json
import mmap
import torch
from pathlib import Path
import numpy as np
import itertools as it
import operator
from copy import deepcopy
from torch.utils.data import ConcatDataset, DataLoader
from mmengine import ConfigDict
from mmengine import print_log
import logging


class JsonlDataset(torch.utils.data.Dataset):
    """

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "tokens": List[int],
    }
    ```

    Note that only the "tokens" key is used.
    """

    def __init__(self, path: str, min_length=50):
        self.path = path
        self.threadlocal = threading.local()
        resolved_path = Path(path).resolve()
        self.resolved_path = resolved_path
        self.meta = Path(f"{resolved_path}.meta")

        # only build the cache in on the primary worker to prevent overloading nfs
        assert os.path.exists(self.meta), f"The cache file:{self.meta} is not found for file:{self.path}"
        try:
            with open(self.meta, "rb") as f:
                meta = np.load(f)
        except Exception as e:
            print(f"Cannot load file {self.meta}...")
            raise e
        self.offsets = meta[:, 0]
        self.lengths = meta[:, -1]

        if min_length > 0:
            mask = self.lengths >= min_length
            self.old_lengths = self.lengths.copy()
            self.old_length = len(self.offsets)
            self.offsets = self.offsets[mask]
            self.lengths = self.lengths[mask]

    def __getitem__(self, idx):
        f = self._get_mmap()
        position = self.offsets[idx]
        f.seek(position)
        item = f.readline().decode("utf-8")
        try:
            item = json.loads(item)
            item["length"] = len(item["tokens"])  # add a length info
        except Exception as err:
            raise json.decoder.JSONDecodeError(
                doc=self.path,
                pos=position,
                msg=(
                    f"Error while loading JSONL line in file {self.path} at byte "
                    f"{position}. Contents of line:\n{item}\n{err}"
                ),
            )
        return item

    def get_dataset_name(self):
        return str(self.resolved_path)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            with open(self.path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.threadlocal.handles = [f, mm]
                if self.path.endswith(".gz") or self.path.endswith(".bz") or self.path.endswith(".bz2"):
                    raise NotImplementedError(
                        "Compressed files are not supported because .seek() would require "
                        "rereading the entire file, making performance too slow."
                    )
        return self.threadlocal.handles[-1]

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "handles"):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def __len__(self):
        # Virtual length of the dataset depends on the epoch number if the number of documents
        # is not perfectly divisible by the data_subshard_count
        return len(self.offsets)


DEFAULT_SEED = 1024
class PackedDataset(torch.utils.data.Dataset):
    """
    The class PackedDataset takes in a dataset and aggregates samples of different
    lengths together based on the packed_length.

    Args:
        dataset: The original dataset to pack.
        packed_length: The length of each packed sample. Default is 8192.
    """

    def __init__(
        self,
        dataset,
        packed_length: int = 8192,
        seed: int = DEFAULT_SEED
    ):
        self.dataset = dataset
        self.packed_length = packed_length
        self.lengths = dataset.lengths
        self.seed = seed

        rng = np.random.RandomState(self.seed)
        shuffled_indices = np.arange(len(self.lengths))
        rng.shuffle(shuffled_indices)
        self.shuffled_indices = shuffled_indices
        self.shuffled_samples_len = list(map(self.lengths.__getitem__, shuffled_indices))
        self.shuffled_accumulated_samples_len = list(it.accumulate(self.shuffled_samples_len, operator.add))
        self.num_tokens = sum(self.lengths)
    
    def __len__(self):
        return self.num_tokens // self.packed_length
    
    def search_sample_index(self, pack_idx: int = 0):
        assert pack_idx >= 0
        length_train = (pack_idx + 1) * self.packed_length
        sample_index = np.searchsorted(self.shuffled_accumulated_samples_len, length_train, side="left")
        return sample_index
    
    def mapping(self, pack_idx: int = 0):
        begin_sample_idx, begin_token_id = 0, 0
        if pack_idx > 0:
            begin_sample_idx = self.search_sample_index(pack_idx - 1)
            begin_token_id = self.shuffled_samples_len[begin_sample_idx] - (
                self.shuffled_accumulated_samples_len[begin_sample_idx] - (pack_idx) * self.packed_length
            )  # 前一条packed数据结束的位置是那条数据的第几个token
            if begin_token_id == self.shuffled_samples_len[begin_sample_idx]:
                begin_sample_idx += 1
                begin_token_id = 0
        
        end_sample_idx = self.search_sample_index(pack_idx)
        end_token_id = self.shuffled_samples_len[end_sample_idx] - (self.shuffled_accumulated_samples_len[end_sample_idx] - (pack_idx + 1) * self.packed_length)
        return begin_sample_idx, begin_token_id, end_sample_idx, end_token_id
    
    def build_pack(self, begin_sample_idx: int, begin_token_id: int, end_sample_idx: int, end_token_id: int):
        pack, cu_seqlens, indexes, labels = [], [0], [], []

        while begin_sample_idx < end_sample_idx:
            sample_idx = self.shuffled_indices[begin_sample_idx]
            sample = self.dataset[sample_idx]
            chunk = sample["tokens"][begin_token_id:]
            pack.extend(chunk)
            _labels = deepcopy(chunk)
            # _labels = list(_labels[1:]) + [-100]
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            cu_seqlens.append(cu_seqlens[-1] + len(chunk))
            indexes.extend(list(range(len(chunk))))
            begin_sample_idx = begin_sample_idx + 1
            begin_token_id = 0

        sample_idx = self.shuffled_indices[end_sample_idx]
        sample = self.dataset[sample_idx]
        chunk = sample["tokens"][begin_token_id:end_token_id]  # fragement of a sample
        pack.extend(chunk)
        _labels = deepcopy(chunk)
        # if end_token_id == len(sample["tokens"]):
        #     _labels = list(_labels[1:]) + [-100]
        # else:
        #     if end_token_id > len(sample["tokens"]):
        #         print(f"end_token_id {end_token_id}, len of sample {len(sample['tokens'])}")
        #     _labels = list(_labels[1:]) + [sample["tokens"][end_token_id]]
        assert len(_labels) == len(chunk), (_labels, chunk)
        labels.extend(_labels)
        cu_seqlens.append(cu_seqlens[-1] + len(chunk))
        indexes.extend(list(range(len(chunk))))

        out = {"input_ids": pack, "cu_seqlens": cu_seqlens, "indexes": indexes, "labels": labels}
        return out
    
    def __getitem__(self, item: int):
        pos_before, token_id_before, pos_after, token_id_after = self.mapping(item)
        return self.build_pack(pos_before, token_id_before, pos_after, token_id_after)


def build_packed_dataset(folder, packed_length=8192, min_length=0):

    assert os.path.exists(folder), f"{folder} does not exist."
    datasets = []
    if dist.get_rank() == 0:
        triples = [list(os.walk(folder, followlinks=True))]
    else:
        triples = [None]
    dist.broadcast_object_list(triples, src=0)
    triples = triples[0]
    # triples = [('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train', ['cn'], []), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn', ['gsm8k_chat', 'rewoo', 'firefly_split_chat_format', 'gorilla_huggingface', 'gorilla_torchhub', 'safety_response', 'flan2022_sampling_each256_niv2_zh_chat_format', 'coigv03_01_chat_format_safety_filtered_v1', 'poem_chat_format', 'government_department_safety_filtered_v1', 'lima_chat_format_safety_filtered_v1', 'dolly_chat_format_safety_filtered_v1', 'zephyr_ultrafeedback_clean_filtered', 'puyu_stylize', 'gorilla_tensorflow', 'flan_v2_official_chat_format_512_safety_filtered_v1', 'char_x10_chat_format', 'openai_summary', 'math3000_calculate_solve_thought', 'greeting_x10', 'ruozhibax10', 'moss_emotion_v2', 'emoji_chat_format', 'ministry_of_foreign_affairs_safety_filtered_v1', 'moss_math_code_debug', 'state_council_policy_safety_filtered_v1', 'pj_characters_x10', 'moss_math_code', 'self_critique_answer_no_critique', 'self_critique_refine_critique', 'puyu_chat_format_v2', 'zephyr_ultrachat_200k_filtered', 'self_critique_qa', 'math3000_calculate_solve_wo_thought', 'lab_info', 'WizardLM', 'chinese_sensitive_v1', 'data_reflow', 'gsm8k_pot', 'toolbench_safety_filtered_v1', 'self_critique_gen_qa', 'gmath', 'gaokao_essay_safety_filtered_v1', 'chinese_poetry_10x', 'unnatural_instructions_chat_format', 'self_critique_refine_answer', 'moss_no_moss_chat_fudan_format_safety_filtered_v1', 'leetcode_filter_chat_refined', 'math3000_solve_wo_thought_wo_answer', 'EvolCode_v4x_r2_0', 'stackoverflow_selected_python_chat_format', 'rolebench_w_sys_filtered', 'share_gpt_v6_chat_format_safety_filtered_v1'], []), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/gsm8k_chat', [], ['train_socratic.bin.meta', 'train_socratic.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/rewoo', [], ['mix_insft_2k.bin', 'mix_insft_2k.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/firefly_split_chat_format', [], ['Program_chat.bin', 'ProductDesc_chat.bin.meta', 'JinYongGeneration_chat.bin', 'TextMatching_chat.bin', 'Couplet_chat.bin.meta', 'KeywordRecognition_chat.bin', 'NER_chat.bin', 'ProductDesc_chat.bin', 'MRC_chat.bin', 'Cot_chat.bin', 'ClassicalChinese_chat.bin', 'OpenQA_chat.bin.meta', 'ClassicalChinese_chat.bin.meta', 'AncientPoem_chat.bin', 'SentimentAnalyze_chat.bin', 'TextMatching_chat.bin.meta', 'JinYongGeneration_chat.bin.meta', 'Summary_chat.bin.meta', 'NER_chat.bin.meta', 'OpenQA_chat.bin', 'MRC_chat.bin.meta', 'MusicComment_chat.bin.meta', 'StoryGeneration_chat.bin', 'TextCorrection_chat.bin.meta', 'Summary_chat.bin', 'Composition_chat.bin', 'AncientPoem_chat.bin.meta', 'TextCorrection_chat.bin', 'Composition_chat.bin.meta', 'StoryGeneration_chat.bin.meta', 'Translation_chat.bin.meta', 'Dictionary_chat.bin.meta', 'ProseGeneration_chat.bin', 'Cot_chat.bin.meta', 'Translation_chat.bin', 'NLI_chat.bin', 'Dictionary_chat.bin', 'LyricGeneration_chat.bin', 'KeywordRecognition_chat.bin.meta', 'NLI_chat.bin.meta', 'ProseGeneration_chat.bin.meta', 'SentimentAnalyze_chat.bin.meta', 'Couplet_chat.bin', 'LyricGeneration_chat.bin.meta', 'MusicComment_chat.bin', 'Program_chat.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/gorilla_huggingface', [], ['huggingface_train.bin.meta', 'huggingface_train.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/gorilla_torchhub', [], ['torchhub_train.bin.meta', 'torchhub_train.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/safety_response', [], ['safety_response.bin', 'safety_response.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/flan2022_sampling_each256_niv2_zh_chat_format', [], ['chat_format_niv2_zh.bin', 'chat_format_niv2_zh.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/coigv03_01_chat_format_safety_filtered_v1', [], ['exam_chat.bin.meta', 'leetcode_chat_clean_v3.bin.meta', 'leetcode_chat_clean_v3.bin', 'cmcc_safety_filterd.bin', 'exam_chat.bin', 'cmcc_safety_filterd.bin.meta', 'human1_chat_format.bin', 'human1_chat_format.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/poem_chat_format', [], ['poem_chat_format.bin.meta', 'poem_chat_format.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/government_department_safety_filtered_v1', [], ['zhengfu_qa_v3.bin.meta', 'zhengfu_qa_v3.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/lima_chat_format_safety_filtered_v1', [], ['lima_chat_format.bin.meta', 'lima_chat_format.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/dolly_chat_format_safety_filtered_v1', [], ['dolly.bin', 'dolly.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/zephyr_ultrafeedback_clean_filtered', [], ['zephyr-ultrafeedback_sft_test_clean.bin', 'zephyr-ultrafeedback_sft_train_clean.bin', 'zephyr-ultrafeedback_sft_test_clean.bin.meta', 'zephyr-ultrafeedback_sft_train_clean.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/puyu_stylize', [], ['puyu_stylize_processd2.bin', 'puyu_stylize_processd2.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/gorilla_tensorflow', [], ['tensorflow_train.bin', 'tensorflow_train.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/flan_v2_official_chat_format_512_safety_filtered_v1', [], ['flan2021.bin', 't0.bin.meta', 'dialog.bin.meta', 'niv2.bin', 'niv2.bin.meta', 't0.bin', 'cot.bin.meta', 'cot.bin', 'dialog.bin', 'flan2021.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/char_x10_chat_format', [], ['char_x10_chat_format.bin', 'char_x10_chat_format.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/openai_summary', [], ['tldr_3_filtered_train.bin', 'tldr_3_filtered_train.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/math3000_calculate_solve_thought', [], ['calculate_solve_thought.bin', 'calculate_solve_thought.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/greeting_x10', [], ['greeting_x10.bin', 'greeting_x10.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/ruozhibax10', [], ['ruozhiba_aug.bin.meta', 'ruozhiba_aug.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/moss_emotion_v2', [], ['moss_emotion.bin.meta', 'moss_emotion.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/emoji_chat_format', [], ['emoji_chat_format.bin.meta', 'emoji_chat_format.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/ministry_of_foreign_affairs_safety_filtered_v1', [], ['waijiaobu_qa_v3.bin', 'waijiaobu_qa_v3.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/moss_math_code_debug', [], ['c_s_format_datum_code_error.bin.meta', 'c_s_format_datum_code_error.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/state_council_policy_safety_filtered_v1', [], ['guowuyuan_qa_v3.bin', 'guowuyuan_qa_v3.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/pj_characters_x10', [], ['pj_characters_x10.bin.meta', 'pj_characters_x10.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/moss_math_code', [], ['calculate_format_datum_random.bin.meta', 'solve_format_datum.bin', 'calculate_format_datum_max_num.bin.meta', 'calculate_format_datum_random.bin', 'solve_format_datum.bin.meta', 'calculate_format_datum_max_num.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/self_critique_answer_no_critique', [], ['base_train_20.bin.meta', 'base_train_20.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/self_critique_refine_critique', [], ['helpfulness_train.bin.meta', 'helpfulness_train.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/puyu_chat_format_v2', [], ['puyu_chat_format_v2.bin', 'puyu_chat_format_v2.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/zephyr_ultrachat_200k_filtered', [], ['zephyr-ultrachat-200k_sft_test.bin', 'zephyr-ultrachat-200k_sft_test.bin.meta', 'zephyr-ultrachat-200k_sft_train.bin', 'zephyr-ultrachat-200k_sft_train.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/self_critique_qa', [], ['base_train_50.bin', 'base_train_50.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/math3000_calculate_solve_wo_thought', [], ['merge_calculate_solve_wo_thought.bin.meta', 'merge_calculate_solve_wo_thought.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/lab_info', [], ['lab_info.bin.meta', 'lab_info.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/WizardLM', [], ['alpaca_evol_instruct_70k.bin.meta', 'alpaca_evol_instruct_70k.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/chinese_sensitive_v1', [], ['sensitive_word_qa_5w.bin.meta', 'red_team_chat_format_0808-0822_refine_2x.bin', 'red_team_chat_format_0808-0822_refine_2x.bin.meta', 'sensitive_word_qa_5w.bin', '18k_sensitive_refinev2_pos2en1cn_qmark_aug_2x_insertpos_addmeta.bin.meta', '18k_sensitive_refinev2_pos2en1cn_qmark_aug_2x_insertpos_addmeta.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/data_reflow', [], ['both_bad_aug.bin.meta', 'gpt4_better_aug.bin', 'both_good_aug.bin.meta', 'both_good_aug.bin', 'both_bad_aug.bin', 'gpt4_better_aug.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/gsm8k_pot', [], ['gsm-train-5k-correct-processed.bin', 'gsm-train-5k-correct-processed.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/toolbench_safety_filtered_v1', [], ['database_1k.bin', 'weather_10k.bin.meta', 'database_1k.bin.meta', 'map_7k.bin', 'bing_search_33k.bin.meta', 'google_places_5k.bin.meta', 'translation_10k.bin', 'meta_analysis_2k_single_tool.bin', 'arxiv_6k.bin.meta', 'chemical_10k.bin.meta', 'wolframalpha_17k.bin.meta', 'weather_10k.bin', 'google_places_5k.bin', 'meta_analysis_2k_single_tool.bin.meta', 'translation_10k.bin.meta', 'chemical_10k.bin', 'stock_10k.bin.meta', 'arxiv_6k.bin', 'bing_search_33k.bin', 'map_7k.bin.meta', 'stock_10k.bin', 'wolframalpha_17k.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/self_critique_gen_qa', [], ['base_train_30.bin.meta', 'base_train_30.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/gmath', [], ['gmath.bin.meta', 'gmath.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/gaokao_essay_safety_filtered_v1', [], ['gaokao_essay.bin.meta', 'gaokao_essay.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/chinese_poetry_10x', [], ['chinese-poetry-10x.bin.meta', 'chinese-poetry-10x.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/unnatural_instructions_chat_format', [], ['full_data.bin.meta', 'full_data.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/self_critique_refine_answer', [], ['critiques_train.bin.meta', 'critiques_train.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/moss_no_moss_chat_fudan_format_safety_filtered_v1', [], ['moss_v1_code.bin', 'moss_v1_continue.bin', 'moss_v1_honest.bin.meta', 'moss_v1_awesome_zh.bin', 'moss_v1_harmless_en.bin', 'moss_v1_writing.bin.meta', 'moss_v1_switching.bin', 'moss_v1_advice.bin.meta', 'moss_v1_harmless_zh_china-related_gpt4_fix_qmark_aug.bin.meta', 'moss_v1_harmless_zh_non-chinarelated.bin.meta', 'moss_v1_rp.bin.meta', 'moss_v1_harmless_zh_non-chinarelated.bin', 'moss_v1_code.bin.meta', 'moss_v1_harmless_en.bin.meta', 'moss_v1_rp.bin', 'moss_v1_continue.bin.meta', 'moss_v1_honest.bin', 'moss_v1_awesome_en.bin', 'moss_v1_awesome_zh.bin.meta', 'moss_v1_harmless_zh_china-related_gpt4_fix_qmark_aug.bin', 'moss_v1_awesome_en.bin.meta', 'moss_v1_advice.bin', 'moss_v1_writing.bin', 'moss_v1_switching.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/leetcode_filter_chat_refined', [], ['leetcode_filter_chat_refined.bin.meta', 'leetcode_filter_chat_refined.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/math3000_solve_wo_thought_wo_answer', [], ['solve_wo_thought_wo_answer.bin', 'solve_wo_thought_wo_answer.bin.meta']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/EvolCode_v4x_r2_0', [], ['polishedcode-v4.x-r2.bin.meta', 'polishedcode-v4.x-r2.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/stackoverflow_selected_python_chat_format', [], ['stackoverflow_selected_python_chat_format.bin.meta', 'stackoverflow_selected_python_chat_format.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/rolebench_w_sys_filtered', [], ['rolebench-zh_general_train.bin.meta', 'rolebench-zh_role_specific_train.bin', 'rolebench-eng_instruction-generalization_general_train.bin.meta', 'rolebench-eng_role-generalization_general_train.bin', 'rolebench-zh_role_specific_train.bin.meta', 'rolebench-eng_instruction-generalization_role_specific_train.bin.meta', 'rolebench-eng_role-generalization_general_train.bin.meta', 'rolebench-eng_instruction-generalization_general_train.bin', 'rolebench-eng_role-generalization_role_specific_train.bin.meta', 'rolebench-eng_instruction-generalization_role_specific_train.bin', 'rolebench-eng_role-generalization_role_specific_train.bin', 'rolebench-zh_general_train.bin']), ('/cpfs01/shared/public/gaojianfei/datasets/llamav7_8k/train/cn/share_gpt_v6_chat_format_safety_filtered_v1', [], ['share_gpt_v6.bin', 'share_gpt_v6.bin.meta'])]

    for root, dirs, files in triples:
        dirs.sort()  # Let the folder need to be returned in a fixed order
        if dist.get_rank() == 0:
            print_log(f'Reading {root}...', logger='current')
        num_token_in_folder = 0

        for fn in tqdm(sorted(files), total=len(files), leave=False, disable=dist.get_rank() != 0):
            if fn.endswith(".bin"):
                fp = os.path.join(root, fn)
                # ds = load_dataset('json', data_files=fp)['train']
                ds = JsonlDataset(fp, min_length=min_length)

                if len(ds) == 0:
                    continue

                ds = PackedDataset(ds, packed_length)

                num_token_in_folder += len(ds) * packed_length
                datasets.append(ds)

    dataset = ConcatDataset(datasets=datasets)

    return dataset


from torch.utils.data import Sampler
from typing import Iterator, Optional, Sized
from mmengine.dist import get_dist_info, sync_random_seed
class DefaultSampler(Sampler):
    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        self.num_samples = len(self.dataset) // world_size
        self.total_size = self.num_samples * world_size

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = np.arange(len(self.dataset))
            rng.shuffle(indices)
            indices = indices.tolist()
        else:
            indices = np.arange(len(self.dataset)).tolist()

        self.indices = indices[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        self.subsample_indices = indices

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



class StaticBatchSampler:
    def __init__(
        self,
        dataset,
        batch_size=192,
        micro_bsz=2,
        seed=0,
        drop_last=True,
    ):
        assert drop_last is True, "Currently only support drop last"

        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.batch_count = 0
        self.micro_bsz = micro_bsz
        self.data_rank = dist.get_rank()
        self.data_world_size = dist.get_world_size()
        self.num_consumed_samples_in_epoch = 0
        self.num_samples = sum([len(ds) for ds in dataset.datasets])

        self.get_indices()  # get data
        self.sampler = None

    def get_indices(self):
        indices = np.arange(0, self.num_samples)
        self.rng_state = self.rng.get_state()
        self.rng.shuffle(indices)
        # Need to consider drop_last
        
        num_samples = self.num_samples // (self.batch_size * self.data_world_size)
        num_samples = num_samples * self.batch_size * self.data_world_size
        indices = indices.astype(int)  # It needs to be spliced with the previous
        indices = indices[:num_samples]
        self.indices = indices
        assert len(self.indices) >= self.batch_size, "The number of samples should be larger than batch_size"
        self.num_consumed_samples_in_epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + self.epoch)

    def __len__(self):
        num_batches = self.num_samples // self.batch_size // self.data_world_size
        return num_batches

    def __iter__(self):
        indices = self.indices[self.data_rank :: self.data_world_size]
        while self.num_consumed_samples_in_epoch < len(indices):
            batch = indices[self.num_consumed_samples_in_epoch : self.num_consumed_samples_in_epoch + self.batch_size]
            yield batch
            self.num_consumed_samples_in_epoch += len(batch)  # Consider multiple processes.
            self.batch_count += 1
        self.get_indices()  # get a new round

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "rng_state": self.rng_state,
            "epoch": self.epoch,
            "seed": self.seed,
            "data_world_size": self.data_world_size,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "batch_count": self.batch_count,  # The batch_count here is due to the existence of multiple processes,
            # the batch may be oversent, and it needs to be overwritten by the external batch_count
            "indices": self.indices,  # The sequence used to breakpoint retraining is the same as before
        }

        return states

    def load_state_dict(self, states):
        for name in ("data_world_size", "raw_rampup_batch_size", "seed"):  # 'batch_size'
            assert states[name] == getattr(self, name), (name, states[name], getattr(self, name))  # should not change
        self.rng.set_state(states["rng_state"])
        self.get_indices(old_indices=None)  # Regenerate indices based on random state
        self.epoch = states["epoch"]
        self.batch_count = states["batch_count"]
        self.num_consumed_samples_in_epoch = states["num_consumed_samples_in_epoch"]


def packed_collate_fn(batch, packed_length, accumulative_counts):

    xs, ys, cu_seqlens, indexes = [], [], [], []
    for b in batch:
        assert (
            len(b["input_ids"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['input_ids'])} and {packed_length})"
        assert (
            len(b["labels"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['labels'])} and {packed_length})"

        input_ids = [abs(w) for w in b["input_ids"]]
        labels = [w if w > 0 else -100 for w in b["labels"]]

        xs.append(torch.LongTensor(input_ids))
        # The labels have been shifted here, so they are aligned with the output corresponding to the token
        ys.append(torch.LongTensor(labels))
        cu_seqlens.append(torch.IntTensor(b["cu_seqlens"]))
        indexes.append(torch.LongTensor(b["indexes"]))

    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    indexes = torch.stack(indexes, dim=0)

    assert xs.shape[1] == packed_length, (xs.shape[1], packed_length)
    assert len(batch) == accumulative_counts
    max_seqlen = [(cu_seqlens[i][1:] - cu_seqlens[i][:-1]).max().item() for i in range(accumulative_counts)]
    data_dict = {"input_ids": xs, "cumulative_len": cu_seqlens, "indexes": indexes, "labels": ys, "max_seqlen": max_seqlen}

    return {'data': data_dict, 'data_samples': None}
