import json
import copy

import numpy as np
from multiprocessing.pool import ThreadPool as Pool

import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoTokenizer

from .internvl_dataset import InternVL_V1_5_Dataset


def get_token_sum(g):
    sum = 0
    for i in g:
        sum += i[2]
    return sum


def get_vit_num(g):
    vit_num = 0
    for _ in g:
        vit_num += _[1]
    return vit_num


DEFAULT_SEED = 1024
class BalancedDataset(Dataset):
    def __init__(self,
                 data_path,
                 model_path,
                 template,
                 max_length,
                 vit_packed_length=15,
                 llm_packed_length=4096,
                 llm_thresh={},
                 worker=64,
                 iter_time=100):
        cfg_dataset_base = {}
        cfg_dataset_base['template'] = template
        cfg_dataset_base['model_path'] = model_path
        cfg_dataset_base['max_length'] = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.vit_packed_length = vit_packed_length
        self.llm_packed_length = llm_packed_length
        self.llm_thresh = {'thresh': llm_thresh}

        self.vit_lengths, self.llm_lengths = [], []
        self.worker = worker
        self.pad_token_id = len(self.tokenizer) - 1
        self.iter_time = iter_time

        # prepare concat dataset
        ds_collections = json.loads(open(data_path).read())
        datasets = []
        for ds_name in ds_collections.keys():
            cfg_dataset = copy.deepcopy(cfg_dataset_base)
            cfg_dataset['repeat_times'] = ds_collections[ds_name]['repeat_time']
            cfg_dataset['data_paths'] = ds_collections[ds_name]['annotation']
            cfg_dataset['image_folders'] = ds_collections[ds_name]['root']

            dataset = InternVL_V1_5_Dataset(**cfg_dataset)
            dataset.meta = ds_collections[ds_name]
            datasets.append(dataset)

        self.dataset = ConcatDataset(datasets)
        print("Begin preprocess dataset", flush=True)
        self.preprocess()
        print("Preprocess dataset successed", flush=True)
        self.seed = DEFAULT_SEED
        self.pack_groups = self.get_packed_groups()

        group_length = []
        for g in self.pack_groups:
            num_img = 0
            length_input = 0
            for item in g:
                num_img += item[3]
                length_input += item[2]

            if num_img != 0:
                group_length.append(length_input)
            else:
                group_length.append(-length_input)

        self.group_length = group_length

    @property
    def modality_length(self):
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length

    def preprocess(self):
        dict_num_tokens = {}
        num_datasets = len(self.dataset.datasets)
        for dataset_idx in range(num_datasets):
            sub_dataset = self.dataset.datasets[dataset_idx]
            if "token_lengths" in sub_dataset.meta:
                print(f"Load from cache for dataset {dataset_idx}", flush=True)
                assert os.path.exists(sub_dataset.meta["token_lengths"]), f"Dataset {dataset_idx} token_lengths file does not exist."
                with open(sub_dataset.meta["token_lengths"], "r") as f:
                    token_lengths = json.load(f)
                dict_num_tokens[dataset_idx] = {
                    "lengths": len(sub_dataset),
                    "token_lengths": token_lengths  # sub_dataset.meta["token_lengths"]
                }
            else:
                print(f"Generate length json for dataset {dataset_idx}", flush=True)
                token_lengths = []
                origin_indexs = list(range(len(sub_dataset)))
                token_lengths_dict = dict()

                def decode_text(idx):
                    meta = sub_dataset.__getitem__(idx)
                    if meta['pixel_values'].sum().item() == 0:
                        image_flags = 0
                    else:
                        image_flags = meta['pixel_values'].shape[0]
                    token_lengths_dict[idx] = {
                        "vit_num": meta['pixel_values'].shape[0],
                        "token_num": len(meta['input_ids']),
                        "image_flags": image_flags
                    }

                with Pool(self.worker) as p:
                    _ = p.map(decode_text, origin_indexs[:])
                for idx in range(len(sub_dataset)):
                    token_lengths.append(
                        token_lengths_dict[idx]
                    )
                dict_num_tokens[dataset_idx] = {
                    "lengths": len(sub_dataset),
                    "token_lengths": token_lengths
                }
                print(f"Finish length json for dataset {dataset_idx}", flush=True)
        self.dict_num_tokens = dict_num_tokens

    def _random_groups(self, token_lengths, seed=None):
        """
        tokens_length: [(idx, vit_img_num, llm_token_len)]
        """
        rng = np.random.RandomState(seed)
        index = list(range(len(token_lengths)))
        rng.shuffle(index)

        pack_groups = []
        vit_token_length_sum, llm_token_length_sum = 0, 0
        each_group = []
        for idx, sample_id in enumerate(index):
            vit_sample_length, llm_sample_length = token_lengths[sample_id][1], token_lengths[sample_id][2]
            if vit_sample_length > self.vit_packed_length or llm_sample_length > self.llm_packed_length:
                continue
            vit_token_length_sum += vit_sample_length
            llm_token_length_sum += llm_sample_length
            if vit_token_length_sum > self.vit_packed_length or llm_token_length_sum > self.llm_packed_length:
                pack_groups.append(each_group)
                vit_token_length_sum = vit_sample_length
                llm_token_length_sum = llm_sample_length
                each_group = [token_lengths[sample_id]]
            else:
                each_group.append(token_lengths[sample_id])
            if idx == len(token_lengths) - 1:
                if len(each_group) > 0:
                    pack_groups.append(each_group)
        return pack_groups

    def process_random_groups_input(self, groups, accu_length=0):
        new_groups = []
        for idx, item in enumerate(groups):
            if item["vit_num"] == -1:
                print(f"item {idx} was filted.", flush=True)
                continue
            new_groups.append((idx + accu_length, item['vit_num'], item['token_num'], item["image_flags"]))
        return new_groups

    def iter_random_groups(self, groups, llm_thresh=None, seed=None, iter_time=300):
        if llm_thresh is None:
            llm_thresh = self.llm_packed_length
        if seed is None:
            seed = self.seed
        groups = self._random_groups(groups, seed=seed)
        if iter_time == 1:
            return groups
        output = []
        for i in range(iter_time - 1):
            print(f"iter_random_groups {i} / {iter_time - 1}", flush=True)
            need_process_groups = []
            for g in groups:
                vit_num = get_vit_num(g)
                llm_num = get_token_sum(g)
                if vit_num == self.vit_packed_length or llm_num >= llm_thresh:
                    output.append(g)
                else:
                    need_process_groups.extend(g)
            if len(need_process_groups) >= 0:
                groups = self._random_groups(need_process_groups, seed + i)
            else:
                break
        if len(need_process_groups) > 0:
            output.extend(self._random_groups(need_process_groups, seed + i))
        return output

    def collect_packed_info(self, packed_groups):
        info_dict = {}
        info_dict['vit_num_info'] = {}
        vit_num_min = 10000000
        vit_num_max = 0
        llm_num_min = 10000000
        llm_num_max = 0
        vit_ave_num = 0
        llm_ave_num = 0
        sample_num = 0
        for group in packed_groups:
            vit_num = get_vit_num(group)
            llm_num = get_token_sum(group)
            if vit_num not in info_dict['vit_num_info']:
                info_dict['vit_num_info'][vit_num] = 0
            info_dict['vit_num_info'][vit_num] += 1
            vit_num_min = min(vit_num_min, vit_num)
            vit_num_max = max(vit_num_max, vit_num)
            llm_num_min = min(llm_num_min, llm_num)
            llm_num_max = max(llm_num_max, llm_num)
            vit_ave_num += vit_num
            llm_ave_num += llm_num
            sample_num += len(group)
        info_dict['vit_num_min'] = vit_num_min
        info_dict['vit_num_max'] = vit_num_max
        info_dict['vit_ave_num'] = vit_ave_num / float(len(packed_groups))
        info_dict['llm_ave_num'] = llm_ave_num / float(len(packed_groups))
        info_dict['sample_num'] = sample_num
        info_dict['packed_group_num'] = len(packed_groups)
        return info_dict

    def find_best_groups(self, input_groups, step=4, step_num=20):
        best_group_num = 10000000000000
        best_groups = []
        best_info_dict = {}
        best_llm_thresh = 0
        llm_thresh = self.llm_packed_length
        for step_id in range(step_num):
            print(f"find_best_groups {step_id} / {step_num}", flush=True)
            groups = self.iter_random_groups(input_groups, llm_thresh, seed=self.seed, iter_time=self.iter_time)
            cur_info_dict = self.collect_packed_info(groups)
            if cur_info_dict['packed_group_num'] < best_group_num:
                best_group_num = cur_info_dict['packed_group_num']
                best_groups = groups
                best_info_dict = cur_info_dict
                best_llm_thresh = llm_thresh
            llm_thresh -= step
        print(f"llm thresh {best_llm_thresh} best info dict", best_info_dict, flush=True)
        return best_groups

    def get_packed_groups(self):
        num_datasets = len(list(self.dict_num_tokens.keys()))
        accu_length = 0
        input_groups = []
        for d_idx in range(num_datasets):
            dict_item = self.dict_num_tokens[d_idx]
            token_lengths = dict_item["token_lengths"]
            groups = self.process_random_groups_input(token_lengths, accu_length)
            print(f"get_packed_groups {d_idx}.", flush=True)
            input_groups.extend(groups)
            accu_length += len(token_lengths)
        if self.llm_thresh.get('thresh', None) is not None:
            groups = self.iter_random_groups(input_groups, llm_thresh=self.llm_thresh['thresh'], seed=self.seed, iter_time=self.iter_time)
        else:
            groups = self.find_best_groups(input_groups, self.llm_thresh.get('step', 4), self.llm_thresh.get('step_num', 10))
        print(self.collect_packed_info(groups), flush=True)
        print("get_packed_groups done!", flush=True)
        return groups

    def __getitem__(self, item: int):
        item = item % len(self.pack_groups)
        # item = random.randint(0, len(self.pack_groups) - 1)
        while True:
            try:
                groups = self.pack_groups[item]

                input_ids, pixel_values = [], []
                labels, position_ids, image_flags = [], [], []
                cu_seqlens = [0]
                for g in groups:
                    idx, _, llm_length, image_flag = g
                    meta = self.dataset.__getitem__(idx)
               #    print("llm_length: ", llm_length, "input_ids: ", len(meta["input_ids"]))
                    assert len(meta["input_ids"]) == llm_length
                    image_flag_ = (torch.sum(meta['pixel_values'], dim=(1, 2, 3)) != 0).sum()
                    assert image_flag == image_flag_.item()
                    input_ids.extend(meta['input_ids'])
                    pixel_values.append(meta['pixel_values'])
                    labels.extend(meta['labels'])
                    cu_seqlens.append(len(meta['input_ids']))
                    position_ids.extend(list(range(len(meta['input_ids']))))
                    image_flags.append(image_flag)

                cu_seqlens = np.cumsum(np.array(cu_seqlens)).tolist()
                input_ids = input_ids[:self.llm_packed_length]
                pixel_values = torch.cat(pixel_values)[:self.vit_packed_length]
                labels = labels[:self.llm_packed_length]
                position_ids = position_ids[:self.llm_packed_length]
                cu_seqlens[-1] = len(position_ids)

                ret = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "cumulative_len": cu_seqlens,
                    "position_ids": position_ids,
                    "pixel_values": pixel_values,
                }
                break
            except Exception as e:
                print(f"{e}", flush=True)
                # i = random.randint(0, len(self.raw_data) - 1)
                item = (item + 100) % len(self.pack_groups)
        return ret

    def __len__(self):
        n_packs = len(self.pack_groups)
        return n_packs