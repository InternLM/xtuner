import json
import os
import re
from collections import OrderedDict

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine import print_log
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import load_state_dict
from transformers.utils import (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME,
                                is_safetensors_available)

SUPPORT_MODELS = (
    'DeepseekV2ForCausalLM',
    'MixtralForCausalLM',
)

ORDER_MAPPING = dict(
    DeepseekV2ForCausalLM=dict(down_proj=0, gate_proj=1, up_proj=2),
    MixtralForCausalLM=dict(down_proj=1, gate_proj=0, up_proj=2),
)

PARAM_NAME_MAPPING = dict(
    DeepseekV2ForCausalLM=dict(
        gate_proj='gate_proj', up_proj='up_proj', down_proj='down_proj'),
    MixtralForCausalLM=dict(gate_proj='w1', up_proj='w3', down_proj='w2'),
)


def print_on_rank0(info):
    if dist.get_rank() == 0:
        print_log(info, 'current')


def get_expert_num_per_shard(model):
    for module in model.modules():
        if hasattr(module, 'expert_in_one_shard'):
            return module.expert_in_one_shard


def mix_sort(expert_name):
    components = re.findall(r'(\D+|\d+)', expert_name)
    out = [int(comp) if comp.isdigit() else comp for comp in components]
    return tuple(out)


def _get_merged_param_name(origin_param_name, expert_num_per_shard):
    split_name = origin_param_name.split('.experts.')
    expert_idx = re.findall(r'\d+', split_name[1])[0]
    expert_idx = int(expert_idx)
    assert expert_idx % expert_num_per_shard == 0
    shard_idx = expert_idx // expert_num_per_shard
    w1w3 = split_name[0] + f'.experts.{shard_idx}.w1w3'
    w2 = split_name[0] + f'.experts.{shard_idx}.w2'
    return w1w3, w2


def _merge_experts_weight(state_dict, expert_num_per_shard, order_mapping):
    experts_name = [key for key in state_dict.keys() if '.experts.' in key]
    experts_name = sorted(experts_name, key=mix_sort)
    linear_num_per_expert = 3
    linear_num_per_shard = expert_num_per_shard * linear_num_per_expert
    expert_shard_num = len(experts_name) // linear_num_per_shard
    for shard_idx in range(expert_shard_num):
        begin, end = shard_idx * linear_num_per_shard, (
            shard_idx + 1) * linear_num_per_shard
        experts_name_cur = experts_name[begin:end]

        down_proj_weight = [
            state_dict.pop(key)
            for key in experts_name_cur[order_mapping['down_proj']::3]
        ]
        gate_proj_weight = [
            state_dict.pop(key)
            for key in experts_name_cur[order_mapping['gate_proj']::3]
        ]
        up_proj_weight = [
            state_dict.pop(key)
            for key in experts_name_cur[order_mapping['up_proj']::3]
        ]
        w1 = torch.stack(gate_proj_weight)
        w3 = torch.stack(up_proj_weight)
        w1w3 = torch.cat([w1, w3], dim=1)
        assert w1w3.ndim == 3, w1w3.shape
        w2 = torch.stack(down_proj_weight)
        assert w2.ndim == 3, w2.shape
        merged_key_w1w3, merged_key_w2 = _get_merged_param_name(
            experts_name_cur[0], expert_num_per_shard)
        print_on_rank0(f'merged key {merged_key_w1w3}')
        state_dict[merged_key_w1w3] = w1w3
        print_on_rank0(f'merged key {merged_key_w2}')
        state_dict[merged_key_w2] = w2

    return


def load_state_dict_into_model(model_to_load, pretrained_model_path):

    model_name = type(model_to_load).__name__
    if model_name not in SUPPORT_MODELS:
        raise RuntimeError(
            f'Only models in {SUPPORT_MODELS} may need to load pretrained '
            f'weights via `load_state_dict_into_model`, but got {model_name}.')
    order_mapping = ORDER_MAPPING[model_name]

    index_file = os.path.join(pretrained_model_path, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(pretrained_model_path,
                                   SAFE_WEIGHTS_INDEX_NAME)
    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    assert index_present or (safe_index_present and is_safetensors_available())
    if safe_index_present and is_safetensors_available():
        load_index = safe_index_file
    else:
        load_index = index_file
    with open(load_index, encoding='utf-8') as f:
        index = json.load(f)
    weight_map = index['weight_map']
    unloaded_shard_files = list(set(weight_map.values()))
    unloaded_shard_files.sort(reverse=True)

    expert_num_per_shard = get_expert_num_per_shard(model_to_load)
    error_msgs = []

    def load(module: nn.Module, state_dict, unloaded_shard_files, prefix=''):
        params_to_gather = []
        param_names = []
        for name, param in module.named_parameters(
                prefix=prefix[:-1], recurse=False):
            while name not in state_dict:
                assert len(unloaded_shard_files) > 0
                shard_file = unloaded_shard_files.pop()
                shard_file = os.path.join(pretrained_model_path, shard_file)
                print_on_rank0(
                    f'{name} not in state_dict, loading {shard_file}')
                new_shard = load_state_dict(shard_file, is_quantized=False)
                state_dict.update(new_shard)
                _merge_experts_weight(state_dict, expert_num_per_shard,
                                      order_mapping)
            params_to_gather.append(param)
            param_names.append(name)
        if len(params_to_gather) > 0:
            args = (state_dict, prefix, {}, True, [], [], error_msgs)
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(
                        params_to_gather, modifier_rank=0):
                    if dist.get_rank() == 0:
                        module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name in param_names:
            print_on_rank0(f'state_dict pop {name}')
            state_dict.pop(name)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, unloaded_shard_files,
                     prefix + name + '.')

    state_dict = OrderedDict()
    load(model_to_load, state_dict, unloaded_shard_files, prefix='')
    print_on_rank0(f'{state_dict.keys()}')
    del state_dict

    return error_msgs


def _get_origin_param_name(merged_param_name, expert_num_per_shard, is_w1w3,
                           param_name_mapping):
    split_name = merged_param_name.split('.experts.')
    shard_idx = re.findall(r'\d+', split_name[1])[0]
    shard_idx = int(shard_idx)
    origin_param_names = [None] * (expert_num_per_shard * (1 + int(is_w1w3)))
    expert_idx_begin = expert_num_per_shard * shard_idx
    for i in range(expert_num_per_shard):
        if is_w1w3:
            gate_proj, up_proj = param_name_mapping[
                'gate_proj'], param_name_mapping['up_proj']
            gate = split_name[
                0] + f'.experts.{expert_idx_begin + i}.{gate_proj}.weight'
            up = split_name[
                0] + f'.experts.{expert_idx_begin + i}.{up_proj}.weight'
            origin_param_names[i * 2] = gate
            origin_param_names[i * 2 + 1] = up
        else:
            down_proj = param_name_mapping['down_proj']
            down = split_name[
                0] + f'.experts.{expert_idx_begin + i}.{down_proj}.weight'
            origin_param_names[i] = down
    return origin_param_names


def _split_param(merged_param, is_w1w3):
    if is_w1w3:
        expert_num, _, hidden_dim = merged_param.shape
        merged_param = merged_param.view(expert_num * 2, -1, hidden_dim)
        return torch.unbind(merged_param, dim=0)
    else:
        # (e, hidden_dim, ffn_dim)
        return torch.unbind(merged_param, dim=0)


def get_origin_state_dict(state_dict, model):

    model_name = type(model).__name__
    if model_name not in SUPPORT_MODELS:
        raise RuntimeError(
            f'Only models in {SUPPORT_MODELS} may need to convert state_dict '
            f'via `get_origin_state_dict` interface, but got {model_name}.')
    param_name_mapping = PARAM_NAME_MAPPING[model_name]

    expert_num_per_shard = get_expert_num_per_shard(model)
    experts_param_name = [
        name for name in state_dict.keys() if '.experts.' in name
    ]
    for expert_param_name in experts_param_name:
        print_on_rank0(f'processing {expert_param_name} ...')
        is_w1w3 = expert_param_name.split('.')[-1] == 'w1w3'
        origin_param_names = _get_origin_param_name(expert_param_name,
                                                    expert_num_per_shard,
                                                    is_w1w3,
                                                    param_name_mapping)
        merged_param = state_dict.pop(expert_param_name)
        origin_params = _split_param(merged_param, is_w1w3)
        assert len(origin_param_names) == len(origin_params)
        for name, param in zip(origin_param_names, origin_params):
            state_dict[name] = param
    return state_dict
