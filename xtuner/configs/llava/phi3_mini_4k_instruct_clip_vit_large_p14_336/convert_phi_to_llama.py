# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os

from mmengine.utils import mkdir_or_exist
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer


def convert_phi_to_llama(phi_path, save_path):
    files = [f for f in os.listdir(phi_path) if f.endswith('safetensors')]
    mkdir_or_exist(save_path)

    index_json = os.path.join(phi_path, 'model.safetensors.index.json')
    config_json = os.path.join(phi_path, 'config.json')

    with open(index_json) as f:
        index = json.load(f)

    with open(config_json) as f:
        config = json.load(f)

    config.pop('_name_or_path')
    if 'auto_map' in config:
        config.pop('auto_map')
    config.pop('embd_pdrop')
    config.pop('resid_pdrop')
    config['architectures'] = ['LlamaForCausalLM']
    config['model_type'] = 'llama'

    for file in tqdm(files, desc='Convert'):
        tensors = {}
        new_path = os.path.join(save_path, file)
        old_path = os.path.join(phi_path, file)
        with safe_open(old_path, framework='pt', device='cpu') as f:
            for key in f.keys():

                if 'qkv_proj' in key:
                    qkv = f.get_tensor(key)

                    q, k, v = qkv.chunk(3, dim=0)
                    q_name = key.replace('qkv_proj', 'q_proj')
                    k_name = key.replace('qkv_proj', 'k_proj')
                    v_name = key.replace('qkv_proj', 'v_proj')

                    tensors[q_name] = q
                    tensors[k_name] = k
                    tensors[v_name] = v

                    index['weight_map'].pop(key)

                    filename = os.path.basename(new_path)
                    index['weight_map'][q_name] = filename
                    index['weight_map'][k_name] = filename
                    index['weight_map'][v_name] = filename

                elif 'gate_up_proj' in key:
                    gate_up_proj = f.get_tensor(key)
                    gate_proj, up_proj = gate_up_proj.chunk(2, dim=0)

                    gate_name = key.replace('gate_up_proj', 'gate_proj')
                    up_name = key.replace('gate_up_proj', 'up_proj')
                    tensors[gate_name] = gate_proj
                    tensors[up_name] = up_proj

                    index['weight_map'].pop(key)
                    filename = os.path.basename(new_path)
                    index['weight_map'][gate_name] = filename
                    index['weight_map'][up_name] = filename
                else:
                    tensors[key] = f.get_tensor(key)
            metadata = f.metadata()
        save_file(tensors, new_path, metadata=metadata)

    new_config_json = os.path.join(save_path, 'config.json')
    with open(new_config_json, 'w') as f:
        json.dump(config, f, indent=2)

    new_index_json = os.path.join(save_path, 'model.safetensors.index.json')
    with open(new_index_json, 'w') as f:
        json.dump(index, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(phi_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    print(f'Saved to {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phi_path')
    parser.add_argument('--save_path')
    args = parser.parse_args()
    convert_phi_to_llama(args.phi_path, args.save_path)


if __name__ == '__main__':
    main()
