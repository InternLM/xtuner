# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge a HuggingFace adapter to LLM')
    parser.add_argument('model_name_or_path', help='model name or path')
    parser.add_argument('adapter_name_or_path', help='adapter name or path')
    parser.add_argument(
        'save_dir', help='the directory to save the merged model')
    parser.add_argument(
        '--max-shard-size',
        type=str,
        default='2GB',
        help='Only applicable for LLM. The maximum size for '
        'each sharded checkpoint.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='cpu',
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)
    model_unmerged = PeftModel.from_pretrained(
        model,
        args.adapter_name_or_path,
        device_map='cpu',
        torch_dtype=torch.float16,
        is_trainable=False)
    model_merged = model_unmerged.merge_and_unload()
    model_merged.save_pretrained(
        args.save_dir, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.save_dir)
    print(f'Save to {args.save_dir}')


if __name__ == '__main__':
    main()
