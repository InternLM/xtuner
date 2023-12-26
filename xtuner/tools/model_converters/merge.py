# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel)

from xtuner.model.utils import LoadWoInit


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge a HuggingFace adapter to base model')
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
    parser.add_argument(
        '--is-clip',
        action='store_true',
        help='Indicate if the model is a clip model')
    parser.add_argument(
        '--device',
        default='cuda',
        choices=('cuda', 'cpu', 'auto'),
        help='Indicate the device')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.is_clip:
        with LoadWoInit():
            model = CLIPVisionModel.from_pretrained(
                args.model_name_or_path, device_map=args.device)
        processor = CLIPImageProcessor.from_pretrained(args.model_name_or_path)
    else:
        with LoadWoInit():
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=args.device,
                trust_remote_code=True)
        processor = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True)
    model_unmerged = PeftModel.from_pretrained(
        model,
        args.adapter_name_or_path,
        device_map=args.device,
        is_trainable=False)
    model_merged = model_unmerged.merge_and_unload()
    print(f'Saving to {args.save_dir}...')
    model_merged.save_pretrained(
        args.save_dir,
        safe_serialization=False,
        max_shard_size=args.max_shard_size)
    processor.save_pretrained(args.save_dir)
    print('All done!')


if __name__ == '__main__':
    main()
