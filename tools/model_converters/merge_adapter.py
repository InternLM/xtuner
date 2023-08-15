# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmengine.config import Config, DictAction

from xtuner.registry import MODELS, TOKENIZER


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge a pth adapter to model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('adapter_checkpoint', help='adapter checkpoint file')
    parser.add_argument(
        'save_dir', help='the directory to save the merged model')
    parser.add_argument('--max-shard-size', type=str, default='2GB')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # load on cpu, with non-quantized
    cfg.model.llm.device_map = 'cpu'
    cfg.model.llm.quantization_config = None
    cfg.model.llm.low_cpu_mem_usage = True
    torch_dtype = cfg.model.llm.get('torch_dtype', torch.float16)
    model = MODELS.build(cfg.model)
    tokenizer = TOKENIZER.build(cfg.tokenizer)
    adapter_checkpoint = torch.load(
        args.adapter_checkpoint, map_location='cpu')
    model.load_state_dict(adapter_checkpoint['state_dict'], strict=False)
    print(f'Load adapter from {args.adapter_checkpoint}')

    model = model.llm
    model_merged = model.merge_and_unload()
    for param in model.parameters():
        param.data = param.data.to(torch_dtype)
    model_merged.save_pretrained(
        args.save_dir, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.save_dir)
    print(f'Save to {args.save_dir}')


if __name__ == '__main__':
    main()
