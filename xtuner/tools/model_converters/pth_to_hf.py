# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil

import torch
from mmengine.config import Config, DictAction

from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert the pth model to HuggingFace model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')
    parser.add_argument(
        'save_dir', help='the directory to save HuggingFace model')
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Save as fp32. If not set, fp16 will be used by default.')
    parser.add_argument(
        '--max-shard-size',
        type=str,
        default='2GB',
        help='Only applicable for LLM. The maximum size for '
        'each sharded checkpoint.')
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


def guess_load_checkpoint(pth_model):
    if os.path.isfile(pth_model):
        state_dict = torch.load(pth_model, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    elif os.path.isdir(pth_model):
        try:
            from deepspeed.utils.zero_to_fp32 import \
                get_fp32_state_dict_from_zero_checkpoint
        except ImportError:
            raise ImportError(
                'The provided PTH model appears to be a DeepSpeed checkpoint. '
                'However, DeepSpeed library is not detected in current '
                'environment. This suggests that DeepSpeed may not be '
                'installed or is incorrectly configured. Please verify your '
                'setup.')
        state_dict = get_fp32_state_dict_from_zero_checkpoint(
            os.path.dirname(pth_model), os.path.basename(pth_model))
    else:
        raise FileNotFoundError(f'Cannot find {pth_model}')
    return state_dict


def main():
    args = parse_args()

    # parse config
    if not os.path.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = BUILDER.build(cfg.model)

    state_dict = guess_load_checkpoint(args.pth_model)
    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    if not args.fp32:
        print('Convert weights to float16')
        model.llm.half()

    print(f'Saving HuggingFace model to {args.save_dir}')
    model.llm.save_pretrained(
        args.save_dir, max_shard_size=args.max_shard_size)
    if 'PeftModel' not in model.llm.__class__.__name__:
        print(f'Saving HuggingFace tokenizer to {args.save_dir}')
        tokenizer = BUILDER.build(cfg.tokenizer)
        tokenizer.save_pretrained(args.save_dir)
    shutil.copyfile(args.config, os.path.join(args.save_dir,
                                              'xtuner_config.py'))
    print('All done!')


if __name__ == '__main__':
    main()
