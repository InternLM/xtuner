# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil

import torch
from mmengine.config import Config, DictAction
from mmengine.utils import mkdir_or_exist

from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert the pth adapter to HuggingFace adapter')
    parser.add_argument(
        'config',
        help='config file name or path. Note: Please use the original '
        'configs, instead of the automatically saved log configs.')
    parser.add_argument('adapter_checkpoint', help='adapter checkpoint file')
    parser.add_argument(
        'save_dir', help='the directory to save the checkpoint')
    parser.add_argument(
        '--is-deepspeed',
        action='store_true',
        help='whether the adapter is saved from deepspeed')
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

    # load on cpu
    cfg.model.llm.device_map = 'cpu'
    if cfg.model.llm.get('quantization_config'):
        cfg.model.llm.quantization_config.\
            llm_int8_enable_fp32_cpu_offload = True

    model = BUILDER.build(cfg.model)

    adapter_checkpoint = torch.load(
        args.adapter_checkpoint, map_location='cpu')
    state_dict_key = 'module' if args.is_deepspeed else 'state_dict'
    model.load_state_dict(adapter_checkpoint[state_dict_key], strict=False)
    print(f'Load adapter from {args.adapter_checkpoint}')

    mkdir_or_exist(args.save_dir)
    model.llm.save_pretrained(args.save_dir)
    shutil.copyfile(args.config, os.path.join(args.save_dir,
                                              'xtuner_config.py'))
    print(f'Save to {args.save_dir}')


if __name__ == '__main__':
    main()
