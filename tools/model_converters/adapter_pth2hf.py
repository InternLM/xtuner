# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import torch
from mmengine.config import Config, DictAction
from mmengine.utils import mkdir_or_exist

from mmchat.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='MMChat test a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('adapter_checkpoint', help='adapter checkpoint file')
    parser.add_argument(
        'save_dir', help='the directory to save the checkpoint')
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

    # load on cpu
    if cfg.model.llm.get('device_map'):
        cfg.model.llm.device_map = 'cpu'
    if cfg.model.llm.get('quantization_config'):
        cfg.model.llm.quantization_config.\
            llm_int8_enable_fp32_cpu_offload = True

    model = MODELS.build(cfg.model)

    adapter_checkpoint = torch.load(
        args.adapter_checkpoint, map_location='cpu')
    model.load_state_dict(adapter_checkpoint['state_dict'], strict=False)
    print(f'Load adapter from {args.adapter_checkpoint}')

    adapter_path = os.path.join(args.save_dir, 'adapter')
    mkdir_or_exist(adapter_path)
    model.llm.save_pretrained(adapter_path)
    print(f'Save to {adapter_path}')


if __name__ == '__main__':
    main()
