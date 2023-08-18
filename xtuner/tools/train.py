# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from xtuner.configs import cfgs_name_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')
    parser.add_argument('config', help='config file name or path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--deepspeed',
        type=str,
        default=None,
        help='the path to the .json file for deepspeed')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # parse config
    if not os.path.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            print(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        from mmengine.optim import AmpOptimWrapper, OptimWrapper
        if optim_wrapper == AmpOptimWrapper:
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == OptimWrapper, (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.deepspeed:
        try:
            import deepspeed  # pre-check  # noqa: F401
        except ImportError:
            raise ImportError(
                'deepspeed is not installed properly, please check.')
        optim_wrapper = cfg.optim_wrapper.type
        from mmengine.optim import DeepSpeedOptimWrapper, OptimWrapper
        if optim_wrapper == DeepSpeedOptimWrapper:
            print_log(
                'Deepspeed training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            optimizer = cfg.optim_wrapper.optimizer
            gradient_clipping = 1.0
            clip_grad = cfg.optim_wrapper.get('clip_grad', None)
            if clip_grad and clip_grad.get('max_norm'):
                gradient_clipping = cfg.optim_wrapper.clip_grad.max_norm
            optim_wrapper = dict(
                type='DeepSpeedOptimWrapper', optimizer=optimizer)
            cfg.__setitem__('optim_wrapper', optim_wrapper)
            if not os.path.isfile(args.deepspeed):
                try:
                    args.deepspeed = cfgs_name_path[args.deepspeed]
                except KeyError:
                    print(f'Cannot find {args.deepspeed}')
            strategy = dict(
                type='DeepSpeedStrategy',
                config=args.deepspeed,
                gradient_clipping=gradient_clipping)
            cfg.__setitem__('strategy', strategy)
            cfg.runner_type = 'FlexibleRunner'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
