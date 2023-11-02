# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil

from mmengine.config import Config, DictAction

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
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
        print('Convert LLM to float16')
        model.llm.half()

    if cfg.model.get('llm') and (not cfg.model.get('freeze_llm', False)
                                 or cfg.model.get('lora')
                                 or cfg.model.get('llm_lora')):
        if not (cfg.model.get('visual_encoder')
                or hasattr(model, 'projector')):
            llm_path = args.save_dir
        elif 'PeftModel' in model.llm.__class__.__name__:
            llm_path = os.path.join(args.save_dir, 'adapter')
        else:
            llm_path = os.path.join(args.save_dir, 'llm')
        if 'PeftModel' in model.llm.__class__.__name__:
            print(f'Saving adapter to {llm_path}')
        else:
            print(f'Saving LLM tokenizer to {llm_path}')
            tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(llm_path)
            print(f'Saving LLM to {llm_path}')
        model.llm.save_pretrained(llm_path, max_shard_size=args.max_shard_size)

    if cfg.model.get('visual_encoder') and not cfg.model.get(
            'freeze_visual_encoder', False):
        visual_encoder_path = os.path.join(args.save_dir, 'visual_encoder')
        print(f'Saving visual_encoder to {visual_encoder_path}')
        model.visual_encoder.save_pretrained(
            visual_encoder_path, max_shard_size=args.max_shard_size)
        print(f'Saving visual_encoder processor to {visual_encoder_path}')
        processor = BUILDER.build(cfg.processor)
        processor.save_pretrained(visual_encoder_path)
    if hasattr(model, 'projector'):
        projector_path = os.path.join(args.save_dir, 'projector')
        print(f'Saving projector to {projector_path}')
        model.projector.save_pretrained(
            projector_path, max_shard_size=args.max_shard_size)

    shutil.copyfile(args.config, os.path.join(args.save_dir,
                                              'xtuner_config.py'))
    print('All done!')


if __name__ == '__main__':
    main()
