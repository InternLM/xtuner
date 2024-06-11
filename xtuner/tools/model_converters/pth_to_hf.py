# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import shutil
import sys
import warnings

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import no_init_weights

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

sys.path.insert(0, os.getcwd())
sys.path.append('./')


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
        help='Save LLM in fp32. If not set, fp16 will be used by default.')
    parser.add_argument(
        '--max-shard-size',
        type=str,
        default='2GB',
        help='Only applicable for LLM. The maximum size for '
        'each sharded checkpoint.')
    parser.add_argument(
        '--safe-serialization',
        action='store_true',
        help='Indicate if using `safe_serialization`')
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
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model_name = cfg.model.type if isinstance(cfg.model.type,
                                              str) else cfg.model.type.__name__
    if 'LLaVAModel' in model_name:
        cfg.model.pretrained_pth = None

    try:
        # Initializing the model with meta-tensor can reduce unwanted memory
        # usage.
        with init_empty_weights():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*non-meta.*', category=UserWarning)
                model = BUILDER.build(cfg.model)
    except NotImplementedError as e:
        # Cannot initialize the model with meta tensor if there is lora
        # in the model.
        if 'Cannot copy out of meta tensor' in str(e):
            model = BUILDER.build(cfg.model)
        else:
            raise e

    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    for name, param in tqdm(state_dict.items(), desc='Load State Dict'):
        set_module_tensor_to_device(model, name, 'cpu', param, torch.float16)

    model.llm.config.use_cache = True

    print(f'Load PTH model from {args.pth_model}')

    if 'LLaVAModel' in model_name:
        if cfg.model.get('llm') and (not cfg.model.get('freeze_llm', False)
                                     or cfg.model.get('llm_lora')):
            if 'PeftModel' in model.llm.__class__.__name__:
                llm_path = osp.join(args.save_dir, 'llm_adapter')
                print(f'Saving LLM adapter to {llm_path}')
            else:
                llm_path = args.save_dir
                print(f'Saving LLM tokenizer to {llm_path}')
                tokenizer = BUILDER.build(cfg.tokenizer)
                tokenizer.save_pretrained(llm_path)
                print(f'Saving LLM to {llm_path}')
            if not args.fp32:
                print('Convert LLM to float16')
                model.llm.half()
            model.llm.save_pretrained(
                llm_path, max_shard_size=args.max_shard_size)

        if cfg.model.get('visual_encoder') and (
                not cfg.model.get('freeze_visual_encoder', False)
                or cfg.model.get('visual_encoder_lora')):
            if 'PeftModel' in model.visual_encoder.__class__.__name__:
                visual_encoder_path = osp.join(args.save_dir,
                                               'visual_encoder_adapter')
                print(
                    f'Saving visual_encoder adapter to {visual_encoder_path}')
            else:
                visual_encoder_path = osp.join(args.save_dir, 'visual_encoder')
                print('Saving visual_encoder image_processor to'
                      f'{visual_encoder_path}')
                image_processor = BUILDER.build(cfg.image_processor)
                image_processor.save_pretrained(visual_encoder_path)
                print(f'Saving visual_encoder to {visual_encoder_path}')
            model.visual_encoder.save_pretrained(
                visual_encoder_path, max_shard_size=args.max_shard_size)

        if hasattr(model, 'projector'):
            projector_path = osp.join(args.save_dir, 'projector')
            print(f'Saving projector to {projector_path}')
            model.projector.save_pretrained(
                projector_path, max_shard_size=args.max_shard_size)
    elif 'Reward' in model_name:
        print(f'Saving LLM tokenizer to {args.save_dir}')
        tokenizer = BUILDER.build(cfg.tokenizer)
        tokenizer.save_pretrained(args.save_dir)

        if 'PeftModel' in model.llm.__class__.__name__:
            # merge adapter
            model.llm = model.llm.merge_and_unload()
        if 'InternLM2' in model.llm.__class__.__name__:
            from modeling_internlm2_reward.modeling_internlm2 import \
                InternLM2ForRewardModel
            print(f'Saving Reward Model to {args.save_dir}')
            hf_cfg = model.llm.config
            hf_cfg.reward_token_id = model.reward_token_id if \
                model.reward_token_id is not None else cfg.reward_token_id
            if not args.fp32:
                dtype = torch.float16
            else:
                dtype = torch.float32
            with no_init_weights():
                reward_model = InternLM2ForRewardModel._from_config(
                    hf_cfg, torch_dtype=dtype)
            reward_model.model.load_state_dict(model.llm.state_dict())
            reward_model.v_head.load_state_dict(model.v_head.state_dict())
            reward_model.save_pretrained(
                args.save_dir, max_shard_size=args.max_shard_size)
            # fix auto_map in config
            with open(os.path.join(args.save_dir, 'config.json')) as fp:
                config_dict = json.load(fp)
            config_dict['auto_map'][
                'AutoModel'] = 'modeling_internlm2.InternLM2ForRewardModel'
            config_dict['auto_map'].pop('AutoModelForCausalLM', None)
            with open(os.path.join(args.save_dir, 'config.json'), 'w') as fp:
                json.dump(config_dict, fp, indent=2)
        else:
            warnings.warn(
                f'The pretrained model type: {model.llm.__class__.__name__} '
                'has no reward model class defined. Use '
                'the SequenceClassification instead.')

            hf_cfg = model.llm.config
            try:
                with no_init_weights():
                    reward_model = \
                        AutoModelForSequenceClassification.from_config(hf_cfg)
            except Exception as e:
                warnings.warn(f'Cannot find SequenceClassification class '
                              f'from transformers: {e}, \n'
                              'try to find it in the dynamic module.')
                module_file, causal_model_name = hf_cfg.auto_map[
                    'AutoModelForCausalLM'].split('.')
                seqcls_model_name = causal_model_name.split(
                    'For')[0] + 'ForSequenceClassification'
                seqcls_class = get_class_from_dynamic_module(
                    f'{module_file}.{seqcls_model_name}', model_name)
                with no_init_weights():
                    reward_model = seqcls_class(hf_cfg)
            reward_model.model.load_state_dict(model.llm.state_dict())
            reward_model.v_head.load_state_dict(model.v_head.state_dict())
            reward_model.save_pretrained(
                args.save_dir, max_shard_size=args.max_shard_size)
    else:
        llm_path = args.save_dir
        if 'PeftModel' in model.llm.__class__.__name__:
            print(f'Saving adapter to {llm_path}')
        else:
            print(f'Saving LLM tokenizer to {llm_path}')
            tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(llm_path)
            print(f'Saving LLM to {llm_path}')
        if not args.fp32:
            print('Convert LLM to float16')
            model.llm.half()
        model.llm.save_pretrained(
            llm_path,
            max_shard_size=args.max_shard_size,
            safe_serialization=args.safe_serialization)

    shutil.copyfile(args.config, osp.join(args.save_dir, 'xtuner_config.py'))
    print('All done!')


if __name__ == '__main__':
    main()
