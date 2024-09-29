# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil
import warnings

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from mmengine import print_log
from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.utils import mkdir_or_exist
from tqdm import tqdm

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
        '--save-format',
        default='xtuner',
        choices=('xtuner', 'official', 'huggingface'),
        help='Only applicable for LLaVAModel. Indicate the save format.')
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
    use_meta_init = True

    if 'LLaVAModel' in model_name:
        cfg.model.pretrained_pth = None
        if args.save_format != 'xtuner':
            use_meta_init = False
    if 'Reward' in model_name:
        use_meta_init = False
        cfg.model.llm.pop('quantization_config', None)
    if hasattr(cfg.model.llm, 'quantization_config'):
        # Can not build a qlora model on meta device
        use_meta_init = False

    if use_meta_init:
        try:
            # Initializing the model with meta-tensor can reduce unwanted
            # memory usage.
            with init_empty_weights():
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore', message='.*non-meta.*', category=UserWarning)
                    model = BUILDER.build(cfg.model)
        except NotImplementedError as e:
            # Cannot initialize the model with meta tensor if the model is
            # quantized.
            if 'Cannot copy out of meta tensor' in str(e):
                model = BUILDER.build(cfg.model)
            else:
                raise e
    else:
        model = BUILDER.build(cfg.model)

    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    for name, param in tqdm(state_dict.items(), desc='Load State Dict'):
        set_module_tensor_to_device(model, name, 'cpu', param)

    model.llm.config.use_cache = True

    print_log(f'Load PTH model from {args.pth_model}', 'current')

    mkdir_or_exist(args.save_dir)

    save_pretrained_kwargs = {
        'max_shard_size': args.max_shard_size,
        'safe_serialization': args.safe_serialization
    }
    model.to_hf(
        cfg=cfg,
        save_dir=args.save_dir,
        fp32=args.fp32,
        save_pretrained_kwargs=save_pretrained_kwargs,
        save_format=args.save_format)

    shutil.copyfile(args.config, osp.join(args.save_dir, 'xtuner_config.py'))
    print_log('All done!', 'current')


if __name__ == '__main__':
    main()
