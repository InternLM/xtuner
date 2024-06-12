# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import random
import subprocess
import sys

from mmengine.logging import print_log

import xtuner

# Define valid modes
MODES = ('list-cfg', 'copy-cfg', 'log-dataset', 'check-custom-dataset',
         'train', 'test', 'chat', 'convert', 'preprocess', 'mmbench',
         'eval_refcoco')

CLI_HELP_MSG = \
    f"""
    Arguments received: {str(['xtuner'] + sys.argv[1:])}. xtuner commands use the following syntax:

        xtuner MODE MODE_ARGS ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode
                ARGS (optional) are the arguments for specific command

    Some usages for xtuner commands: (See more by using -h for specific command!)

        1. List all predefined configs:
            xtuner list-cfg
        2. Copy a predefined config to a given path:
            xtuner copy-cfg $CONFIG $SAVE_FILE
        3-1. Fine-tune LLMs by a single GPU:
            xtuner train $CONFIG
        3-2. Fine-tune LLMs by multiple GPUs:
            NPROC_PER_NODE=$NGPUS NNODES=$NNODES NODE_RANK=$NODE_RANK PORT=$PORT ADDR=$ADDR xtuner dist_train $CONFIG $GPUS
        4-1. Convert the pth model to HuggingFace's model:
            xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL
        4-2. Merge the HuggingFace's adapter to the pretrained base model:
            xtuner convert merge $LLM $ADAPTER $SAVE_PATH
            xtuner convert merge $CLIP $ADAPTER $SAVE_PATH --is-clip
        4-3. Split HuggingFace's LLM to the smallest sharded one:
            xtuner convert split $LLM $SAVE_PATH
        5-1. Chat with LLMs with HuggingFace's model and adapter:
            xtuner chat $LLM --adapter $ADAPTER --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
        5-2. Chat with VLMs with HuggingFace's model and LLaVA:
            xtuner chat $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --image $IMAGE --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
        6-1. Preprocess arxiv dataset:
            xtuner preprocess arxiv $SRC_FILE $DST_FILE --start-date $START_DATE --categories $CATEGORIES
        6-2. Preprocess refcoco dataset:
            xtuner preprocess refcoco --ann-path $RefCOCO_ANN_PATH --image-path $COCO_IMAGE_PATH --save-path $SAVE_PATH
        7-1. Log processed dataset:
            xtuner log-dataset $CONFIG
        7-2. Verify the correctness of the config file for the custom dataset:
            xtuner check-custom-dataset $CONFIG
        8. MMBench evaluation:
            xtuner mmbench $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --prompt-template $PROMPT_TEMPLATE --data-path $MMBENCH_DATA_PATH
        9. Refcoco evaluation:
            xtuner eval_refcoco $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --prompt-template $PROMPT_TEMPLATE --data-path $REFCOCO_DATA_PATH
        10. List all dataset formats which are supported in XTuner

    Run special commands:

        xtuner help
        xtuner version

    GitHub: https://github.com/InternLM/xtuner
    """  # noqa: E501


CONVERT_HELP_MSG = \
    f"""
    Arguments received: {str(['xtuner'] + sys.argv[1:])}. xtuner commands use the following syntax:

        xtuner MODE MODE_ARGS ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode
                ARGS (optional) are the arguments for specific command

    Some usages for convert: (See more by using -h for specific command!)

        1. Convert the pth model to HuggingFace's model:
            xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL
        2. Merge the HuggingFace's adapter to the pretrained LLM:
            xtuner convert merge $LLM $ADAPTER $SAVE_PATH
        3. Split HuggingFace's LLM to the smallest sharded one:
            xtuner convert split $LLM $SAVE_PATH

    GitHub: https://github.com/InternLM/xtuner
    """  # noqa: E501


PREPROCESS_HELP_MSG = \
    f"""
    Arguments received: {str(['xtuner'] + sys.argv[1:])}. xtuner commands use the following syntax:

        xtuner MODE MODE_ARGS ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode
                ARGS (optional) are the arguments for specific command

    Some usages for preprocess: (See more by using -h for specific command!)

        1. Preprocess arxiv dataset:
            xtuner preprocess arxiv $SRC_FILE $DST_FILE --start-date $START_DATE --categories $CATEGORIES
        2. Preprocess refcoco dataset:
            xtuner preprocess refcoco --ann-path $RefCOCO_ANN_PATH --image-path $COCO_IMAGE_PATH --save-path $SAVE_PATH

    GitHub: https://github.com/InternLM/xtuner
    """  # noqa: E501

special = {
    'help': lambda: print_log(CLI_HELP_MSG, 'current'),
    'version': lambda: print_log(xtuner.__version__, 'current')
}
special = {
    **special,
    **{f'-{k[0]}': v
       for k, v in special.items()},
    **{f'--{k}': v
       for k, v in special.items()}
}


def list_dataset_format():
    from xtuner.tools import list_dataset_format
    return list_dataset_format.__file__


def list_cfg():
    from xtuner.tools import list_cfg
    return list_cfg.__file__


def copy_cfg():
    from xtuner.tools import copy_cfg
    return copy_cfg.__file__


def log_dataset():
    from xtuner.tools import log_dataset
    return log_dataset.__file__


def check_custom_dataset():
    from xtuner.tools import check_custom_dataset
    return check_custom_dataset.__file__


def train():
    from xtuner.tools import train
    return train.__file__


def test():
    from xtuner.tools import test
    return test.__file__


def chat():
    from xtuner.tools import chat
    return chat.__file__


def mmbench():
    from xtuner.tools import mmbench
    return mmbench.__file__


def pth_to_hf():
    from xtuner.tools.model_converters import pth_to_hf
    return pth_to_hf.__file__


def merge():
    from xtuner.tools.model_converters import merge
    return merge.__file__


def split():
    from xtuner.tools.model_converters import split
    return split.__file__


def arxiv_preprocess():
    from xtuner.tools.data_preprocess import arxiv as arxiv_preprocess
    return arxiv_preprocess.__file__


def convert_refcoco():
    from xtuner.tools.data_preprocess import convert_refcoco
    return convert_refcoco.__file__


def convert_help_msg():
    print_log(CONVERT_HELP_MSG, 'current')


def preprocess_help_msg():
    print_log(PREPROCESS_HELP_MSG, 'current')


def eval_refcoco():
    from xtuner.tools import eval_refcoco
    return eval_refcoco.__file__


modes = {
    'list-cfg': list_cfg,
    'copy-cfg': copy_cfg,
    'log-dataset': log_dataset,
    'check-custom-dataset': check_custom_dataset,
    'train': train,
    'test': test,
    'chat': chat,
    'mmbench': mmbench,
    'convert': {
        'pth_to_hf': pth_to_hf,
        'merge': merge,
        'split': split,
        '--help': convert_help_msg,
        '-h': convert_help_msg
    },
    'preprocess': {
        'arxiv': arxiv_preprocess,
        'refcoco': convert_refcoco,
        '--help': preprocess_help_msg,
        '-h': preprocess_help_msg
    },
    'eval_refcoco': eval_refcoco,
    'list-dataset-format': list_dataset_format
}

HELP_FUNCS = [preprocess_help_msg, convert_help_msg]
MAP_FILE_FUNCS = [
    list_cfg, copy_cfg, log_dataset, check_custom_dataset, train, test, chat,
    mmbench, pth_to_hf, merge, split, arxiv_preprocess, eval_refcoco,
    convert_refcoco, list_dataset_format
]


def cli():
    args = sys.argv[1:]
    if not args:  # no arguments passed
        print_log(CLI_HELP_MSG, 'current')
        return
    if args[0].lower() in special:
        special[args[0].lower()]()
        return
    elif args[0].lower() in modes:
        try:
            fn_or_dict = modes[args[0].lower()]
            n_arg = 0

            if isinstance(fn_or_dict, dict):
                n_arg += 1
                fn = fn_or_dict[args[n_arg].lower()]
            else:
                fn = fn_or_dict

            assert callable(fn)

            if fn in HELP_FUNCS:
                fn()
            else:
                slurm_launcher = False
                for i in range(n_arg + 1, len(args)):
                    if args[i] == '--launcher':
                        if i + 1 < len(args) and args[i + 1] == 'slurm':
                            slurm_launcher = True
                        break
                nnodes = int(os.environ.get('NNODES', 1))
                nproc_per_node = int(os.environ.get('NPROC_PER_NODE', 1))
                if slurm_launcher or (nnodes == 1 and nproc_per_node == 1):
                    subprocess.run(['python', fn()] + args[n_arg + 1:])
                else:
                    port = os.environ.get('PORT', None)
                    if port is None:
                        port = random.randint(20000, 29999)
                        print_log(f'Use random port: {port}', 'current',
                                  logging.WARNING)
                    torchrun_args = [
                        f'--nnodes={nnodes}',
                        f"--node_rank={os.environ.get('NODE_RANK', 0)}",
                        f'--nproc_per_node={nproc_per_node}',
                        f"--master_addr={os.environ.get('ADDR', '127.0.0.1')}",
                        f'--master_port={port}'
                    ]
                    subprocess.run(['torchrun'] + torchrun_args + [fn()] +
                                   args[n_arg + 1:] +
                                   ['--launcher', 'pytorch'])
        except Exception as e:
            print_log(f"WARNING: command error: '{e}'!", 'current',
                      logging.WARNING)
            print_log(CLI_HELP_MSG, 'current', logging.WARNING)
            return
    else:
        print_log('WARNING: command error!', 'current', logging.WARNING)
        print_log(CLI_HELP_MSG, 'current', logging.WARNING)
        return
