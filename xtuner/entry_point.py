# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import random
import subprocess
import sys

from mmengine.logging import print_log

import xtuner
from xtuner.tools import (chat, check_custom_dataset, copy_cfg, list_cfg,
                          log_dataset, test, train)
from xtuner.tools.data_preprocess import arxiv as arxiv_preprocess
from xtuner.tools.model_converters import merge, pth_to_hf, split

# Define valid modes
MODES = ('list-cfg', 'copy-cfg', 'log-dataset', 'check-custom-dataset',
         'train', 'test', 'chat', 'convert', 'preprocess')

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
        4-2. Merge the HuggingFace's adapter to the pretrained LLM:
            xtuner convert merge $NAME_OR_PATH_TO_LLM $NAME_OR_PATH_TO_ADAPTER $SAVE_PATH
        4-3. Split HuggingFace's LLM to the smallest sharded one:
            xtuner convert split $NAME_OR_PATH_TO_LLM $SAVE_PATH
        5. Chat with LLMs with HuggingFace's model and adapter:
            xtuner chat $NAME_OR_PATH_TO_LLM --adapter $NAME_OR_PATH_TO_ADAPTER --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
        6-1. Preprocess arxiv dataset:
            xtuner preprocess arxiv $SRC_FILE $DST_FILE --start-date $START_DATE --categories $CATEGORIES
        7-1. Log processed dataset:
            xtuner log-dataset $CONFIG
        7-2. Verify the correctness of the config file for the custom dataset.
            xtuner check-custom-dataset

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
            xtuner convert merge $NAME_OR_PATH_TO_LLM $NAME_OR_PATH_TO_ADAPTER $SAVE_PATH
        3. Split HuggingFace's LLM to the smallest sharded one:
            xtuner convert split $NAME_OR_PATH_TO_LLM $SAVE_PATH

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

modes = {
    'list-cfg': list_cfg.__file__,
    'copy-cfg': copy_cfg.__file__,
    'log-dataset': log_dataset.__file__,
    'check-custom-dataset': check_custom_dataset.__file__,
    'train': train.__file__,
    'test': test.__file__,
    'chat': chat.__file__,
    'convert': {
        'pth_to_hf': pth_to_hf.__file__,
        'merge': merge.__file__,
        'split': split.__file__,
        '--help': lambda: print_log(CONVERT_HELP_MSG, 'current'),
        '-h': lambda: print_log(CONVERT_HELP_MSG, 'current')
    },
    'preprocess': {
        'arxiv': arxiv_preprocess.__file__,
        '--help': lambda: print_log(PREPROCESS_HELP_MSG, 'current'),
        '-h': lambda: print_log(PREPROCESS_HELP_MSG, 'current')
    }
}


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
            module = modes[args[0].lower()]
            n_arg = 0
            while not isinstance(module, str) and not callable(module):
                n_arg += 1
                module = module[args[n_arg].lower()]
            if callable(module):
                module()
            else:
                nnodes = os.environ.get('NNODES', 1)
                nproc_per_node = os.environ.get('NPROC_PER_NODE', 1)
                if nnodes == 1 and nproc_per_node == 1:
                    subprocess.run(['python', module] + args[n_arg + 1:])
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
                    subprocess.run(['torchrun'] + torchrun_args + [module] +
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
