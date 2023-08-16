import logging
import os
import subprocess
import sys

from mmengine.logging import print_log

import xtuner
from xtuner.tools import chat, chat_hf, copy_cfg, list_cfg, test, train
from xtuner.tools.data_preprocess import arxiv as arxiv_preprocess
from xtuner.tools.model_converters import (adapter_pth2hf, merge_adapter,
                                           merge_adapter_hf, split_hf_llm)

# Define valid modes
MODES = ('list-cfg', 'copy-cfg', 'train', 'dist_train', 'test', 'chat',
         'convert', 'preprocess')

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
            NNODES=$NNODES NODE_RANK=$NODE_RANK PORT=$PORT xtuner dist_train $CONFIG $GPUS
        4-1. Chat with LLMs with HuggingFace's model and adapter:
            xtuner chat hf $NAME_OR_PATH_TO_HF_MODEL --adapter $NAME_OR_PATH_TO_HF_ADAPTER --prompt-template $PROMPT_TEMPLATE
        4-2. Chat with LLMs with xTuner's config and adapter:
            xtuner chat xtuner $CONFIG --adapter $PATH_TO_PTH_ADAPTER --prompt $PROMPT_TEMPLATE
        5-1. Convert the pth adapter to HuggingFace's adapter:
            xtuner convert adapter_pth2hf $CONFIG $PATH_TO_PTH_ADAPTER $SAVE_PATH_TO_HF_ADAPTER
        5-2. Merge the HuggingFace's adapter to the pretrained LLM:
            xtuner convert merge_adapter_hf $NAME_OR_PATH_TO_HF_MODEL $NAME_OR_PATH_TO_HF_ADAPTER $SAVE_PATH
        5-3. Merge the xTuner's adapter to the pretraiend LLM:
            xtuner convert merge_adapter $CONFIG $NAME_OR_PATH_TO_HF_ADAPTER $SAVE_PATH
        5-4. Split HuggingFace's LLM to the smallest sharded one:
            xtuner convert split_hf_llm $NAME_OR_PATH_TO_HF_MODEL $SAVE_PATH
        6-1. Preprocess arxiv dataset:
            xtuner preprocess arxiv $SRC_FILE $DST_FILE --start-date $START_DATE --categories $CATEGORIES

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

        1. Convert the pth adapter to HuggingFace's adapter:
            xtuner convert adapter_pth2hf $CONFIG $PATH_TO_PTH_ADAPTER $SAVE_PATH_TO_HF_ADAPTER
        2. Merge the HuggingFace's adapter to the pretrained LLM:
            xtuner convert merge_adapter_hf $NAME_OR_PATH_TO_HF_MODEL $NAME_OR_PATH_TO_HF_ADAPTER $SAVE_PATH
        3. Merge the xTuner's
        adapter to the pretraiend LLM:
            xtuner convert merge_adapter $CONFIG $NAME_OR_PATH_TO_HF_ADAPTER $SAVE_PATH
        4. Split HuggingFace's LLM to the smallest sharded one:
            xtuner convert split_hf_llm $NAME_OR_PATH_TO_HF_MODEL $SAVE_PATH

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


CHAT_HELP_MSG = \
    f"""
    Arguments received: {str(['xtuner'] + sys.argv[1:])}. xtuner commands use the following syntax:

        xtuner MODE MODE_ARGS ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode
                ARGS (optional) are the arguments for specific command

    Some usages for chat: (See more by using -h for specific command!)

        1. Chat with LLMs with HuggingFace's model and adapter:
            xtuner chat hf $NAME_OR_PATH_TO_HF_MODEL --adapter $NAME_OR_PATH_TO_HF_ADAPTER --prompt-template $PROMPT_TEMPLATE
        2. Chat with LLMs with xTuner's config and adapter:
            xtuner chat xtuner internlm_7b_qlora_alpaca --adapter $PATH_TO_PTH_ADAPTER --prompt $PROMPT_TEMPLATE

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
    'train': train.__file__,
    'test': test.__file__,
    'chat': {
        'hf': chat_hf.__file__,
        'xtuner': chat.__file__,
        '--help': lambda: print_log(CHAT_HELP_MSG, 'current'),
        '-h': lambda: print_log(CHAT_HELP_MSG, 'current')
    },
    'convert': {
        'adapter_pth2hf': adapter_pth2hf.__file__,
        'merge_adapter': merge_adapter.__file__,
        'merge_adapter_hf': merge_adapter_hf.__file__,
        'split_hf_llm': split_hf_llm.__file__,
        '--help': lambda: print_log(CONVERT_HELP_MSG, 'current'),
        '-h': lambda: print_log(CONVERT_HELP_MSG, 'current')
    },
    'preprocess': {
        'arxiv': arxiv_preprocess.__file__,
        '--help': lambda: print_log(PREPROCESS_HELP_MSG, 'current'),
        '-h': lambda: print_log(PREPROCESS_HELP_MSG, 'current')
    }
}

dist_modes = {'dist_train': train.__file__}


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
                subprocess.run(['python3', module] + args[n_arg + 1:])
        except Exception as e:
            print_log(f"WARNING: command error: '{e}'!", 'current',
                      logging.WARNING)
            print_log(CLI_HELP_MSG, 'current', logging.WARNING)
            return
    elif args[0].lower() in dist_modes:
        try:
            module = dist_modes[args[0].lower()]
            n_arg = 0
            while not isinstance(module, str) and not callable(module):
                n_arg += 1
                module = module[args[n_arg].lower()]
            if callable(module):
                module()
            else:
                dist_args = [
                    f"--nnodes={os.environ.get('NNODES', 1)}",
                    f"--node_rank={os.environ.get('NODE_RANK', 0)}",
                    f'--nproc_per_node={args[n_arg + 2]}',
                    f"--master_port={os.environ.get('PORT', 29500)}"
                ]
                # remove GPUS arg
                del args[n_arg + 2]
                subprocess.run(['python3', '-m', 'torch.distributed.launch'] +
                               dist_args + [module] + args[n_arg + 1:] +
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
