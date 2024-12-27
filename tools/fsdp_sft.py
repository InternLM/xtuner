# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import os
import sys
import re
import time
import shutil
import requests
import gc
from collections import OrderedDict
from contextlib import nullcontext
from concurrent.futures import wait
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.distributed as dist
from torch.nn import functional as F
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from accelerate.utils import set_module_tensor_to_device
from mmengine import mkdir_or_exist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env

from peft import LoraConfig, get_peft_model
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_state_dict, set_state_dict)
from torch.distributed.checkpoint.stateful import Stateful

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import _or_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.import_utils import is_flash_attn_2_available

from xtuner._lite import (AutoTokenizer, get_device, get_logger,
                          get_torch_device_module)
from xtuner._lite.accelerate import (LORA_TARGET_MAP, dispatch_hf_code, LoadWoInit,
                                     packed_sequence, varlen_attn_is_available, profile_time_and_memory)
from xtuner._lite.algorithms.sft import SftCollator, SftTokenizeFunction
from xtuner._lite.chat import CHAT_TEMPLATE_MAP
from xtuner._lite.datasets import (DATASET_CLS_MAP, OPENAI_CONVERT_MAP,
                                   SoftPackDataset, HardPackDataset, load_datasets)
from xtuner._lite.parallel import (LengthGroupedSampler, ParallelSampler,
                                   get_dp_mesh, get_sp_mesh,
                                   pad_for_sequence_parallel,
                                   reduce_sequence_parallel_loss,
                                   setup_parallel, split_for_sequence_parallel)

from xtuner._lite.parallel import (ParallelSampler, get_dp_mesh, get_fsdp_mesh,
                                   get_sp_mesh, get_tp_mesh, get_world_mesh, get_same_data_mesh,
                                   pad_for_sequence_parallel, setup_parallel,
                                   reduce_sequence_parallel_loss,
                                   split_for_sequence_parallel)
from xtuner._lite.parallel.megatron import megatron_parallelize
from xtuner._lite.parallel.fsdp import clip_grad_norm_

gc.disable()
logger = get_logger()

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

SUPPORT_DATA_FORMATS = OPENAI_CONVERT_MAP.keys()

def log_format(rank, debug=False):

    sp_rank = get_sp_mesh().get_local_rank()
    dp_rank = get_dp_mesh().get_local_rank()
    tp_rank = get_tp_mesh().get_local_rank()
    fsdp_rank = get_fsdp_mesh().get_local_rank()

    formatter = f'[XTuner][RANK {rank}][DP {dp_rank}][SP {sp_rank}][TP {tp_rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter

def send_to_feishu(web_hook, msg):

    header = {
        "Content-Type" : "application/json;charset=UTF-8"
    }

    body = {
        "msg_type" : "text",
        "content" : { "text" : f"<at user_id=\"all\">所有人</at>{msg}"}
    }

    try:
        requests.post(url=web_hook, json=body, headers=header, timeout=1)
    except requests.exceptions.RequestException:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Model Related Settings')
    model_args.add_argument('--llm', help='repo id or local path of the model')
    model_args.add_argument(
        '-t',
        '--tokenizer',
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
    model_args.add_argument(
        '--chat-template',
        choices=CHAT_TEMPLATE_MAP.keys(),
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
    model_args.add_argument(
        '--use-lora', action='store_true', help='Apply the adapter to LLM.')
    model_args.add_argument(
        '--lora-targets',
        default=None,
        nargs='*',
        help='The names of the modules to apply the adapter to. ')
    model_args.add_argument(
        '--lora-r', default=64, type=int, help="Not updating vit's parameters")
    model_args.add_argument(
        '--lora-alpha',
        default=16,
        type=int,
        help='The alpha parameter for Lora scaling.')
    model_args.add_argument(
        '--lora-dropout',
        default=0.1,
        type=float,
        help='The dropout probability for Lora layers.')
    model_args.add_argument(
        '--lora-bias',
        default='none',
        help='The dropout probability for Lora layers.')
    model_args.add_argument(
        '--dtype',
        default='auto',
        choices=['fp16', 'bf16', 'auto'],
        help=("the dtype of the model forward. When set to 'auto', it will "
              'automatically determine whether bf16 is available, '
              'prioritizing the use of bf16.'))

    model_args.add_argument(
        '--selective-recompute',
        default=1.0,
        type=float,
        help=('the ratio of re-computation for transforemer layers. '
              'The maximum is 1; the larger the value, the less memory '
              'required for training. The default is 1, meaning all layers '
              'need to be re-computated.'))
    model_args.add_argument(
        '--shard-strategy',
        default='full',
        choices=['full', 'hybrid'],
        help=('The sharding strategy to be used for distributed training.'))
    model_args.add_argument('--cpu-offload', action='store_true', help=(''))
    model_args.add_argument('--sp-size', type=int, default=1, help='')
    data_args = parser.add_argument_group('data', 'Dataset Related Settings')
    data_args.add_argument(
        '--datasets',
        nargs='*',
        help=('repo id or local path or dir of the datasets. For repo ids, '
              'the `dset-sources` needs to be appropriately set to '
              '`modelscope` or `huggingface`. For local dir, all json and '
              'jsonl files will be loaded by default. The type of loaded '
              'files can be controlled by setting `dset-file-type`'))
    data_args.add_argument(
        '--dset-file-types',
        nargs='*',
        default=DATASET_CLS_MAP.keys(),
        choices=DATASET_CLS_MAP.keys(),
        help='the file type that needs to be loaded')
    data_args.add_argument(
        '--dset-sources',
        nargs='*',
        default=['local'],
        choices=['local', 'huggingface', 'modelscope'],
        help=('the source of each dataset; it can accept one or the same '
              'number of args as the number of `datasets`, with one arg '
              'indicating that all datasets come from the same source. '
              '`local` represents the local path, `huggingface` represents '
              'the open-source data in the Huggingface Hub, `modelscope` '
              'indicates the open-source data in the Modelscope Hub.'))
    data_args.add_argument(
        '--dset-formats',
        nargs='*',
        default=['openai'],
        help=('the format of each dataset; it can accept one or the same '
              'number of args as the number of `datasets`, with one arg '
              'indicating that all datasets are the same format.'))
    data_args.add_argument(
        '--dset-sample-ratios',
        nargs='*',
        type=float,
        default=[1.0],
        help=('the sample ratio of each dataset; it can accept one or the '
              'same number of args as the number of `datasets`, with one arg '
              'indicating that all datasets use the same sample ratio.'))
    data_args.add_argument(
        '--dset-cache-dir',
        help=('the cache dir of the loaded datasets. When the `datasets` is '
              'set, the loaded datasets will be cached to this dir. If the '
              '`datasets` are not set, the cached dataset in this dir will be '
              'loaded.'))
    data_args.add_argument(
        '--dset-pack-level',
        choices=['hard', 'soft'],
        help=('the level of data packing. When `hard`, multiple data will be '
              'packed to `max_length`, potentially causing some data to be '
              'truncated, and the length of the packed data will always '
              'be `max_length`; When `soft`, it will pack multiple  data '
              'into nearly `max_length` without truncating the data.'))
    data_args.add_argument(
        '--global-pack',
        action='store_true',
        help='A subsequence in the packed data comes from different files.')
    data_args.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help=('the maximum length of each piece of data, any excess will be '
              'truncated.'))
    data_args.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='how many subprocesses to use for data loading.')
    data_args.add_argument('--file-pattern', type=str, default=None)
    data_args.add_argument('--group-by-length', action='store_true')

    optim_args = parser.add_argument_group('optim', 'Optim Related Settings')
    optim_args.add_argument(
        '--mirco-batch-size',
        type=int,
        default=1,
        help='batch size for each forward + backward pass')
    optim_args.add_argument(
        '--global-batch-size',
        type=int,
        default=16,
        help='batch size for each optimizer step')

    optim_args.add_argument(
        '--lr', default=4e-5, type=float, help='learning rate.')
    optim_args.add_argument(
        '--lr-min', default=6e-6, type=float, help='min learning rate.')
    optim_args.add_argument(
        '--wd', default=0.01, type=float, help='weight decay.')
    optim_args.add_argument(
        '--max-grad-norm', default=1, type=float, help='gradient clipping')
    optim_args.add_argument(
        '-e', '--epochs', default=1, type=int, help='total training epochs.')
    optim_args.add_argument(
        '--warmup-ratio',
        default=0.03,
        type=float,
        help=('the proportion of training steps for learning rate warm-up in '
              'relation to the total training steps.'))

    parser.add_argument('-c', '--config', default=None)
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and checkpoints')
    parser.add_argument(
        '--feishu-webhook', default=None, help='Webhook of Feishu Group Chat Bot')
    parser.add_argument('--gc-interval', default=100, type=int)
    parser.add_argument(
        '--checkpoint-interval',
        default=-1,
        type=float,
        help=('how many steps to save a checkpoint; it can be a floating '
              'point number less than 1, or an integer greater than or equal '
              "to 1. When it's a floating point, it will be multiplied by the "
              'total number of training steps.'))
    parser.add_argument(
        '--checkpoint-max-keep',
        default=1,
        type=int,
        help=('Maximum number of saved checkpoints。'))
    parser.add_argument(
        '--checkpoint-drop-optimizer',
        action='store_true',
        help=('only model parameters are saved when saving a checkpoint. '
              'This can significantly reduce the size of checkpoint files, '
              'but the saved checkpoints cannot be resumed.'))
    parser.add_argument(
        '--log-interval', default=1, type=int, help='log interval')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed for the training')
    parser.add_argument(
        '--debug', action='store_true', help='Set logger level to `DEBUG`')
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map


def build_llm_model(args, config, world_size, dtype=torch.float32):
    with LoadWoInit():
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm, config=config, attn_implementation='flash_attention_2', 
            trust_remote_code=True)

    # Ensure all numerical values in the optimizer are fp32.
    # FSDP will use low precision during forward.
    llm.to(dtype)

    if args.use_lora:
        llm.requires_grad_(False)
        if world_size > 1:
            llm.to(dtype)

        if args.lora_targets is None:
            llm_cls = llm.__class__.__name__
            args.lora_targets = LORA_TARGET_MAP[llm_cls]
        llm_lora_cfg = LoraConfig(
            target_modules=args.lora_targets,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type='CAUSAL_LM')
        llm = get_peft_model(llm, llm_lora_cfg)

    return llm


class TrainState(Stateful):

    def __init__(self, total_steps, seed):
        super().__init__()

        self.seed = seed
        self.cur_step = -1
        self.total_steps = total_steps
        self.if_nan_skip_steps = 0

    def load_state_dict(self, state_dict):
        assert self.total_steps == state_dict['total_steps']
        self.cur_step = state_dict['current_step']
        self.if_nan_skip_steps = state_dict['if_nan_skip_steps']

    def state_dict(self):
        return {
            'seed': self.seed, 'current_step': self.cur_step, 
            'total_steps': self.total_steps, 
            'if_nan_skip_steps': self.if_nan_skip_steps
        }

    def step(self):
        self.cur_step = self.cur_step + 1

    def found_nan(self):
        self.if_nan_skip_steps += 1


def find_latest_timestamp(work_dir):
    # Initialize variables to keep track of the latest timestamp and its corresponding directory
    latest_timestamp = None

    # Iterate over all files and directories in the specified directory
    for entry in os.listdir(work_dir):
        full_path = os.path.join(work_dir, entry)
        
        # Check if the entry is a directory
        if os.path.isdir(full_path):
            try:
                # Try to interpret the directory name as a timestamp
                timestamp = datetime.strptime(entry, '%Y%m%d%H%M%S')

                # Update the latest timestamp and directory if this one is more recent
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
            except ValueError:
                # If conversion fails, skip this entry
                continue
    
    if latest_timestamp is not None:
        latest_timestamp = latest_timestamp.strftime( '%Y%m%d%H%M%S')

    return latest_timestamp


def find_checkpoints(directory, prefix='ckpt'):

    if prefix == 'ckpt':
        pattern = r'^ckpt-(\d+)$'
    elif prefix == 'hf':
        pattern = r'^hf-(\d+)$'
    else:
        raise ValueError
    
    latest_step = -1
    latest_checkpoint = None

    all_folders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    checkpoints = []
    for folder in all_folders:
        match = re.match(pattern, folder)
        if match:
            # 将文件夹名和匹配到的数字转换为整数并存储为元组
            checkpoints.append((folder, int(match.group(1))))
    
    checkpoints.sort(key=lambda x: x[1])

    return [os.path.join(directory, folder[0]) for folder in checkpoints]




# @logger.catch
def sft(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    setup_parallel(sp_size=args.sp_size, tp_size=1)
    set_random_seed(args.seed)

    dp_mesh = get_dp_mesh()
    tp_mesh = get_tp_mesh()
    sp_mesh = get_sp_mesh()
    fsdp_mesh = get_fsdp_mesh()  # dp_size * sp_size
    world_mesh = get_world_mesh()  # dp_size * sp_size * tp_size
    
    dp_size = dp_mesh.size()
    tp_size = tp_mesh.size()
    sp_size = sp_mesh.size()
    world_size = world_mesh.size()

    cpu_comm_timeout = timedelta(minutes=60)
    gloo_group = dist.new_group(backend='gloo', timeout=cpu_comm_timeout)

    if args.global_batch_size < dp_size or args.global_batch_size % dp_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         'should be divisible by the '
                         f'world_size({world_size}).')

    if (args.global_batch_size / dp_size) % args.mirco_batch_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         f'should be divisible by the world_size({world_size})'
                         f' * `mirco_batch_size`({args.mirco_batch_size})')

    rank = dist.get_rank()

    if args.resume:
        mkdir_or_exist(args.work_dir)
        timestamp = find_latest_timestamp(args.work_dir)
        
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    else:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    args.work_dir = os.path.join(args.work_dir, timestamp)
    mkdir_or_exist(args.work_dir)

    log_file = os.path.join(args.work_dir, f'rank{rank}.log')

    logger.remove()
    # Change the log format printed in the terminal
    lvl = 'DEBUG' if args.debug else 'INFO'
    logger.add(sys.stderr, level=lvl, format=log_format(rank, args.debug))
    # Change the format saved in the log file
    logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)

    if args.feishu_webhook and rank == 0:
        def log_handler(record):
            if record['level'].name == "WARNING":
                send_to_feishu(args.feishu_webhook, f"[WARNING] {record['message']}\n{args.work_dir}")
            elif record['level'].name == "TRACE":
                send_to_feishu(args.feishu_webhook, f"[TRACE] {record['message']}\n{args.work_dir}")
            elif record['level'].name == "ERROR":
                send_to_feishu(args.feishu_webhook, f"[ERROR] 任务失败\n{args.work_dir}")

        logger.add(sys.stderr, level='TRACE', filter=log_handler, catch=True)

        logger.trace('任务开始')

    logger.info(args)
    if rank == 0:
        env = collect_env()
        import transformers

        import xtuner
        env['Transformers'] = transformers.__version__
        env['XTuner'] = f'{xtuner.__version__}+{get_git_hash(digits=6)}'
        runtime_env = OrderedDict()
        runtime_env.update(env)
        runtime_env['Seed'] = args.seed
        runtime_env['World Size'] = world_size
        runtime_env['DP Size'] = dp_size
        runtime_env['SP Size'] = sp_size
        runtime_env['TP Size'] = tp_size
        # runtime_env['Distributed launcher'] = dist_launcher

        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        logger.info('\n' + dash_line + '\nRuntime environment:' +
                    runtime_env_info + '\n' + dash_line + '\n')
    # -------------------    Environment  End  ------------------------------ #

    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################

    start_load_data_t = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.llm,
        trust_remote_code=True,
        use_fast=False,
        padding_side='right')

    if args.chat_template:
        chat_template = CHAT_TEMPLATE_MAP[args.chat_template]
    else:
        chat_template = None

    tokenize_fns = []
    for dset_format in args.dset_formats:
        # If your data format is not in `SUPPORT_DATA_FORMATS`, you should
        # redefine a `tokenize_fn`, defining how to convert a piece of raw
        # data into tokenized data.
        # The tokenized data must include `input_ids`, `labels``,
        # and `num_tokens`.
        tokenize_fn = SftTokenizeFunction(tokenizer, chat_template,
                                          dset_format)
        tokenize_fns.append(tokenize_fn)

    _datasets = load_datasets(
        paths=args.datasets,
        cache_dir=args.dset_cache_dir,
        file_types=args.dset_file_types,
        sources=args.dset_sources,
        sample_ratios=args.dset_sample_ratios,
        map_fns=tokenize_fns,
        file_pattern=args.file_pattern,
        max_length=args.max_length
    )

    if args.dset_pack_level and rank == 0 and args.debug:
        # Only the tokenized datasets can count the number of tokens
        num_tokens = sum(dset.num_tokens.sum() for dset in _datasets)
        logger.debug(f'[Dataset] {num_tokens} tokens.')

    if args.dset_pack_level == 'soft':
        train_dataset = SoftPackDataset(_datasets, target=args.max_length, blend=args.global_pack)
    elif args.dset_pack_level == 'hard':
        raise NotImplementedError
    else:
        train_dataset = ConcatDataset(_datasets)

    if args.dset_pack_level and rank == 0:
        ori_samples = sum([len(dset) for dset in _datasets])
        packed_samples = len(train_dataset)
        logger.info(f'[Dataset] (Original) {ori_samples} samples.')
        logger.info(f'[Dataset] (Packed) {packed_samples} samples.')

    assert varlen_attn_is_available()
    collator = SftCollator(
        pack_batch=varlen_attn_is_available(),
        max_length=args.max_length)

    if args.group_by_length:
        sampler = LengthGroupedSampler(train_dataset, dp_mesh,
                                       args.global_batch_size)
    else:
        sampler = ParallelSampler(
            train_dataset, dp_mesh, args.global_batch_size, shuffle=True)

    gc.collect()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        # Ensure to round up or drop last based on the `global_batch_size`,
        # if you want to replace a custom sampler.
        sampler=sampler,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0)

    if rank == 0:
        logger.info(f'[Dataloader] {len(train_dataloader)} batches.')
        _first_batch = [train_dataset[i] for i in range(args.mirco_batch_size)]
        _first_batch = collator(_first_batch)
        _decoded = tokenizer.batch_decode(_first_batch['input_ids'])
        logger.debug(f'[Dataloader] Training Batch:\n{_first_batch}')
        logger.debug(f'[Dataloader] Training Batch(Decoded):\n{_decoded}')
    dist.barrier()

    gc.collect()
    load_data_cost_time = time.time() - start_load_data_t
    logger.info(f'[Dataset & Dataloader] Cost {load_data_cost_time:.2f}s')
    # -------------------    Dataset & Dataloader  End  --------------------- #

    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################

    start_model_t = time.time()

    if args.dtype == 'auto':
        args.dtype = 'bf16' if DEVICE_MODULE.is_bf16_supported() else 'fp16'

    if args.dtype == 'fp16':
        dtype = torch.float16
        autocast = torch.amp.autocast(DEVICE, enabled=True, dtype=dtype)
        scaler = ShardedGradScaler()
    elif args.dtype == 'bf16':
        if DEVICE_MODULE.is_bf16_supported():
            dtype = torch.bfloat16
            autocast = torch.amp.autocast(DEVICE, enabled=True, dtype=dtype)
            scaler = None
        else:
            raise RuntimeError('The device does not support `bf16`, '
                               'please set `dtype` to `fp16`.')
    else:
        raise RuntimeError('`dtype` only supports `fp16`, `bf16` or `auto`, '
                           f'but found {args.dtype}.')

    llm_cfg = AutoConfig.from_pretrained(args.llm, trust_remote_code=True)
    if is_flash_attn_2_available():
        llm_cfg.attn_implementation = 'flash_attention_2'

    llm_cfg.use_cache = False
    llm_cfg.torch_dtype = dtype
    
    

    # Only load parameters on rank 0 to avoid each rank repeatedly loading the
    # same model into the CPU, wasting memory
    if rank == 0:
        with torch.device('cpu'):
            rank0_llm = build_llm_model(args, llm_cfg, world_size, dtype)
    else:
        rank0_llm = None

    dist.monitored_barrier(group=gloo_group, timeout=cpu_comm_timeout)

    with torch.device('meta'):
        # Ensure all numerical values in the optimizer are fp32.
        # FSDP will use low precision during forward.
        llm = build_llm_model(args, llm_cfg, world_size, dtype)
        dispatch_hf_code(llm)
        for module in llm.modules():
            for p_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    param_fp32 = torch.nn.Parameter(
                        param.to(dtype=torch.float32))
                    setattr(module, p_name, param_fp32)

    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

    with profile_time_and_memory('[Parallelize LLM]'):
        megatron_parallelize(
            llm,
            rank0_llm,
            dp_mesh=fsdp_mesh,
            tp_mesh=tp_mesh,
            mp_policy=mp_policy,
            recompute_ratio=args.selective_recompute,
            reshard_after_forward=True)
        
        llm.train()

    dist.barrier()
    gc.collect()
    # --------------------------    FSDP  End  ------------------------------ #

    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################
    requried_grad_params = [
        param for param in llm.parameters() if param.requires_grad
    ]
    optimizer = AdamW(
        requried_grad_params,
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.95))

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    # `iter` means once forward+backward
    # `step` means once optimizer step
    # `iters_per_step` means gradient accumulative counts
    iters_per_step = global_batch_size // mirco_batch_size // dp_size
    iters_per_epoch = len(train_dataloader)
    steps_per_epoch = math.ceil(iters_per_epoch / iters_per_step)

    total_epochs = args.epochs
    total_steps = steps_per_epoch * total_epochs
    if_nan_skip_steps = 0
    train_state = TrainState(total_steps, args.seed)

    if args.checkpoint_interval == -1:
        checkpoint_interval = total_steps
    elif args.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * args.checkpoint_interval)
    else:
        checkpoint_interval = int(args.checkpoint_interval)

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min)

    start_step = 0
    gc.collect()
    # ----------------    Optimizer & Scheduler End   ----------------------- #

    ###########################################################################
    #                      5. (Optional) Resume                               #
    ###########################################################################
    if args.resume:
        

        _checkpoints = find_checkpoints(args.work_dir)

        latest_checkpoint = None

        for _ckpt_dir in reversed(_checkpoints):
            if os.path.exists(os.path.join(_ckpt_dir, '.metadata')):
                latest_checkpoint = _ckpt_dir
                break

        if latest_checkpoint:

            with profile_time_and_memory('[Resume]'):
                _options = StateDictOptions(
                    cpu_offload=True, ignore_frozen_params=True)
                (shard_model_state_dict,
                shard_optimizer_state_dict) = get_state_dict(
                    llm, optimizer, options=_options)
                state_dict = {
                    'model': shard_model_state_dict,
                    'optimizer': shard_optimizer_state_dict,
                    'train_state': train_state,
                    'warmup_scheduler': warmup_scheduler,
                    'cosine_scheduler': cosine_scheduler
                }

                # inplace state_dict
                dcp.load(
                    state_dict=state_dict,
                    checkpoint_id=latest_checkpoint,
                )

                _options = StateDictOptions(
                    cpu_offload=True, strict=False)
                set_state_dict(
                    llm,
                    optimizer,
                    model_state_dict=state_dict["model"],
                    optim_state_dict=state_dict["optimizer"],
                    options=_options
                )

            start_step = train_state.cur_step + 1
        
        else:
            logger.warning(f'There is no checkpoint available for resuming training in {args.work_dir}.')

    ###########################################################################
    #                          6. Training                                    #
    ###########################################################################
    ckpt_handle = None
    start_train_t = time.time()
    DEVICE_MODULE.empty_cache()
    DEVICE_MODULE.reset_peak_memory_stats()
    max_memory = DEVICE_MODULE.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024**3):.1f}GB')

    for step in range(start_step, total_steps):

        if is_interval(step + 1, total_steps, args.gc_interval):
            gc.collect()

        epoch = step // steps_per_epoch
        epoch_inner_step = step % steps_per_epoch
        if epoch_inner_step == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.
            train_dataloader.sampler.set_epoch(epoch, epoch_inner_step * iters_per_step )
            data_iterator = iter(train_dataloader)

        train_state.step()

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_last_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_last_lr()[0]

        DEVICE_MODULE.reset_peak_memory_stats()

        step_loss = 0
        step_data_time = 0
        step_start_t = time.time()
        step_consumed_tokens = 0

        _data_start_t = time.time()

        step_data_list = [next(data_iterator) for _ in range(iters_per_step)]
        rank_grad_tokens = 0
        for _iter in range(iters_per_step):
            _iter_data = step_data_list[_iter]
            _iter_labels = _iter_data['labels'][:, 1:]
            rank_grad_tokens += (_iter_labels >= 0).sum()
        rank_grad_tokens = rank_grad_tokens.to(DEVICE)
        dist.all_reduce(rank_grad_tokens)
        global_grad_tokens = rank_grad_tokens  / sp_size / tp_size

        
        step_data_time = time.time() - _data_start_t

        for _iter in range(iters_per_step):
            
            data = step_data_list[_iter]
            input_ids = data['input_ids'][:, :-1].to(DEVICE)

            labels = data['labels'][:, 1:].to(DEVICE)
            num_tokens = data['num_tokens'].to(DEVICE)

            if num_tokens[-1] == 1:
                num_tokens = num_tokens[:-1]
            else:
                num_tokens[-1] = num_tokens[-1] - 1

            if sp_size > 1:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                input_ids = pad_for_sequence_parallel(input_ids, 0, sp_mesh, dim=1)
                _num_pad = input_ids.numel() - num_tokens.sum()
                if _num_pad > 0:
                    _num_pad = torch.IntTensor([_num_pad]).to(DEVICE)
                    num_tokens = torch.cat([num_tokens, _num_pad], dim=-1)

                input_ids = split_for_sequence_parallel(
                    input_ids, dim=1, sp_mesh=sp_mesh)

                labels = pad_for_sequence_parallel(labels, -100,sp_mesh, dim=1)
                labels = split_for_sequence_parallel(
                    labels, dim=1, sp_mesh=sp_mesh)

            packed_ctx = packed_sequence(num_tokens, sp_mesh=sp_mesh)

            with packed_ctx, autocast if args.use_lora else nullcontext():
                loss = llm(input_ids=input_ids, labels=labels, label_shifted=True, use_cache=False).loss
                
                loss = loss * (labels >= 0).sum() / global_grad_tokens * dp_size 
               
                if scaler and args.use_lora:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            step_loss += loss.item()
            step_consumed_tokens += num_tokens.sum() / sp_size / tp_size

        step_reduced_loss = torch.Tensor([step_loss]).to(DEVICE)
        dist.all_reduce(step_reduced_loss)
        step_reduced_loss = step_reduced_loss.item() / world_size

        grad_norm = clip_grad_norm_(
            requried_grad_params, fsdp_mesh, args.max_grad_norm)

        if grad_norm.isnan() or grad_norm.isinf():
            train_state.found_nan()
            logger.warning(f"[Step {step}] The grad norm is NaN or Inf, skip this step. Skipped {train_state.if_nan_skip_steps} steps in total.")
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time)
        max_memory = DEVICE_MODULE.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            logger.info(f'[Train] (Epoch {epoch + 1}) Step '
                        f'{step + 1}/{total_steps}  '
                        f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                        f'loss(reduced): {step_reduced_loss:.3f}  '
                        f'grad_norm: {grad_norm:.2f}  '
                        f'if_nan_skip: {train_state.if_nan_skip_steps}  '
                        f'max_memory: {(max_memory / 1024**3):.1f}GB  '
                        f'text_tokens: {step_consumed_tokens}  '
                        f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                        f'time: {step_time:.2f}s  '
                        f'eta: {eta}')

        if is_interval(step, total_steps, max(1, int(total_steps * 0.1))):
            logger.trace(f'Step {step}/{total_steps}, loss {step_loss:.3f}, tgs {tgs}')

        if is_interval(step, total_steps, checkpoint_interval):
            
            num_digits = len(str(abs(total_steps)))
            work_dir = args.work_dir
            ckpt_dir = os.path.join(work_dir, f'ckpt-{step+1:0{num_digits}}')
            hf_dir = os.path.join(work_dir, f'hf-{step+1:0{num_digits}}')
            
            with profile_time_and_memory('[HF Checkpoint]'):
            
                from torch.distributed._tensor import DTensor

                if rank == 0:
                    llm_state_dict = {}
                
                for name, param in llm.state_dict().items():
                    if isinstance(param, DTensor):
                        with torch.no_grad():
                            full_param = param.full_tensor().cpu()
                    else:
                        full_param = param.cpu()
                    
                    if rank == 0:
                        llm_state_dict[name] = full_param
                
                if rank == 0:
                    rank0_llm.load_state_dict(llm_state_dict)
                    rank0_llm.save_pretrained(hf_dir)
                    tokenizer.save_pretrained(hf_dir)
                
                dist.barrier()

            saved_hf_checkpoints = find_checkpoints(args.work_dir, prefix='hf')
                
            if len(saved_hf_checkpoints) > args.checkpoint_max_keep:
                for _ckpt in saved_hf_checkpoints[:-args.checkpoint_max_keep]:
                    if rank == 0:
                        shutil.rmtree(_ckpt)
                        logger.info('[HF Checkpoint] Delete the oldest checkpoint.')


            if args.checkpoint_drop_optimizer:
                logger.warning('The saved checkpoint cannot be resumed. '
                               'If you want to save a resumable checkpoint, '
                               'please remove `--checkpoint-drop-optimizer` '
                               'from the command.')
            else:
                
                with profile_time_and_memory('[PT Checkpoint]'):
                    if ckpt_handle is not None:
                        wait([ckpt_handle])

                    # FSDP cannot be saved via torch.save
                    # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
                    _options = StateDictOptions(
                        cpu_offload=True, ignore_frozen_params=True)
                    (shard_model_state_dict,
                    shard_optimizer_state_dict) = get_state_dict(
                        llm, optimizer, options=_options)

                    state_dict = {
                        'model': shard_model_state_dict,
                        'optimizer': shard_optimizer_state_dict,
                        'train_state': train_state.state_dict(),
                        'warmup_scheduler': warmup_scheduler.state_dict(),
                        'cosine_scheduler': cosine_scheduler.state_dict()
                    }

                    mkdir_or_exist(ckpt_dir)
                    ckpt_handle = dcp.async_save(state_dict, checkpoint_id=ckpt_dir, process_group=gloo_group)

                saved_checkpoints = find_checkpoints(args.work_dir)
                
                if len(saved_checkpoints) > args.checkpoint_max_keep:
                    for _ckpt in saved_checkpoints[:-args.checkpoint_max_keep]:
                        if rank == 0:
                            shutil.rmtree(_ckpt)
                            logger.info('[PT Checkpoint] Delete the oldest checkpoint.')

    if ckpt_handle is not None:
        wait([ckpt_handle])

    logger.trace('Task Finished')

    train_cost_time = time.time() - start_train_t
    logger.info(f'[Train] Cost {timedelta(seconds=int(train_cost_time))}')
    # ------------------------    Training  End  ---------------------------- #

if __name__ == '__main__':

    args = parse_args()
    sft(args)
