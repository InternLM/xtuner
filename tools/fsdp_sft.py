# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate.utils import set_module_tensor_to_device
from datasets import Dataset
from mmengine import mkdir_or_exist
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
from peft import LoraConfig, get_peft_model
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_state_dict)
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
from transformers.utils.import_utils import (is_flash_attn_2_available,
                                             is_torch_sdpa_available)

from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import (LORA_TARGET_MAP, LoadWoInit,
                                     dispatch_modules, packed_sequence)
from xtuner._lite.accelerate.fsdp import (RECOMPUTE_MODULES,
                                          all_required_grad_wrap_policy,
                                          checkpoint_check_fn, dp_lazy_init,
                                          layer_auto_wrap_policy)
from xtuner._lite.chat import CHAT_TEMPLATE_MAP
from xtuner._lite.datasets import (OPENAI_FORMAT_MAP, HardPackerForText,
                                   SoftPackerForText, TextCollator,
                                   TextOnlineTokenizeDataset,
                                   TextTokenizedDataset, TextTokenizeFunction)
from xtuner._lite.datasets.load import (LOAD_FN_MAP, load_datasets,
                                        load_from_cache)
from xtuner._lite.parallel import ParallelSampler

logger = get_logger()

SUPPORT_DATA_FORMATS = OPENAI_FORMAT_MAP.keys()


def log_format(rank, debug=False):

    formatter = f'[XTuner][RANK {rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


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
        default=LOAD_FN_MAP.keys(),
        choices=LOAD_FN_MAP.keys(),
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
        '--dset-from-cache',
        action='store_true',
        help=('Load data directly from `dset-cache-dir`. This can save time '
              'on online tokenization, but if the tokenizer changed, '
              'recaching is needed.'))
    data_args.add_argument(
        '--dset-pack-level',
        choices=['hard', 'soft'],
        help=('the level of data packing. When `hard`, multiple data will be '
              'packed to `max_length`, potentially causing some data to be '
              'truncated, and the length of the packed data will always '
              'be `max_length`; When `soft`, it will pack multiple  data '
              'into nearly `max_length` without truncating the data.'))
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
        '--checkpoint-interval',
        default=-1,
        type=float,
        help=('how many steps to save a checkpoint; it can be a floating '
              'point number less than 1, or an integer greater than or equal '
              "to 1. When it's a floating point, it will be multiplied by the "
              'total number of training steps.'))
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
        type=str,
        default=None,
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


# @logger.catch
def sft(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(args.seed)

    world_size = int(os.environ['WORLD_SIZE'])
    dp_size = world_size

    if args.global_batch_size < dp_size or args.global_batch_size % dp_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         'should be divisible by the '
                         f'world_size({world_size}).')

    if (args.global_batch_size / dp_size) % args.mirco_batch_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         f'should be divisible by the world_size({world_size})'
                         f' * `mirco_batch_size`({args.mirco_batch_size})')

    if args.dset_cache_dir and os.path.isdir(args.dset_cache_dir):
        if len(os.listdir(args.dset_cache_dir)) and not args.dset_from_cache:
            raise RuntimeError(f'`{args.dset_cache_dir}` is not an empty '
                               'folder, which may lead to inaccurate '
                               'cache results.')

    device_mesh = init_device_mesh(
        'cuda', (dp_size, ), mesh_dim_names=('dp', ))

    dp_mesh = device_mesh['dp']

    rank = dp_mesh.get_local_rank()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    args.work_dir = os.path.join(args.work_dir, timestamp)
    mkdir_or_exist(args.work_dir)

    log_file = os.path.join(args.work_dir, f'rank{rank}.log')

    # Change the log format printed in the terminal
    lvl = 'DEBUG' if args.debug else 'INFO'
    logger.add(sys.stderr, level=lvl, format=log_format(rank, args.debug))
    # Change the format saved in the log file
    logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)

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
        runtime_env['Distributed launcher'] = dist_launcher

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

    chat_template = CHAT_TEMPLATE_MAP[args.chat_template]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.llm,
        trust_remote_code=True,
        padding_side='right')

    if args.dset_from_cache:
        # packer = partial(SoftPackerForText, max_length=args.max_length)
        _datasets = load_from_cache(args.dset_cache_dir)
    else:

        tokenize_fns = []
        init_fns = []
        for dset_format in args.dset_formats:
            # If your data format is not in `SUPPORT_DATA_FORMATS`, you should
            # redefine a `tokenize_fn`, defining how to convert a piece of raw
            # data into tokenized data.
            # The tokenized data must include `input_ids`, `labels``,
            # and `num_tokens`.
            tokenize_fn = TextTokenizeFunction(tokenizer, chat_template,
                                               dset_format)

            if args.dset_cache_dir or args.dset_pack_level:
                # Before caching or packing dataset, you need to first
                # tokenize the dataset and then transform it into a
                # Huggingface `Dataset`
                init_fn = Dataset.from_list
            else:
                # Use online tokenize when there is no need to cache or pack
                # the dataset, thereby saving startup time.
                init_fn = partial(
                    TextOnlineTokenizeDataset, tokenize_fn=tokenize_fn)
                tokenize_fn = None

            tokenize_fns.append(tokenize_fn)
            init_fns.append(init_fn)

        _datasets = load_datasets(
            paths=args.datasets,
            cache_dir=args.dset_cache_dir,
            file_types=args.dset_file_types,
            sources=args.dset_sources,
            sample_ratios=args.dset_sample_ratios,
            num_proc=max(args.num_workers, 1),
            map_fns=tokenize_fns,
            init_fns=init_fns)

    if (args.dset_pack_level or args.cache_dir) and rank == 0 and args.debug:
        # # Only the tokenized datasets can count the number of tokens
        num_tokens = sum(sum(dset['num_tokens']) for dset in _datasets)
        logger.debug(f'[Dataset] {num_tokens} tokens.')

    num_datasets = len(_datasets)
    datasets = []
    if args.dset_pack_level and args.dset_pack_level == 'soft':
        pack_infos = SoftPackerForText.get_pack_infos(_datasets,
                                                      args.max_length)
        for i in range(num_datasets):
            _infos = pack_infos[i]
            _dset = _datasets[i]
            _packed_dset = SoftPackerForText(_dset, args.max_length, _infos)
            datasets.append(_packed_dset)
    elif args.dset_pack_level and args.dset_pack_level == 'hard':
        pack_infos = HardPackerForText.get_pack_infos(_datasets,
                                                      args.max_length)
        for i in range(num_datasets):
            _infos = pack_infos[i]
            _dset = _datasets[i]
            _packed_dset = HardPackerForText(_dset, args.max_length, _infos)
            datasets.append(_packed_dset)
    elif args.dset_pack_level is None and args.dset_cache_dir:
        datasets = []
        for dset in _datasets:
            datasets.append(TextTokenizedDataset(dset))
    else:
        datasets = []
        for dset in _datasets:
            datasets.append(TextOnlineTokenizeDataset(dset))

    train_dataset = ConcatDataset(datasets)

    if args.dset_pack_level and rank == 0:
        ori_samples = sum([len(dset) for dset in _datasets])
        packed_samples = len(train_dataset)
        logger.info(f'[Dataset] (Original) {ori_samples} samples.')
        logger.info(f'[Dataset] (Packed) {packed_samples} samples.')

    pack_batch = is_flash_attn_2_available()
    collator = TextCollator(pack_batch=pack_batch)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        # Ensure to round up or drop last based on the `global_batch_size`,
        # if you want to replace a custom sampler.
        sampler=ParallelSampler(
            train_dataset, dp_mesh, args.global_batch_size, shuffle=True),
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

    load_data_cost_time = time.time() - start_load_data_t
    logger.info(f'[Dataset & Dataloader] Cost {load_data_cost_time:.2f}s')
    # -------------------    Dataset & Dataloader  End  --------------------- #

    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################

    start_model_t = time.time()

    if args.dtype == 'auto':
        args.dtype = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'

    if args.dtype == 'fp16':
        dtype = torch.float16
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
        scaler = ShardedGradScaler()
    elif args.dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
            scaler = None
        else:
            raise RuntimeError('The device does not support `bf16`, '
                               'please set `dtype` to `fp16`.')
    else:
        raise RuntimeError('`dtype` only supports `fp16`，`bf16`, or `auto`, '
                           f'but found {args.dtype}.')

    llm_cfg = AutoConfig.from_pretrained(args.llm, trust_remote_code=True)
    if is_flash_attn_2_available():
        llm_cfg.attn_implementation = 'flash_attention_2'
    elif is_torch_sdpa_available():
        llm_cfg.attn_implementation = 'sdpa'

    llm_cfg.use_cache = False
    llm_cfg.torch_dtype = dtype

    with torch.device('meta'):
        with LoadWoInit():
            meta_llm = AutoModelForCausalLM.from_pretrained(
                args.llm, config=llm_cfg, trust_remote_code=True)

        # Ensure all numerical values in the optimizer are fp32.
        # FSDP will use low precision during forward.
        meta_llm.to(torch.float32)

        if pack_batch:
            dispatch_modules(meta_llm)

        if args.use_lora:
            meta_llm.requires_grad_(False)
            if world_size > 1:
                meta_llm.to(dtype)

            if args.lora_targets is None:
                meta_llm_cls = meta_llm.__class__.__name__
                args.lora_targets = LORA_TARGET_MAP[meta_llm_cls]
            llm_lora_cfg = LoraConfig(
                target_modules=args.lora_targets,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type='CAUSAL_LM')
            meta_llm = get_peft_model(meta_llm, llm_lora_cfg)

    # Only load parameters on rank 0 to avoid each rank repeatedly loading the
    # same model into the CPU, wasting memory
    if rank == 0:
        with torch.device('cpu'), LoadWoInit():
            llm = AutoModelForCausalLM.from_pretrained(
                args.llm, config=llm_cfg, trust_remote_code=True)

            # Ensure all numerical values in the optimizer are fp32.
            # FSDP will use low precision during forward.
            llm.to(torch.float32)

            if pack_batch:
                dispatch_modules(meta_llm)

            if args.use_lora:
                llm.requires_grad_(False)
                if world_size > 1:
                    llm.to(dtype)

                if args.lora_targets is None:
                    meta_llm_cls = meta_llm.__class__.__name__
                    args.lora_targets = LORA_TARGET_MAP[meta_llm_cls]
                llm_lora_cfg = LoraConfig(
                    target_modules=args.lora_targets,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias=args.lora_bias,
                    task_type='CAUSAL_LM')
                llm = get_peft_model(llm, llm_lora_cfg)
        rank0_meta_llm = copy.deepcopy(meta_llm)
        meta_llm_map = map_meta_modules(llm, meta_llm)
    else:
        meta_llm_map = None

    dist.barrier()

    param_init_fn = partial(
        dp_lazy_init, module_map=meta_llm_map, dp_mesh=dp_mesh)

    policies = [layer_auto_wrap_policy]
    if args.use_lora:
        policies.append(all_required_grad_wrap_policy)

    if args.shard_strategy == 'full':
        strategy = ShardingStrategy.FULL_SHARD
    elif args.shard_strategy == 'hybrid':
        strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise ValueError

    torch.cuda.reset_peak_memory_stats()
    shard_llm = FSDP(
        meta_llm,
        device_mesh=dp_mesh,
        sharding_strategy=strategy,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload),
        auto_wrap_policy=partial(_or_policy, policies=policies),
        mixed_precision=MixedPrecision(
            param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        param_init_fn=param_init_fn,
        sync_module_states=True,
    )

    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Model] During building the FSDP model, the peak GPU memory '
                f'is {max_memory/1024**3:.1f}GB.')

    if args.selective_recompute:
        check_fn = partial(
            checkpoint_check_fn,
            target=RECOMPUTE_MODULES,
            selective=args.selective_recompute)
        apply_activation_checkpointing(shard_llm, check_fn=check_fn)

    fsdp_cost_time = time.time() - start_model_t
    logger.info(f'[Model] Cost {fsdp_cost_time:.2f}s')
    # --------------------------    FSDP  End  ------------------------------ #

    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################
    requried_grad_params = [
        param for param in shard_llm.parameters() if param.requires_grad
    ]
    optimizer = AdamW(requried_grad_params, lr=args.lr, weight_decay=args.wd)

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
        optimizer, T_max=total_steps - warmup_steps, eta_min=0)

    start_step = 0

    # ----------------    Optimizer & Scheduler End   ----------------------- #

    ###########################################################################
    #                          5. Training                                    #
    ###########################################################################

    start_train_t = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024**3):.1f}GB')
    for step in range(start_step, total_steps):

        epoch = step // steps_per_epoch
        epoch_inner_step = step % steps_per_epoch
        if epoch_inner_step == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.

            train_dataloader.sampler.set_epoch(epoch, epoch_inner_step)
            data_iterator = iter(train_dataloader)

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_lr()[0]

        torch.cuda.reset_peak_memory_stats()

        step_loss = 0
        step_data_time = 0
        step_start_t = time.time()
        step_consumed_tokens = 0
        for _ in range(iters_per_step):

            _data_start_t = time.time()
            data = next(data_iterator)
            step_data_time += time.time() - _data_start_t

            input_ids = data['input_ids'].cuda()
            labels = data['labels'].cuda()
            attention_mask = data['attention_mask'].cuda()
            num_tokens = data['num_tokens'].cuda()

            packed_ctx = packed_sequence(num_tokens, enable=pack_batch)

            with packed_ctx:
                with autocast if args.use_lora else nullcontext():
                    outputs = shard_llm(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask)
                    avg_iter_loss = outputs.loss / iters_per_step

                if scaler and args.use_lora:
                    scaler.scale(avg_iter_loss).backward()
                else:
                    avg_iter_loss.backward()

            step_loss += avg_iter_loss.item()
            if args.dset_pack_level == 'soft':
                # During a soft pack process, the data with a length that is
                # still smaller than the max length after packing, will be
                # padded to the max length. The last element of num tokens
                # represents the count of pad tokens.
                step_consumed_tokens += num_tokens[:-1].sum()
            else:
                step_consumed_tokens += num_tokens.sum()

        grad_norm = shard_llm.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time)
        max_memory = torch.cuda.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            logger.info(f'[Train] (Epoch {epoch}) Step {step}/{total_steps}  '
                        f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                        f'grad_norm: {grad_norm:.2f}  '
                        f'max_memory: {(max_memory / 1024**3):.1f}GB  '
                        f'text_tokens: {step_consumed_tokens}  '
                        f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                        f'time: {step_time:.2f}s  '
                        f'eta: {eta}')

        if is_interval(step, total_steps, checkpoint_interval):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            max_memory = torch.cuda.max_memory_allocated()
            logger.info('[Checkpoint] Before saving checkpoint, the peak GPU '
                        f'memory is {max_memory/1024**3:.1f}GB.')

            num_digits = len(str(abs(total_steps)))
            work_dir = args.work_dir
            ckpt_dir = os.path.join(work_dir, f'ckpt-{step:0{num_digits}}')
            hf_dir = os.path.join(work_dir, f'hf-{step:0{num_digits}}')
            _options = StateDictOptions(cpu_offload=True, full_state_dict=True)

            full_model_state_dict = get_model_state_dict(
                shard_llm, options=_options)
            if rank == 0:
                saved_llm = copy.deepcopy(rank0_meta_llm)
                saved_llm.to(dtype)
                for name, param in full_model_state_dict.items():
                    set_module_tensor_to_device(saved_llm, name, 'cpu', param)

                if args.use_lora:
                    saved_llm = saved_llm.merge_and_unload()

                saved_llm.save_pretrained(hf_dir)
                tokenizer.save_pretrained(hf_dir)
                del saved_llm

            dist.barrier()
            del full_model_state_dict

            if args.checkpoint_drop_optimizer:
                logger.warning('The saved checkpoint cannot be resumed. '
                               'If you want to save a resumable checkpoint, '
                               'please remove `--checkpoint-drop-optimizer` '
                               'from the command.')
            else:
                # FSDP cannot be saved via torch.save
                # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
                _options = StateDictOptions(
                    cpu_offload=True, ignore_frozen_params=True)
                (shard_model_state_dict,
                 shard_optimizer_state_dict) = get_state_dict(
                     shard_llm, optimizer, options=_options)

                state_dict = {
                    'model': shard_model_state_dict,
                    'optimizer': shard_optimizer_state_dict,
                    'step': step,
                    'total_steps': total_steps,
                    'warmup_scheduler': warmup_scheduler.state_dict(),
                    'cosine_scheduler': cosine_scheduler.state_dict()
                }

                writer = dcp.FileSystemWriter(ckpt_dir)
                mkdir_or_exist(ckpt_dir)
                dcp.save(state_dict, writer)

            max_memory = torch.cuda.max_memory_allocated()
            logger.info('[Checkpoint] During saving checkpoint, the peak GPU '
                        f'memory is {max_memory/1024**3:.1f}GB.')

    train_cost_time = time.time() - start_train_t
    logger.info(f'[Train] Cost {timedelta(seconds=int(train_cost_time))}')
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':

    args = parse_args()
    sft(args)
