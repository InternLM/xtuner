# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timedelta
from datasets import Dataset
import torch
import torch.distributed as dist
from mmengine import mkdir_or_exist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
from torch.distributed._composable.checkpoint_activation import checkpoint
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from transformers.utils.import_utils import is_flash_attn_2_available
import json
from xtuner._lite import (AutoTokenizer, get_device, get_logger,
                          get_torch_device_module)
from xtuner._lite.accelerate import (contiguous_batching_generate,
                                     dispatch_hf_code, packed_sequence,
                                     profile_time_and_memory, unpack_sequence)
from xtuner._lite.algorithms.ppo import (CriticLoss, InferDataset, PPOPolicyLoss, RewardBuffer,
                                         PPOTokenizeFunction, RewardBufferCollator,
                                         build_actor_model, build_reward_model,
                                         compute_advantages_and_returns,
                                         compute_kl_rewards, gather_logprobs)
from xtuner._lite.algorithms.sft import SftCollator
from xtuner._lite.chat import CHAT_TEMPLATE_MAP
from xtuner._lite.datasets import (OPENAI_CONVERT_MAP, JsonlDataset, DATASET_CLS_MAP, 
                                   load_datasets, SoftPackDataset)
from xtuner._lite.parallel import (ParallelSampler, get_dp_mesh, get_fsdp_mesh,
                                   get_sp_mesh, get_tp_mesh, get_world_mesh,
                                   pad_for_sequence_parallel, setup_parallel,
                                   reduce_sequence_parallel_loss,
                                   split_for_sequence_parallel)
from xtuner._lite.parallel.fsdp import clip_grad_norm_                             
from xtuner._lite.parallel.megatron import megatron_parallelize
import xtuner._lite.algorithms.ppo.dataset as PPO_DATASET_MOD
PPO_DATASET_MOD.FASTER = True
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Model Related Settings')
    model_args.add_argument(
        '--actor', help='repo id or local path of the actor model')
    model_args.add_argument(
        '--reference', help='repo id or local path of the reference model')
    model_args.add_argument(
        '--reward', help='repo id or local path of the reward model')
    model_args.add_argument(
        '--critic', help='repo id or local path of the critic model')
    model_args.add_argument(
        '--chat-template',
        choices=CHAT_TEMPLATE_MAP.keys(),
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
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
    model_args.add_argument('--cpu-offload', action='store_true', help=(''))
    model_args.add_argument(
        '--tp-size', type=int, default=1, help='Tensor Parallel Size')
    model_args.add_argument(
        '--sp-size', type=int, default=1, help='Sequence Parallel Size')

    data_args = parser.add_argument_group('data', 'Dataset Related Settings')
    data_args.add_argument('--cache-dir', type=str)
    data_args.add_argument(
        '--datasets',
        nargs='*',
        help=('repo id or local path or dir of the datasets. For repo ids, '
              'the `dset-sources` needs to be appropriately set to '
              '`modelscope` or `huggingface`. For local dir, all json and '
              'jsonl files will be loaded by default. The type of loaded '
              'files can be controlled by setting `dset-file-type`'))
    data_args.add_argument(
        '--pretrain-datasets',
        nargs='*',
        help=('repo id or local path or dir of the datasets. For repo ids, '
              'the `dset-sources` needs to be appropriately set to '
              '`modelscope` or `huggingface`. For local dir, all json and '
              'jsonl files will be loaded by default. The type of loaded '
              'files can be controlled by setting `dset-file-type`'))
    data_args.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help=('the maximum length of each piece of data, any excess will be '
              'truncated.'))
    data_args.add_argument(
        '--pretrain-max-length',
        type=int,
        default=32768,
        help=('the maximum length of each piece of data, any excess will be '
              'truncated.'))
    data_args.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='how many subprocesses to use for data loading.')

    generate_args = parser.add_argument_group('generate',
                                              'Generate Related Settings')
    generate_args.add_argument('--max-new-tokens', type=int, default=128)
    generate_args.add_argument(
        '--gen-global-batch', type=int, default=1, help='')
    generate_args.add_argument(
        '--gen-mirco-batch', type=int, default=64, help='')
    generate_args.add_argument(
        '--gen-max-new', type=int, default=1024, help='')
    generate_args.add_argument(
        '--gen-max-prefill', type=int, default=1024, help='')
    generate_args.add_argument(
        '--gen-max-length', type=int, default=1, help='')
    generate_args.add_argument('--gen-top-k', type=int, default=0, help='')
    generate_args.add_argument('--gen-top-p', type=float, default=1.0, help='')
    generate_args.add_argument(
        '--gen-do-sample', action='store_true', help='')
    optim_args = parser.add_argument_group('optim', 'Optim Related Settings')
    optim_args.add_argument(
        '--ppo-global-batch',
        type=int,
        default=16,
        help='batch size for each optimizer step')
    optim_args.add_argument(
        '--ppo-mirco-batch',
        type=int,
        help='batch size for each optimizer step')
    optim_args.add_argument(
        '--pretrain-global-batch',
        type=int,
        default=16,
        help='batch size for each optimizer step')
    optim_args.add_argument(
        '--pretrain-mirco-batch',
        type=int,
        help='batch size for each optimizer step')

    optim_args.add_argument(
        '--actor-lr', default=4e-5, type=float, help='learning rate.')
    optim_args.add_argument(
        '--actor-min-lr', default=0, type=float, help='learning rate.')
    optim_args.add_argument(
        '--critic-lr', default=4e-5, type=float, help='learning rate.')
    optim_args.add_argument(
        '--critic-min-lr', default=0, type=float, help='learning rate.')
    optim_args.add_argument(
        '--pretrain-loss-weight', default=0.5, type=float, help='learning rate.')
    optim_args.add_argument(
        '--actor-freeze-steps', default=0, type=int, help='learning rate.')
    optim_args.add_argument(
        '--wd', default=0.01, type=float, help='weight decay.')
    optim_args.add_argument(
        '--max-grad-norm', default=1, type=float, help='gradient clipping')
    optim_args.add_argument(
        '--policy-epoch', default=1, type=int, help='training epochs.')
    optim_args.add_argument(
        '--kl-coef', default=0.01, type=float, help='training epochs.')
    optim_args.add_argument(
        '--gamma', default=1.0, type=float, help='training epochs.')
    optim_args.add_argument(
        '--gae-lambda', default=0.99, type=float, help='training epochs.')
    optim_args.add_argument(
        '--reward-min', default=-5, type=float, help='training epochs.')
    optim_args.add_argument(
        '--reward-max', default=5, type=float, help='training epochs.')
    optim_args.add_argument(
        '--reward-normalize', action='store_true', help='Set logger level to `DEBUG`')
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


def parse_dataset_info(input_string):
    import re

    # Define the regular expression pattern,
    # considering that prompt_type and prompt_options are optional.
    pattern = r'(?P<file>.+?)::(?P<ratio>[^[]+)(?:\[(?P<type>[^\]]+)\])?(?::(?P<prompt>.+))?'  # noqa: E501
    match = re.match(pattern, input_string)

    if match:
        file_path = match.group('file')
        sample_ratio = match.group('ratio') or None
        prompt_type = match.group('type') or None
        prompt_option = match.group('prompt') or None

        return file_path, float(sample_ratio), prompt_type, prompt_option
    else:
        raise ValueError('Input string format is incorrect')

class InternEVODataMapping():

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, item):
        item['input_ids'] = item['tokens']
        del item['tokens']
        if len(item['input_ids']) > self.max_length:
            item['input_ids'] = item['input_ids'][:self.max_length]
        labels = [x if x > 0 else -100 for x in item['input_ids']]
        item['input_ids'] = [abs(x) for x in item['input_ids']]
        item['labels'] = labels
        item['num_tokens'] = len(item['input_ids'])
        return item

# @logger.catch
def ppo(args):
    # TODO system prompt
    # TODO top p generate
    # TODO critic init std 0.02
    # TODO log ref kl
    # TODO 

    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    setup_parallel(tp_size=args.tp_size, sp_size=args.sp_size)
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

    rank = world_mesh.get_rank()

    if (args.gen_global_batch < args.ppo_global_batch
            or args.gen_global_batch % args.ppo_global_batch):
        raise ValueError(f'The `gen-global-batch`({args.gen_global_batch}) '
                         'should be divisible by the `ppo-global-batch`'
                         f'({args.ppo_global_batch}).')

    if args.ppo_global_batch < dp_size or args.ppo_global_batch % dp_size:
        raise ValueError(f'The `ppo-global-batch`({args.ppo_global_batch}) '
                         'should be divisible by the '
                         f'dp size({dp_size}).')

    if args.ppo_mirco_batch is None:
        args.ppo_mirco_batch = args.ppo_global_batch // dp_size

    if (args.ppo_global_batch / dp_size) % args.ppo_mirco_batch:
        raise ValueError(f'The `ppo-global-batch`({args.ppo_global_batch}) '
                         f'should be divisible by the dp_size({dp_size})'
                         f' * `ppo-mirco-batch`({args.ppo_mirco_batch})')

    if args.pretrain_mirco_batch is None:
        args.pretrain_mirco_batch = args.pretrain_global_batch // dp_size

    if (args.pretrain_datasets and args.pretrain_global_batch < dp_size
            or args.pretrain_global_batch % dp_size):
        raise ValueError(
            f'The `pretrain-global-batch`({args.pretrain_global_batch}) '
            'should be divisible by the '
            f'dp size({dp_size}).')

    if (args.pretrain_datasets and
            args.pretrain_global_batch / dp_size % args.pretrain_mirco_batch):
        raise ValueError(
            f'The `pretrain-global-batch`({args.pretrain_global_batch}) '
            f'should be divisible by the dp_size({dp_size})'
            f' * `pretrain-mirco-batch`({args.pretrain_mirco_batch})')

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0, group=world_mesh.get_group())
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

        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        logger.info('\n' + dash_line + '\nRuntime environment:' +
                    runtime_env_info + '\n' + dash_line + '\n')
    # -------------------    Environment  End  ------------------------------ #

    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################

    tokenizer = AutoTokenizer.from_pretrained(
        args.actor, trust_remote_code=True, padding_side='right')

    chat_template = CHAT_TEMPLATE_MAP[args.chat_template]
    tokenize_fn = PPOTokenizeFunction(tokenizer, chat_template)

    stop_token_ids = []
    for word in chat_template.stop_words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_ids) > 1:
            raise NotImplementError
        stop_token_ids.append(word_ids[0])

    with profile_time_and_memory('[Dataset & Dataloader]'):

        datasets = []
        for dset_info in args.datasets:
            _path, _ratio, _sys_type, _sys_prompt = parse_dataset_info(dset_info)
            _dataset = JsonlDataset(_path, _ratio, tokenize_fn)
            datasets.append(_dataset)

        msg_dataset = ConcatDataset(datasets)

        if rank == 0:
            num_samples = sum([len(dset) for dset in datasets])
            logger.info(f'[Dataset] {num_samples} samples.')

        assert is_flash_attn_2_available()
        msg_collator = SftCollator(pack_batch=True)

        msg_sampler = ParallelSampler(
            msg_dataset, fsdp_mesh, args.gen_global_batch, shuffle=True)

        msg_dataloader = DataLoader(
            msg_dataset,
            batch_size=args.gen_global_batch // fsdp_mesh.size(),
            num_workers=args.num_workers,
            # Ensure to round up or drop last based on the `global_batch_size`,
            # if you want to replace a custom sampler.
            sampler=msg_sampler,
            collate_fn=msg_collator,
            persistent_workers=args.num_workers > 0)

        if rank == 0:
            logger.info(f'[Dataloader] {len(msg_dataloader)} batches.')
            _first_batch = [msg_dataset[i] for i in range(args.gen_global_batch)]
            logger.debug(f'[Dataloader] Training Batch:\n{_first_batch}')
        

        if args.pretrain_datasets:
            pretrain_tokenize_fn = InternEVODataMapping(args.pretrain_max_length)

            DATASET_CLS_MAP['.bin'] = JsonlDataset
            pretrain_datasets = load_datasets(
                args.pretrain_datasets,
                file_types='.bin',
                map_fns=[pretrain_tokenize_fn],
                cache_dir=args.cache_dir)
            pretrain_dataset = SoftPackDataset(pretrain_datasets, target=args.pretrain_max_length)
            pretrain_sampler = ParallelSampler(
                pretrain_dataset,
                dp_mesh,
                args.pretrain_global_batch,
                shuffle=True)

            pretrain_dataloader = DataLoader(
                pretrain_dataset,
                batch_size=args.pretrain_mirco_batch,
                num_workers=args.num_workers,
                # Ensure to round up or drop last based on the `global_batch_size`,
                # if you want to replace a custom sampler.
                sampler=pretrain_sampler,
                collate_fn=msg_collator,
                persistent_workers=args.num_workers > 0)

            if rank == 0:
                logger.info(
                    f'[Pretrain Dataloader] {len(pretrain_dataloader)} batches.')
                _first_batch = [
                    pretrain_dataset[i] for i in range(args.pretrain_global_batch)
                ]
                logger.debug(
                    f'[Pretrain Dataloader] Training Batch:\n{_first_batch}')

    dist.barrier()
    # -------------------    Dataset & Dataloader  End  --------------------- #

    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################
    if args.dtype == 'auto':
        args.dtype = 'bf16' if DEVICE_MODULE.is_bf16_supported() else 'fp16'

    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        if DEVICE_MODULE.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            raise RuntimeError('The device does not support `bf16`, '
                               'please set `dtype` to `fp16`.')
    else:
        raise RuntimeError('`dtype` only supports `fp16`, `bf16` or `auto`, '
                           f'but found {args.dtype}.')

    if rank == 0:
        # Only load parameters on rank 0 to avoid each rank repeatedly loading
        # the same model into the CPU, wasting memory
        with torch.device('cpu'), profile_time_and_memory('[RANK_0 Load]'):
            rank0_actor_model = build_actor_model(args.actor, dtype)
            rank0_ref_model = build_actor_model(args.reference, dtype)
            rank0_reward_model = build_reward_model(args.reward, dtype=dtype)
            rank0_critic_model = build_reward_model(args.critic, dtype=dtype)
        
        # torch.nn.LayerNorm
        torch.nn.init.normal_(rank0_critic_model.v_head[0].weight, mean=0,std=0.02)
        rank0_critic_model.v_head[1].reset_parameters()
        torch.nn.init.normal_(rank0_critic_model.v_head[-1].weight, mean=0,std=0.02)
    else:
        
        rank0_actor_model = None
        rank0_ref_model = None
        rank0_reward_model = None
        rank0_critic_model = None

    load_sucessed = [True]
    dist.broadcast_object_list(load_sucessed, src=0)

    with torch.device('meta'):
        # In order to save CPU memory and GPU memory,
        # initialize an empty complete model on all ranks first.
        # At the same time, a non-empty complete model will be loaded
        # on the CPU of rank0.
        # After the model is parallelized, the parameters of the complete
        # model on rank0 will be loaded.
        actor_model = build_actor_model(args.actor, dtype)
        dispatch_hf_code(actor_model)
        for module in actor_model.modules():
            for p_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:

                    param_fp32 = torch.nn.Parameter(
                        param.to(dtype=torch.float32))
                    setattr(module, p_name, param_fp32)

        ref_model = build_actor_model(args.reference, dtype)
        dispatch_hf_code(ref_model)
        for param in ref_model.parameters():
            param.requires_grad = False

        critic_model = build_reward_model(args.critic, dtype=dtype)
        dispatch_hf_code(critic_model)
        for module in critic_model.modules():
            for p_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    # Ensure all numerical values in the optimizer are fp32.
                    # Don't worry about speed, FSDP will use `dtype`
                    # during forward.
                    param_fp32 = torch.nn.Parameter(
                        param.to(dtype=torch.float32))
                    setattr(module, p_name, param_fp32)

        reward_model = build_reward_model(args.reward, dtype=dtype)
        # HACK reward model originally only needs to return
        # the score of the last token of each sequence,
        # but for parallel training, we dispatched it's forward
        # to calculates the scores of all sequences.
        dispatch_hf_code(reward_model)

        for param in reward_model.parameters():
            param.requires_grad = False

        # Some reward models will add a reward token id to each sequence.
        has_reward_token = hasattr(reward_model, 'reward_token_id')

    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

    with profile_time_and_memory('[Parallelize Actor]'):
        megatron_parallelize(
            actor_model,
            rank0_actor_model,
            dp_mesh=fsdp_mesh,
            tp_mesh=tp_mesh,
            mp_policy=mp_policy,
            recompute_ratio=args.selective_recompute,
            reshard_after_forward=False)
        actor_model.train()
    dist.barrier()

    with profile_time_and_memory('[Parallelize Reference]'):
        megatron_parallelize(
            ref_model,
            rank0_ref_model,
            dp_mesh=fsdp_mesh,
            tp_mesh=tp_mesh,
            mp_policy=mp_policy,
            recompute_ratio=0)
        ref_model.eval()
    dist.barrier()

    with profile_time_and_memory('[Parallelize Critic]'):
        megatron_parallelize(
            critic_model,
            rank0_critic_model,
            dp_mesh=fsdp_mesh,
            tp_mesh=tp_mesh,
            mp_policy=mp_policy,
            recompute_ratio=args.selective_recompute)
        critic_model.train()
    dist.barrier()

    with profile_time_and_memory('[Parallelize Reward]'):
        megatron_parallelize(
            reward_model,
            rank0_reward_model,
            dp_mesh=fsdp_mesh,
            tp_mesh=tp_mesh,
            mp_policy=mp_policy,
            recompute_ratio=0)
        reward_model.eval()
    dist.barrier()
    # --------------------------    FSDP  End  ------------------------------ #

    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################
    actor_params = [p for p in actor_model.parameters() if p.requires_grad]
    actor_optimizer = AdamW(
        actor_params,
        lr=args.actor_lr)

    critic_params = [p for p in critic_model.parameters() if p.requires_grad]
    critic_optimizer = AdamW(
        critic_params,
        lr=args.critic_lr)

    total_steps = len(msg_dataloader)
    if args.pretrain_datasets:
        pretrain_steps = args.pretrain_global_batch // dp_size // args.pretrain_mirco_batch 
    else:
        pretrain_steps = 0

    if args.checkpoint_interval == -1:
        checkpoint_interval = total_steps
    elif args.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * args.checkpoint_interval)
    else:
        checkpoint_interval = int(args.checkpoint_interval)

    # ----------------    Optimizer & Scheduler End   ----------------------- #

    ###########################################################################
    #                          5. Training                                    #
    ###########################################################################

    critic_loss_fn = CriticLoss(loss_type='per_token')
    policy_loss_fn = PPOPolicyLoss(loss_type='per_token')

    msg_iterator = iter(msg_dataloader)
    if args.pretrain_datasets:
        pretrain_iterator = iter(pretrain_dataloader)

    start_step = 0
    start_train_t = time.time()
    DEVICE_MODULE.empty_cache()
    DEVICE_MODULE.reset_peak_memory_stats()
    max_memory = DEVICE_MODULE.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024**3):.1f}GB')

    for step in range(start_step, total_steps):
        
        DEVICE_MODULE.reset_peak_memory_stats()

        step_policy_loss = 0
        step_critic_loss = 0
        step_pretrain_loss = 0
        step_start_t = time.time()

        # if step < warmup_steps:
        #     actor_warmup_scheduler.step()
        #     critic_warmup_scheduler.step()
        #     cur_actor_lr = actor_warmup_scheduler.get_last_lr()[0]
        #     cur_critic_lr = critic_warmup_scheduler.get_last_lr()[0]
        # else:
        #     actor_cos_scheduler.step()
        #     critic_cos_scheduler.step()
        #     cur_actor_lr = actor_cos_scheduler.get_last_lr()[0]
        #     cur_critic_lr = critic_cos_scheduler.get_last_lr()[0]

        if step < args.actor_freeze_steps:
            # Only update the parameters of the critic model and skip pretrain.
            update_actor = False
        else:
            update_actor = True

        DEVICE_MODULE.reset_peak_memory_stats()

        data = next(msg_iterator)
        prompts = unpack_sequence(data['input_ids'].to(DEVICE),
                                  data['num_tokens'])
        infer_num_tokens = data['num_tokens'].to(DEVICE)

        # Stage 1,  Actor Model Generation
        step_avg_new_tokens = 0
        step_gen_start_t = time.time()

        # actor_model.eval()
        # gradient checkpointing will affect the generation speed.
        for block in actor_model.model.layers:
            checkpoint.state(block).enable_hook = False

        # During the generation stage, sequence parallelism was not used,
        # even when the sp size is greater than 1.
        # Per sp rank processes different prompts in parallel.
        responses = contiguous_batching_generate(
            actor_model,
            prompts,
            stop_token_ids,
            max_length=args.gen_max_length,
            max_batch_size=args.gen_mirco_batch,
            max_new_tokens=args.gen_max_new,
            do_sample=args.gen_do_sample,
            top_k=args.gen_top_k,
            top_p=args.gen_top_p,
            tp_size=args.tp_size)

        # restore gradient checkpointing
        for block in actor_model.model.layers:
            checkpoint.state(block).enable_hook = True

        # actor_model.train()

        dist.barrier()

        step_avg_new_tokens = sum([len(res) for res in responses]) / len(responses)
        step_gen_time = time.time() - step_gen_start_t
        
        prompts = [p[0].tolist() for p in prompts]

        if sp_size > 1:
            # Retrieve prompts and responses from other sp rank,
            # concatenate these data and train with sequence parallel
            sp_prompts = [None] * sp_size
            sp_responses = [None] * sp_size
            dist.all_gather_object(sp_prompts, prompts, sp_mesh.get_group())
            dist.all_gather_object(sp_responses, responses,
                                   sp_mesh.get_group())

            prompts = []
            responses = []
            for _prompts, _responses in zip(sp_prompts, sp_responses):
                prompts.extend(_prompts)
                responses.extend(_responses)

        # Stage 2,  Infer
        step_infer_start_t = time.time()
        step_infer_consumed_tokens = 0

        infer_dataset = InferDataset(prompts, responses)
        infer_dataloader = DataLoader(
            infer_dataset,
            batch_size=args.ppo_mirco_batch,
            num_workers=args.num_workers,
            collate_fn=SftCollator(pack_batch=True),
            shuffle=False,
            persistent_workers=False)

        policies = []
        for infer_packed_seq in infer_dataloader:
            
            # labels are shifted
            infer_labels = infer_packed_seq['labels'].to(DEVICE)
            infer_input_ids = infer_packed_seq['input_ids'].to(DEVICE)
            infer_num_tokens = infer_packed_seq['num_tokens'].to(DEVICE)
            infer_batch_size = infer_num_tokens.numel()

            step_infer_consumed_tokens += infer_num_tokens.sum() / sp_size / tp_size
            
            if has_reward_token:
                # Some reward models will add a reward token id to
                # each sequence.
                sequences = unpack_sequence(infer_input_ids, infer_num_tokens, dim=1)
                reward_token_id = reward_model.reward_token_id
                reward_token_id = torch.IntTensor([[reward_token_id]]).to(DEVICE)
                _sequences = []
                for seq in sequences:
                    _sequences.append(
                        torch.cat([seq, reward_token_id], dim=-1))
                reward_input_ids = torch.cat(_sequences, dim=1)

                # add 1 to the length of each sequence
                reward_num_tokens = infer_num_tokens + 1
            else:
                reward_input_ids = infer_input_ids
                reward_num_tokens = infer_num_tokens

            if sp_size > 1:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                infer_input_ids = pad_for_sequence_parallel(infer_input_ids, 0, dim=1)
                _num_pad = infer_input_ids.numel() - infer_num_tokens.sum()
                if _num_pad > 0:
                    _num_pad = torch.IntTensor([_num_pad]).to(DEVICE)
                    infer_num_tokens = torch.cat(
                        [infer_num_tokens, _num_pad], dim=-1)

                infer_input_ids = split_for_sequence_parallel(
                    infer_input_ids, dim=1, sp_mesh=sp_mesh)

                reward_input_ids = pad_for_sequence_parallel(
                    reward_input_ids, 0, dim=1)

                _num_pad = reward_input_ids.numel() - reward_num_tokens.sum()
                if _num_pad > 0:
                    _num_pad = torch.IntTensor([_num_pad]).to(DEVICE)
                    reward_num_tokens = torch.cat(
                        [reward_num_tokens, _num_pad], dim=-1)

                reward_input_ids = split_for_sequence_parallel(
                    reward_input_ids, dim=1, sp_mesh=sp_mesh)

                infer_labels = pad_for_sequence_parallel(infer_labels, -100, dim=1)
                infer_labels = split_for_sequence_parallel(
                    infer_labels, dim=1, sp_mesh=sp_mesh)

            # Some reward models will add a reward token id to each sequence,
            # requiring each sequence to increase its length by one.
            with packed_sequence(reward_num_tokens, sp_size=sp_size):
                with torch.no_grad():
                    packed_scores = reward_model(reward_input_ids, use_cache=False).logits

            with packed_sequence(infer_num_tokens, sp_size=sp_size):
                with torch.no_grad():
                    packed_ref_logits = ref_model(infer_input_ids, use_cache=False).logits
                    packed_old_logits = actor_model(infer_input_ids, use_cache=False).logits
                    packed_old_values = critic_model(infer_input_ids, use_cache=False).logits

            # The labels of prefill tokens and last token are -100.
            # HACK: (for sp) The -100 part takes the value of 0,
            # this part will be masked later.
            packed_old_logprobs = gather_logprobs(packed_old_logits, infer_labels.clip(0))
            packed_ref_logprobs = gather_logprobs(packed_ref_logits, infer_labels.clip(0))

            if sp_size > 1:
                # In sequence parallelism, before calculating loss,
                # it is necessary to restore back to the full sequence,
                # same on each sp rank.
                sp_group = sp_mesh.get_group()
                _sp_packed_old_logprobs = dist.nn.functional.all_gather(packed_old_logprobs, sp_group)
                _sp_packed_ols_values = dist.nn.functional.all_gather(
                    packed_old_values, sp_group)
                _sp_packed_ref_logprobs = dist.nn.functional.all_gather(
                    packed_ref_logprobs, sp_group)
                _sp_packed_scores = dist.nn.functional.all_gather(
                    packed_scores, sp_group)
                _sp_infer_labels = dist.nn.functional.all_gather(infer_labels, sp_group)
                _sp_input_ids = dist.nn.functional.all_gather(infer_input_ids, sp_group)

                infer_input_ids = torch.cat(_sp_input_ids, dim=1)
                infer_labels = torch.cat(_sp_infer_labels, dim=1)
                packed_old_logprobs = torch.cat(_sp_packed_old_logprobs, dim=1)
                packed_ref_logprobs = torch.cat(_sp_packed_ref_logprobs, dim=1)
                packed_old_values = torch.cat(_sp_packed_ols_values, dim=1)
                packed_scores = torch.cat(_sp_packed_scores, dim=1)

            unpacked_input_ids = unpack_sequence(infer_input_ids, infer_num_tokens, dim=1)
            unpacked_labels = unpack_sequence(infer_labels, infer_num_tokens, dim=1)

            old_logprobs = unpack_sequence(
                packed_old_logprobs, infer_num_tokens, dim=1)
            ref_logprobs = unpack_sequence(
                packed_ref_logprobs, infer_num_tokens, dim=1)
            
            old_values = unpack_sequence(
                packed_old_values, infer_num_tokens, dim=1)
            # The length of the sequence for 'scores' differs from
            # other sequences.
            reward_scores = unpack_sequence(
                packed_scores, reward_num_tokens, dim=1)

            for i in range(infer_batch_size):
                assert unpacked_input_ids[i].numel() == infer_num_tokens[i]
                assert unpacked_labels[i].numel() == infer_num_tokens[i]

                # from the last prefill token, to the second-to-last token (excluding the eos token)
                mask = unpacked_labels[i] >= 0

                _ref_logprobs = ref_logprobs[i][mask]
                _old_logprobs = old_logprobs[i][mask]
                _old_values = old_values[i][mask]
                _score = reward_scores[i][0, -1]

                _kl_rewards = compute_kl_rewards(_old_logprobs, _ref_logprobs,
                                           _score, args.kl_coef)
                _advantages, _returns = compute_advantages_and_returns(
                    _old_values, _kl_rewards, args.gamma, args.gae_lambda)

                _policy = {
                    'reward': _score.item(), 'kl_rewards': _kl_rewards.cpu(),
                    'advantages': _advantages.cpu(), 'returns': _returns.cpu(),
                    'old_values':  _old_values.cpu(),
                    'ref_logprobs':  _ref_logprobs.cpu(), 
                    'old_logprobs': _old_logprobs.cpu(),
                    'input_ids': unpacked_input_ids[i].flatten().tolist(),
                    'labels': unpacked_labels[i].flatten().tolist(),
                    'num_tokens': infer_num_tokens[i].item()
                }

                policies.append(_policy)

        step_infer_time = time.time() - step_infer_start_t

        # Stage 3, PPO
        step_ppo_start_t = time.time()

        _global_policies = [None] * dp_size
        dist.all_gather_object(_global_policies, policies, dp_mesh.get_group())

        global_policies = []
        for _rank_policies in _global_policies:
            global_policies.extend(_rank_policies)
        
        ppo_dataset = RewardBuffer(global_policies, args.reward_min, args.reward_max, args.reward_normalize, True)
        if rank == 0:
            policies_dir = os.path.join(args.work_dir, 'policies')
            mkdir_or_exist(policies_dir)
            policies_file = os.path.join(policies_dir, f'step.{step}.jsonl')
            ppo_dataset.dump_jsonl(policies_file, tokenizer, args.debug)

        ppo_loader = DataLoader(
            ppo_dataset,
            batch_size=args.ppo_mirco_batch,
            num_workers=args.num_workers,
            collate_fn = RewardBufferCollator(pack_batch=True),
            shuffle=False,
            sampler=ParallelSampler(
                ppo_dataset,
                dp_mesh,
                args.ppo_global_batch,
                shuffle=False),
            persistent_workers=False)

        # Count the total number of tokens used for training PPO on all ranks
        # It is necessary for `per-token` loss, otherwise the number of tokens
        # for each backward is unbalanced.
        global_action_tokens = ppo_dataset.num_action_tokens
        
        step_sum_values = 0
        step_action_tokens = 0
        step_avg_reward = ppo_dataset.reward_mean
        step_avg_gen_entropy = ppo_dataset.entropy_mean
        step_avg_ref_kl = ppo_dataset.kl_mean
        step_ppo_consumed_tokens = 0

        for packed_policy in ppo_loader:
            
            ppo_input_ids = packed_policy['input_ids'].to(DEVICE)
            ppo_num_tokens = packed_policy['num_tokens'].to(DEVICE)
            assert ppo_input_ids.numel() == ppo_num_tokens.sum()
            ppo_batch_size = ppo_num_tokens.numel()
            # labels are shifted
            ppo_labels = packed_policy['labels'].to(DEVICE)

            ref_logprobs = packed_policy['ref_logprobs']
            old_values = packed_policy['old_values']
            old_logprobs = packed_policy['old_logprobs']
            rewards = packed_policy['rewards']

            advantages = packed_policy['advantages']
            returns = packed_policy['returns']
            kl_rewards = packed_policy['kl_rewards']

            if sp_size > 1:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                ppo_input_ids = pad_for_sequence_parallel(ppo_input_ids, 0, dim=1)
                _num_pad = ppo_input_ids.numel() - ppo_num_tokens.sum()
                if _num_pad > 0:
                    _num_pad = torch.IntTensor([_num_pad]).to(DEVICE)
                    ppo_num_tokens = torch.cat([ppo_num_tokens, _num_pad], dim=-1)
            
                ppo_input_ids = split_for_sequence_parallel(
                    ppo_input_ids, dim=1, sp_mesh=sp_mesh)

                ppo_labels = pad_for_sequence_parallel(ppo_labels, -100, dim=1)
                ppo_labels = split_for_sequence_parallel(
                    ppo_labels, dim=1, sp_mesh=sp_mesh)

            with packed_sequence(ppo_num_tokens, sp_size=sp_size):
                packed_values = critic_model(
                    ppo_input_ids, use_cache=False).logits
                with nullcontext() if update_actor else torch.no_grad():
                    actor_logits = actor_model(ppo_input_ids, use_cache=False).logits
               
            # The labels of prefill tokens and last token are -100.
            # HACK: (for sp) The -100 part takes the value of 0,
            # this part will be masked later.
            packed_logprobs = gather_logprobs(actor_logits, ppo_labels.clip(0))
            
            if sp_size > 1:
                # In sequence parallelism, before calculating loss,
                # it is necessary to restore back to the full sequence,
                # same on each sp rank.
                sp_group = sp_mesh.get_group()
                _sp_packed_logprobs = dist.nn.functional.all_gather(packed_logprobs, sp_group)
                _sp_packed_values = dist.nn.functional.all_gather(
                    packed_values, sp_group)
                _sp_ppo_labels = dist.nn.functional.all_gather(ppo_labels, sp_group)

                ppo_labels = torch.cat(_sp_ppo_labels, dim=1)
                packed_logprobs = torch.cat(_sp_packed_logprobs, dim=1)
                packed_values = torch.cat(_sp_packed_values, dim=1)

            logprobs = unpack_sequence(
                packed_logprobs, ppo_num_tokens, dim=1)
            unpacked_labels = unpack_sequence(ppo_labels, ppo_num_tokens, dim=1)
            critic_values = unpack_sequence(
                packed_values, ppo_num_tokens, dim=1)
            
            _policy_losses = []
            _critic_losses = []
            for i in range(ppo_batch_size):
                assert unpacked_labels[i].numel() == ppo_num_tokens[i]
                # from the last prefill token, to the second-to-last token (excluding the eos token)
                action_mask = unpacked_labels[i] >= 0
                
                _values = critic_values[i][action_mask]
                _logprobs = logprobs[i][action_mask]

                _ref_logprobs = ref_logprobs[i].to(DEVICE)
                _old_logprobs = old_logprobs[i].to(DEVICE)
                _old_values = old_values[i].to(DEVICE)
                _score = rewards[i]

                _kl_rewards = kl_rewards[i].to(DEVICE)
                _advantages = advantages[i].to(DEVICE)
                _returns = returns[i].to(DEVICE)
                
                # When using per token loss, it is necessary to calibrate the 
                # loss based on the global number of action tokens.
                _policy_loss = policy_loss_fn(_logprobs, _old_logprobs,
                                              _advantages,
                                              dp_size / global_action_tokens)
                _critic_loss = critic_loss_fn(_values, _old_values, _returns,
                                              dp_size / global_action_tokens)

                _policy_losses.append(_policy_loss)
                _critic_losses.append(_critic_loss)

                step_sum_values += _old_values.sum().item()
                step_action_tokens += action_mask.sum().item()

            policy_loss = sum(_policy_losses) 
            critic_loss = sum(_critic_losses) 

            with packed_sequence(ppo_num_tokens, sp_size=sp_size):
                # The context needs to be activated when backward,
                # otherwise the recompute result is incorrect.
                if update_actor:
                    policy_loss.backward()
                critic_loss.backward()
                

            step_policy_loss += policy_loss.item()
            step_critic_loss += critic_loss.item()
            step_ppo_consumed_tokens += ppo_num_tokens.sum() / tp_size / sp_size

        critic_grad_norm = clip_grad_norm_(
            critic_params, fsdp_mesh, args.max_grad_norm)
        critic_grad_norm = critic_grad_norm.item()
        critic_optimizer.step()
        critic_optimizer.zero_grad()

        step_avg_values = step_sum_values / step_action_tokens
        actor_grad_norm = 0
        # If there is a pretrain stage, temporally not update the parameters.
        if update_actor and args.pretrain_datasets is None :
            
            actor_grad_norm = clip_grad_norm_(
                actor_params, fsdp_mesh, args.max_grad_norm)
            actor_grad_norm = actor_grad_norm.item()
            actor_optimizer.step()
            actor_optimizer.zero_grad()

        step_ppo_time = time.time() - step_ppo_start_t
    
        # State 4, Pretraining
        
        step_pt_consumed_tokens = 0
        if update_actor:
            step_pretrain_start_t = time.time() 
            for pt_step in range(pretrain_steps):
                pretrain_data = next(pretrain_iterator)
                pt_input_ids = pretrain_data['input_ids'][:, :-1].to(DEVICE)
                pt_labels = pretrain_data['labels'][:, 1:].to(DEVICE)

                pt_num_tokens = pretrain_data['num_tokens'].to(DEVICE)
                if pt_num_tokens[-1] == 1:
                    pt_num_tokens = pt_num_tokensp[:-1]
                else:
                    pt_num_tokens[-1] = pt_num_tokens[-1] - 1

                if sp_size > 1:
                    # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                    pt_input_ids = pad_for_sequence_parallel(pt_input_ids, 0, dim=1)
                    pt_num_pad = pt_input_ids.numel() - pt_num_tokens.sum()
                    if pt_num_pad > 0:
                        pt_num_pad = torch.IntTensor([pt_num_pad]).to(DEVICE)
                        pt_num_tokens = torch.cat(
                            [pt_num_tokens, pt_num_pad], dim=-1)

                    pt_input_ids = split_for_sequence_parallel(
                        pt_input_ids, dim=1, sp_mesh=sp_mesh)

                    pt_labels = pad_for_sequence_parallel(pt_labels, -100, dim=1)
                    pt_labels = split_for_sequence_parallel(
                        pt_labels, dim=1, sp_mesh=sp_mesh)
                
                with packed_sequence(pt_num_tokens, sp_size=sp_size):
                    pt_logits = actor_model(
                        input_ids=pt_input_ids, use_cache=False).logits
                    from torch.nn import functional as F
                    pt_loss = F.cross_entropy(pt_logits.squeeze(), pt_labels.squeeze(), reduction='none') # 1, seqlen
                    
                    if sp_size > 1:
                        # tokens_cal_loss = (pt_labels != -100).sum()
                        # pt_loss = reduce_sequence_parallel_loss(
                        #     pt_loss, tokens_cal_loss, sp_mesh)
                        sp_group = sp_mesh.get_group()
                        sp_pt_loss = dist.nn.functional.all_gather(pt_loss, sp_group)
                        sp_pt_labels = dist.nn.functional.all_gather(pt_labels, sp_group)

                        pt_loss = torch.cat(sp_pt_loss, dim=-1)
                        pt_labels = torch.cat(sp_pt_labels, dim=-1)
                    
                    pt_loss = pt_loss.sum() / (pt_labels != -100).sum() / pretrain_steps * args.pretrain_loss_weight
                    pt_loss.backward()

                step_pt_consumed_tokens += pt_num_tokens.sum() / sp_size / tp_size
                step_pretrain_loss += pt_loss.item()
            
            actor_grad_norm = clip_grad_norm_(
                actor_params, fsdp_mesh, args.max_grad_norm)
            actor_grad_norm = actor_grad_norm.item()
            actor_optimizer.step()
            actor_optimizer.zero_grad()

            step_pt_time = time.time() - step_pretrain_start_t
        else:
            step_pt_time = 0

        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))

        infer_tgs = int(step_infer_consumed_tokens / step_infer_time)
        ppo_tgs = int(step_ppo_consumed_tokens / step_ppo_time)
        pretrain_tgs = int(step_pt_consumed_tokens) / (step_pt_time + 1e-8)

        actor_lr = args.actor_lr if update_actor else 0.0
        max_memory = DEVICE_MODULE.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            logger.info('[Train] Step '
                        f'{step + 1}/{total_steps}  '
                        f'actor_lr: {actor_lr:.6f}  '
                        f'critic_lr: {args.critic_lr:.6f}  '
                        f'actor_grad_norm: {actor_grad_norm:.3f}  '
                        f'critic_grad_norm: {critic_grad_norm:.3f}  '
                        f'avg_reward: {step_avg_reward:.3f}  '
                        f'avg_value: {step_avg_values:.3f}  '
                        f'avg_gen_entropy: {step_avg_gen_entropy:.3f}  '
                        f'avg_ref_kl: {step_avg_ref_kl:.8f}  '
                        f'policy_loss: {step_policy_loss:.3f}  '
                        f'critic_loss: {step_critic_loss:.3f}  '
                        f'pretrain_loss: {step_pretrain_loss:.3f}  '
                        f'max_memory: {(max_memory / 1024**3):.1f}GB  '
                        f'avg_new_tokens: {int(step_avg_new_tokens)}  '
                        f'num_ppo_tokens: {int(step_ppo_consumed_tokens)}  '
                        f'num_pretrain_tokens: {int(step_pt_consumed_tokens)}  '
                        f'infer_tgs: {int(infer_tgs)}  '
                        f'ppo_tgs: {int(ppo_tgs)}  '
                        f'pretrain_tgs: {int(pretrain_tgs)}  '
                        f'gen_time: {step_gen_time:.2f}s  '
                        f'infer_time: {step_infer_time:.2f}s  '
                        f'ppo_time: {step_ppo_time:.2f}s  '
                        f'pretrain_time: {step_pt_time:.2f}s  '
                        f'total_time: {step_time:.2f}s  '
                        f'eta: {eta}')

        if is_interval(step, total_steps, checkpoint_interval):
            DEVICE_MODULE.empty_cache()

            num_digits = len(str(abs(total_steps)))
            work_dir = args.work_dir
            ckpt_dir = os.path.join(work_dir, f'ckpt-{step+1:0{num_digits}}')
            hf_dir = os.path.join(work_dir, f'hf-{step+1:0{num_digits}}')
                
            with profile_time_and_memory('[Checkpoint]'):
            
                from torch.distributed._tensor import DTensor

                if rank == 0:
                    actor_state_dict = {}
                
                for name, param in actor_model.state_dict().items():
                    if isinstance(param, DTensor):
                        with torch.no_grad():
                            full_param = param.full_tensor().cpu()
                    else:
                        full_param = param.cpu()
                    
                    if rank == 0:
                        actor_state_dict[name] = full_param
                
                if rank == 0:
                    rank0_actor_model.load_state_dict(actor_state_dict)
                    rank0_actor_model.save_pretrained(hf_dir)
                    tokenizer.save_pretrained(hf_dir)
                
                dist.barrier()
        
    train_cost_time = time.time() - start_train_t
    logger.success(f'[Train] Cost {timedelta(seconds=int(train_cost_time))}')
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':

    args = parse_args()
    ppo(args)
