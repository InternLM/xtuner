# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from mmengine import mkdir_or_exist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
from torch.distributed._composable.checkpoint_activation import checkpoint
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from transformers.utils.import_utils import is_flash_attn_2_available

from xtuner._lite import (AutoTokenizer, get_device, get_logger,
                          get_torch_device_module)
from xtuner._lite.accelerate import (contiguous_batching_generate,
                                     dispatch_hf_code, packed_sequence,
                                     profile_time_and_memory, unpack_sequence)
from xtuner._lite.algorithms.ppo import (CriticLoss, PPODataset, PPOPolicyLoss,
                                         PPOTokenizeFunction,
                                         build_actor_model, build_reward_model,
                                         compute_advantages_and_returns,
                                         compute_rewards, gather_logprobs)
from xtuner._lite.algorithms.sft import SftCollator
from xtuner._lite.chat import CHAT_TEMPLATE_MAP
from xtuner._lite.datasets import (OPENAI_CONVERT_MAP, JsonlDataset,
                                   load_datasets)
from xtuner._lite.parallel import (ParallelSampler, get_dp_mesh, get_fsdp_mesh,
                                   get_sp_mesh, get_tp_mesh, get_world_mesh,
                                   pad_for_sequence_parallel, setup_parallel,
                                   split_for_sequence_parallel)
from xtuner._lite.parallel.megatron import megatron_parallelize

logger = get_logger()

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

SUPPORT_DATA_FORMATS = OPENAI_CONVERT_MAP.keys()


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
        '--num-workers',
        type=int,
        default=1,
        help='how many subprocesses to use for data loading.')

    generate_args = parser.add_argument_group('generate',
                                              'Generate Related Settings')
    generate_args.add_argument('--max-new-tokens', type=int, default=128)
    generate_args.add_argument('--max-batch-size', type=int, default=128)
    generate_args.add_argument(
        '--gen-global-batch', type=int, default=1, help='')
    generate_args.add_argument(
        '--gen-mirco-batch', type=int, default=1, help='')
    generate_args.add_argument('--top-k', type=int, default=1, help='')
    generate_args.add_argument('--top-p', type=float, default=1, help='')

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
        '--actor-freeze-steps', default=0, type=int, help='learning rate.')
    optim_args.add_argument(
        '--wd', default=0.01, type=float, help='weight decay.')
    optim_args.add_argument(
        '--max-grad-norm', default=1, type=float, help='gradient clipping')
    optim_args.add_argument(
        '--policy-epoch', default=1, type=int, help='training epochs.')
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


# @logger.catch
def ppo(args):
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

    start_load_data_t = time.time()

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
        msg_dataset, fsdp_mesh, args.gen_global_batch, shuffle=False)

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
        pretrain_datasets = load_datasets(
            args.pretrain_datasets,
            file_types='.bin',
            cache_dir=args.cache_dir)
        pretrain_dataset = ConcatDataset(pretrain_datasets)
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

    load_data_cost_time = time.time() - start_load_data_t
    logger.info(f'[Dataset & Dataloader] Cost {load_data_cost_time:.2f}s')
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

    if rank == 0:
        # Only load parameters on rank 0 to avoid each rank repeatedly loading
        # the same model into the CPU, wasting memory
        with torch.device('cpu'), profile_time_and_memory('[RANK_0 Load]'):
            rank0_actor_model = build_actor_model(args.actor, dtype)
            rank0_ref_model = build_actor_model(args.reference, dtype)
            rank0_reward_model = build_reward_model(args.reward, dtype=dtype)
            rank0_critic_model = build_reward_model(args.critic, dtype=dtype)
    else:
        rank0_actor_model = None
        rank0_ref_model = None
        rank0_reward_model = None
        rank0_critic_model = None

    dist.barrier()

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
        lr=args.actor_lr,
        weight_decay=args.wd,
        betas=(0.9, 0.95))

    critic_params = [p for p in critic_model.parameters() if p.requires_grad]
    critic_optimizer = AdamW(
        critic_params,
        lr=args.critic_lr,
        weight_decay=args.wd,
        betas=(0.9, 0.95))

    total_steps = len(msg_dataloader)

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

    start_step = 0
    start_train_t = time.time()
    DEVICE_MODULE.empty_cache()
    DEVICE_MODULE.reset_peak_memory_stats()
    max_memory = DEVICE_MODULE.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024**3):.1f}GB')

    for step in range(start_step, total_steps):

        step_policy_loss = 0
        step_critic_loss = 0
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
        num_tokens = data['num_tokens'].to(DEVICE)

        # Stage 1,  Actor Model Generation
        step_gen_start_t = time.time()
        with profile_time_and_memory('[Generate]'):
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
                max_length=2048,
                max_batch_size=512,
                max_new_tokens=args.max_new_tokens,
                tp_size=args.tp_size)

            # restore gradient checkpointing
            for block in actor_model.model.layers:
                checkpoint.state(block).enable_hook = True

        dist.barrier()
        step_gen_time = time.time() - step_gen_start_t

        num_prefill_tokens = data['num_tokens'].sum()
        num_new_tokens = sum([len(res) for res in responses])
        gen_throughput = (num_prefill_tokens + num_new_tokens) / step_gen_time
        logger.debug(
            f'[Generate] Prefill {num_prefill_tokens} tokens, '
            f'Generate {num_new_tokens} tokens, '
            f'{num_tokens.tolist()}, {[len(res) for res in responses]}')

        prompts = [p[0].tolist() for p in prompts]

        # Count the total number of tokens used for training PPO on all ranks
        # It is necessary for `per-token` loss, otherwise the number of tokens
        # for each backward is unbalanced.
        total_ppo_tokens = sum(
            [len(r) + len(p) for r, p in zip(responses, prompts)])
        total_ppo_tokens = torch.IntTensor([total_ppo_tokens]).to(DEVICE)
        dist.all_reduce(total_ppo_tokens)

        total_response_tokens = sum([len(res) for res in responses])
        total_response_tokens = torch.IntTensor([total_response_tokens
                                                 ]).to(DEVICE)
        dist.all_reduce(total_response_tokens)

        if tp_size > 1:
            # The results within each tp group are repetitive.
            total_ppo_tokens = total_ppo_tokens / tp_size
            total_response_tokens = total_response_tokens / tp_size

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

        sequence_dataset = PPODataset(prompts, responses)
        packed_sequence_dataloader = DataLoader(
            sequence_dataset,
            batch_size=args.ppo_mirco_batch,
            num_workers=args.num_workers,
            collate_fn=SftCollator(pack_batch=True),
            shuffle=False,
            sampler=ParallelSampler(
                sequence_dataset,
                dp_mesh,
                args.ppo_global_batch,
                shuffle=False),
            persistent_workers=False)

        # Stage 2,  PPO (Proximal Policy Optimization)

        step_ppo_consumed_tokens = 0
        step_ppo_start_t = time.time()
        for packed_seq in packed_sequence_dataloader:

            input_ids = packed_seq['input_ids'].to(DEVICE)
            num_tokens = packed_seq['num_tokens'].to(DEVICE)
            # labels are shifted
            labels = packed_seq['labels'].to(DEVICE)

            if has_reward_token:
                # Some reward models will add a reward token id to
                # each sequence.
                sequences = unpack_sequence(input_ids, num_tokens, dim=1)
                reward_token_id = reward_model.reward_token_id
                reward_token_id = torch.IntTensor([reward_token_id
                                                   ]).to(DEVICE).unsqueeze(0)
                _sequences = []
                for seq in sequences:
                    _sequences.append(
                        torch.cat([seq, reward_token_id], dim=-1))
                reward_input_ids = torch.cat(_sequences, dim=1)

                num_policy_tokens = num_tokens
                # add 1 to the length of each sequence
                num_reward_tokens = num_tokens + 1
            else:
                reward_input_ids = input_ids
                num_policy_tokens = num_tokens
                num_reward_tokens = num_tokens

            if sp_size > 1:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                input_ids = pad_for_sequence_parallel(input_ids, 0, dim=1)
                _num_pad_tokens = input_ids.numel() - num_policy_tokens.sum()
                if _num_pad_tokens > 0:
                    _num_pad_tokens = torch.IntTensor([_num_pad_tokens
                                                       ]).to(DEVICE)
                    num_policy_tokens = torch.cat(
                        [num_policy_tokens, _num_pad_tokens], dim=-1)

                input_ids = split_for_sequence_parallel(
                    input_ids, dim=1, sp_mesh=sp_mesh)

                reward_input_ids = pad_for_sequence_parallel(
                    reward_input_ids, 0, dim=1)

                _num_pad_tokens = reward_input_ids.numel(
                ) - num_reward_tokens.sum()
                if _num_pad_tokens > 0:
                    _num_pad_tokens = torch.IntTensor([_num_pad_tokens
                                                       ]).to(DEVICE)
                    num_reward_tokens = torch.cat(
                        [num_reward_tokens, _num_pad_tokens], dim=-1)

                reward_input_ids = split_for_sequence_parallel(
                    reward_input_ids, dim=1, sp_mesh=sp_mesh)

                labels = pad_for_sequence_parallel(labels, -100, dim=1)
                labels = split_for_sequence_parallel(
                    labels, dim=1, sp_mesh=sp_mesh)

            # Some reward models will add a reward token id to each sequence,
            # requiring each sequence to increase its length by one.
            with profile_time_and_memory('[Infer]'):
                with packed_sequence(num_reward_tokens, sp_size=sp_size):
                    with torch.no_grad():
                        reward_scores = reward_model(reward_input_ids).logits

                with packed_sequence(num_policy_tokens, sp_size=sp_size):
                    critic_values = critic_model(
                        input_ids, use_cache=False).logits
                    with nullcontext() if update_actor else torch.no_grad():
                        actor_logits = actor_model(input_ids).logits
                    with torch.no_grad():
                        ref_logits = ref_model(input_ids).logits

            # The labels of prefill tokens and last token are -100.
            # HACK: (for sp) The -100 part takes the value of 0,
            # this part will be masked later.
            logprobs = gather_logprobs(actor_logits, labels.clip(0))
            ref_logprobs = gather_logprobs(ref_logits, labels.clip(0))

            if sp_size > 1:
                # In sequence parallelism, before calculating loss,
                # it is necessary to restore back to the full sequence,
                # same on each sp rank.
                sp_group = sp_mesh.get_group()
                sp_logprobs = dist.nn.functional.all_gather(logprobs, sp_group)
                sp_critic_values = dist.nn.functional.all_gather(
                    critic_values, sp_group)
                sp_ref_logprobs = dist.nn.functional.all_gather(
                    ref_logprobs, sp_group)
                sp_reward_scores = dist.nn.functional.all_gather(
                    reward_scores, sp_group)
                sp_labels = dist.nn.functional.all_gather(labels, sp_group)

                labels = torch.cat(sp_labels, dim=1)
                logprobs = torch.cat(sp_logprobs, dim=1)
                ref_logprobs = torch.cat(sp_ref_logprobs, dim=1)
                critic_values = torch.cat(sp_critic_values, dim=1)
                reward_scores = torch.cat(sp_reward_scores, dim=1)

            unpacked_logprobs = unpack_sequence(
                logprobs, num_policy_tokens, dim=1)
            unpacked_ref_logprobs = unpack_sequence(
                ref_logprobs, num_policy_tokens, dim=1)
            unpacked_labels = unpack_sequence(labels, num_policy_tokens, dim=1)
            unpacked_values = unpack_sequence(
                critic_values, num_policy_tokens, dim=1)
            # The length of the sequence for 'scores' differs from
            # other sequences.
            unpacked_scores = unpack_sequence(
                reward_scores, num_reward_tokens, dim=1)

            _policy_losses = []
            _critic_losses = []
            for i in range(num_tokens.numel()):
                assert unpacked_labels[i].size(1) == num_tokens[i]
                # drop prefill tokens and last token
                mask = unpacked_labels[i] >= 0

                _values = unpacked_values[i][mask].unsqueeze(0)
                _ref_logprobs = unpacked_ref_logprobs[i][mask].unsqueeze(0)
                _logprobs = unpacked_logprobs[i][mask].unsqueeze(0)
                # the generated data is only trained for one epoch,
                # `logprobs`` and `old_logprobs`` are the same
                # `values`` and `old_values`` are the same
                _old_logprobs = _logprobs.detach()
                _old_values = _values.detach()

                # unpacked_scores[i] shape : (1, seq_len)
                _score = unpacked_scores[i][:, -1].unsqueeze(0)

                _rewards = compute_rewards(_old_logprobs, _ref_logprobs,
                                           _score)
                _advantages, _returns = compute_advantages_and_returns(
                    _old_values, _rewards)

                # In the first policy epoch, the actor_model and critic_model
                # have not updated their parameters yet.
                _policy_loss = policy_loss_fn(_logprobs, _old_logprobs,
                                              _advantages,
                                              1 / total_response_tokens)
                _critic_loss = critic_loss_fn(_values, _old_values, _returns,
                                              1 / total_response_tokens)

                _policy_losses.append(_policy_loss)
                _critic_losses.append(_critic_loss)

            policy_loss = sum(_policy_losses) / sp_size
            critic_loss = sum(_critic_losses) / sp_size

            with packed_sequence(num_policy_tokens, sp_size=sp_size):
                # The context needs to be activated when backward,
                # otherwise the recompute result is incorrect.
                critic_loss.backward()
                if update_actor:
                    policy_loss.backward()

            step_policy_loss += policy_loss.item()
            step_critic_loss += critic_loss.item()
            step_ppo_consumed_tokens += num_tokens.sum() / tp_size / sp_size

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_params, args.max_grad_norm)
        critic_grad_norm = critic_grad_norm.item()
        critic_optimizer.step()
        critic_optimizer.zero_grad()
        step_ppo_time = time.time() - step_ppo_start_t

        # State 3, Pretraining

        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_params, args.max_grad_norm)
        actor_grad_norm = actor_grad_norm.item()
        actor_optimizer.step()
        actor_optimizer.zero_grad()

        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        ppo_tgs = int(step_ppo_consumed_tokens / step_time)
        actor_lr = args.actor_lr if update_actor else 0.0
        max_memory = DEVICE_MODULE.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            logger.info('[Train] Step '
                        f'{step + 1}/{total_steps}  '
                        f'actor_lr: {actor_lr:.6f}  '
                        f'critic_lr: {args.critic_lr:.6f}  '
                        f'actor_grad_norm: {actor_grad_norm:.2f}  '
                        f'critic_grad_norm: {critic_grad_norm:.2f}  '
                        f'policy_loss: {step_policy_loss:.3f}  '
                        f'critic_loss: {step_critic_loss:.3f}  '
                        f'max_memory: {(max_memory / 1024**3):.1f}GB  '
                        f'gen_throughput: {gen_throughput:.2f} '
                        f'num_ppo_tokens: {step_ppo_consumed_tokens}  '
                        f'ppo_tgs: {ppo_tgs} '
                        # f'data_time: {step_data_time:.2f}s  '
                        f'gen_time: {step_gen_time:.2f}s '
                        f'ppo_time: {step_ppo_time:.2f}s '
                        f'time: {step_time:.2f}s  '
                        f'eta: {eta}')

    train_cost_time = time.time() - start_train_t
    logger.info(f'[Train] Cost {timedelta(seconds=int(train_cost_time))}')
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':

    args = parse_args()
    ppo(args)
