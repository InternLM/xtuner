# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from collections import OrderedDict

import torch
import torch.distributed.checkpoint as dcp
from mmengine import mkdir_or_exist
from mmengine.dist import init_dist
from torch.distributed._tensor import DTensor, Replicate, distribute_tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (get_state_dict,
                                                     set_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               parallelize_module)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader

from xtuner._lite import AutoModelForCausalLM, AutoTokenizer, get_logger
from xtuner._lite.accelerate import packed_sequence_fwd_and_bwd
from xtuner._lite.chat import ChatMessages, ChatTemplate
from xtuner._lite.datasets.text import (SoftPackTextDataset,
                                        HardPackTextDataset, TextDataset,
                                        text_collate_fn)
from xtuner._lite.parallel import ParallelSampler
from xtuner._lite.datasets.format import OPENAI_FORMAT_MAP
from xtuner._lite.datasets.load import load_datasets
from mmengine.utils.dl_utils import collect_env
from mmengine.utils import get_git_hash
from mmengine.dist import infer_launcher
from mmengine.runner import set_random_seed
# from transformers import AutoModelForCausalLM

layer_tp_plan = {
    'attention.wqkv': ColwiseParallel(),
    'attention.wo': RowwiseParallel(),
    'feed_forward.w1': ColwiseParallel(),
    'feed_forward.w2': RowwiseParallel(),
    'feed_forward.w3': ColwiseParallel(),
}

# from transformers import AutoModelForCausalLM

logger = get_logger()


def parallel_formatter(dp_rank, tp_rank, debug=False):

    formatter = f'[DP_RANK {dp_rank}][TP_RANK {tp_rank}]'
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
        '-m', '--model', help='repo id or local path of the model')
    model_args.add_argument(
        '-t',
        '--tokenizer',
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
    model_args.add_argument(
        '--dtype', 
        default='auto', 
        choices=['fp16', 'bf16', 'auto'], 
        help=("the dtype of the model forward. When set to 'auto', it will "
              "automatically determine whether bf16 is available, "
              "prioritizing the use of bf16."))
    
    model_args.add_argument(
        '--selective-recompute',
        default=1.0,
        type=float,
        help=('the ratio of re-computation for transforemer layers. '
              'The maximum is 1; the larger the value, the less memory '
              'required for training. The default is 1, meaning all layers '
              'need to be re-computated.'))
    model_args.add_argument(
        '--tp-size', type=int, default=1, help='tensor Parallel Size')
    model_args.add_argument(
        '--sp-size', type=int, default=1, help='sequence Parallel Size')

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
        default=['json', 'jsonl'],
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
        '--dset-pack-level',
        choices=['fixed', 'dynamic'],
        help=('the level of data packing. When `fixed`, multiple data will be '
              'packed to `max_length`, potentially causing some data to be '
              'truncated, and the length of the packed data will always '
              'be `max_length`; When `dynamic`, it will pack multiple  data '
              'into nearly `max_length` without truncating the data.'))
    data_args.add_argument(
        '--dset-file-alone',
        action='store_true',
        help=('each file is treated as a separate dataset. This setting only '
              'affects the training results when `dset-pack-level` is set. '
              'Each piece of data in the concatenated data comes from the '
              'same file at this time.'))
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
        help='batch size for each backward pass')
    optim_args.add_argument(
        '--global-batch-size',
        type=int,
        default=16,
        help='batch size for each parameter update')

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
        default=10,
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
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


# @logger.catch
def sft(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(args.seed)
    
    world_size = int(os.environ['WORLD_SIZE'])
    dp_size = world_size // args.tp_size
    tp_size = args.tp_size
    
    device_mesh = init_device_mesh(
        'cuda', (dp_size, tp_size), mesh_dim_names=('dp', 'tp'))
    tp_mesh = device_mesh['tp']
    dp_mesh = device_mesh['dp']

    dp_rank = dp_mesh.get_local_rank()
    tp_rank = tp_mesh.get_local_rank()

    mkdir_or_exist(args.work_dir)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    log_file = os.path.join(args.work_dir,
                            f'{timestamp}.dp{dp_rank}.tp{tp_rank}.log')
    formatter = parallel_formatter(dp_rank, tp_rank)
    # Change the log format printed in the terminal
    logger.add(sys.stderr, format=formatter)
    # Change the format saved in the log file
    logger.add(log_file, format=formatter, backtrace=True, catch=True)
    
    if dp_rank == 0 and tp_rank == 0:
        env = collect_env()
        import transformers, xtuner
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
        logger.info('\n' + dash_line + 
                    '\nRuntime environment:' + runtime_env_info + '\n' +
                    dash_line + '\n')
    # -------------------    Environment  End  ------------------------------ #
    
    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################
    
    start_load_data_t = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.model,
        trust_remote_code=True,
        padding_side='right')

    chat_template = ChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>\n',
        stop_words=['<|im_end|>'])
    
    
    # Define how to convert raw data into OpenAI format.
    # If you want to use other data format, you need to define the 
    # corresponding `foramt_fn` and `tokenize fn`.
    format_fns = [OPENAI_FORMAT_MAP[f] for f in args.dset_formats]

    # The following function is used to tokenize a piece of data in the OpenAI 
    # format.
    # If your data format is different, you should redefine a `tokenize_fn`
    # The tokenized data must include `input_ids`, `labels``, and `num_tokens`.
    def tokenize_fn(item):
        msg = ChatMessages.from_dict(item)
        tokenized = msg.tokenize(tokenizer, chat_template)
        return tokenized

    datasets = load_datasets(
        paths=args.datasets,
        sources=args.dset_sources, 
        format_fns=format_fns,
        sample_ratios=args.dset_sample_ratios,
        split_per_file=args.dset_file_alone,
        tokenize_fn=tokenize_fn)

    cat_datasets = []
    for dset in datasets:
        if args.dset_pack_level and args.dset_pack_level == 'fixed':
            _dset = HardPackTextDataset(dset, args.max_length)
        elif args.dset_pack_level and args.dset_pack_level == 'dynamic':
            _dset = SoftPackTextDataset(dset, args.max_length)
        elif not args.dset_pack_level:
            _dset = TextDataset(dset)
        else:
            raise RuntimeError

        cat_datasets.append(_dset)

    all_dataset = ConcatDataset(cat_datasets)
    
    train_dataloader = DataLoader(
        all_dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        sampler=ParallelSampler(all_dataset, dp_mesh, shuffle=True),
        collate_fn=text_collate_fn,
        persistent_workers=args.num_workers > 0)
    
    load_data_cost_time = time.time() - start_load_data_t
    logger.info(f'Build dataloader cost {load_data_cost_time:.2f}s')
    # -------------------    Dataset & Dataloader  End  --------------------- #

    
    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################
    
    start_fsdp_t = time.time()
    
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            dtype == torch.bfloat16
        else:
            raise RuntimeError("The device does not support `bf16`, "
                               "please set `dtype` to `fp16`.")
    elif args.dtype == 'auto':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        raise RuntimeError("`dtype` only supports `fp16`，`bf16`, or `auto`, "
                           f"but found {args.dtype}.")
    
    
    with torch.device('meta'):
        # model parameters must be in fp32.
        # this ensures that all numerical values in the optimizer are in fp32.
        # FSDP will use low precision during forward.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.float32)
        model.config.use_cache = False

    
    # Only load parameters on rank 0 to avoid each rank repeatedly loading the 
    # same model into the CPU, wasting memory
    if dp_rank == 0 and tp_rank ==0:
        with torch.device('cpu'):
            master_model = AutoModelForCausalLM.from_pretrained(
                args.model, trust_remote_code=True, torch_dtype=dtype)

        master_mods = {name: mod for name, mod in master_model.named_modules()}
        master_mod_map = {
            mod: master_mods[name]
            for name, mod in model.named_modules()
        }
    else:
        master_mod_map = None

    if args.tp_size > 1:
        for layer in model.model.layers:
            attention = layer.attention
            attention.num_heads = attention.num_heads // tp_mesh.size()
            attention.hidden_size = attention.hidden_size // tp_mesh.size()
            parallelize_module(
                module=layer,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan,
            )

        model = parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan={
                'model.tok_embeddings':
                RowwiseParallel(input_layouts=Replicate(), ),
                'output': ColwiseParallel(output_layouts=Replicate(), ),
            })



    @torch.no_grad
    def lazy_param_init_fn(module):
        device = torch.cuda.current_device()
        module.to_empty(device=torch.cuda.current_device(), recurse=False)

        if dp_mesh.get_local_rank() == 0 and tp_mesh.get_local_rank() == 0:
            master_module = master_mod_map[module]
            master_params = {
                name: param
                for name, param in master_module.named_parameters(
                    recurse=False)
            }
            master_buffers = {
                name: buffer
                for name, buffer in master_module.named_buffers(recurse=False)
            }
        else:
            master_params = None
            master_buffers = None

        if dp_mesh.get_local_rank() == 0:

            for name, param in module.named_parameters(recurse=False):

                if isinstance(param, DTensor):

                    p_full = param.full_tensor()
                    if tp_mesh.get_local_rank() == 0:
                        p_copy = master_params[name]
                        p_copy = p_copy.to(device).to(torch.float32)
                    else:
                        p_copy = torch.empty_like(p_full)

                    mesh = param.device_mesh
                    placements = param.placements

                    p_dtensor = distribute_tensor(p_copy, mesh, placements)
                    param.data.copy_(p_dtensor)

                else:
                    if tp_mesh.get_local_rank() == 0:
                        # breakpoint()
                        p_copy = master_params[name]
                        p_copy = p_copy.to(device).to(torch.float32)
                    else:
                        p_copy = torch.empty_like(param)

                    tp_group = tp_mesh.get_group()
                    torch.distributed.broadcast(p_copy, 0, tp_group)
                    param.data.copy_(p_copy)

            for name, buffer in module.named_buffers(recurse=False):

                if isinstance(buffer, DTensor):

                    b_full = buffer.full_tensor()
                    if tp_mesh.get_local_rank() == 0:
                        b_copy = master_buffers[name]
                        b_copy = b_copy.to(device).to(torch.float32)
                    else:
                        b_copy = torch.empty_like(b_full)

                    mesh = buffer.device_mesh
                    placements = buffer.placements

                    b_dtensor = distribute_tensor(b_copy, mesh, placements)
                    buffer.data.copy_(b_dtensor)

                else:
                    if tp_mesh.get_local_rank() == 0:
                        b_copy = master_buffers[name]
                        b_copy = b_copy.to(device).to(torch.float32)
                    else:
                        b_copy = torch.empty_like(buffer)

                    tp_group = tp_mesh.get_group()
                    torch.distributed.broadcast(b_copy, 0, tp_group)
                    buffer.data.copy_(b_copy)

    torch.cuda.reset_peak_memory_stats()
    shard_model = FSDP(
        model,
        device_mesh=dp_mesh,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        param_init_fn=lazy_param_init_fn,
        sync_module_states=True,
    )
    max_memory = torch.cuda.max_memory_allocated()
    logger.info('The peak GPU memory when building the FSDP model is '
                f'{max_memory/1024**3:.1f}GB.')
    
    if args.selective_recompute:

        def checkpoint_check_fn(submodule, target='InternLM2DecoderLayer'):
            ret = False
            if type(submodule).__name__ == target:
                if random.uniform(0, 1) < args.selective_recompute:
                    ret = True
            return ret

        apply_activation_checkpointing(
            shard_model, check_fn=checkpoint_check_fn)
        
    fsdp_cost_time = time.time() - start_fsdp_t
    logger.info(f'Build FSDP model cost {fsdp_cost_time:.2f}s')
    # --------------------------    FSDP  End  ------------------------------ #
    
    
    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################
    optimizer = AdamW(
        shard_model.parameters(), lr=args.lr, weight_decay=args.wd)

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    # `iter` means onec forward+backward
    # `step` means once optimizer step
    # `per_step_iters` means gradient accumulative counts
    per_step_iters = global_batch_size // mirco_batch_size // dp_size
    per_epoch_iters = len(train_dataloader)
    per_epoch_steps = math.ceil(per_epoch_iters / per_step_iters)

    total_epochs = args.epochs
    total_steps = per_epoch_steps * total_epochs
    
    if args.checkpoint_interval < 1:
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
    #                        5. Resume (Optional)                             #
    ###########################################################################
    if args.resume:
        torch.cuda.reset_peak_memory_stats()
        start_resume_t = time.time()
        model_state_dict, optim_state_dict = get_state_dict(
            shard_model, optimizer)
        warump_state_dict = warmup_scheduler.state_dict()
        cosine_state_dict = cosine_scheduler.state_dict()

        state_dict = {
            'model': model_state_dict,
            'optimizer': optim_state_dict,
            'step': start_step,
            'total_steps': total_steps,
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'cosine_scheduler': cosine_scheduler.state_dict()
        }
        reader = dcp.FileSystemReader(args.resume)
        dcp.load(state_dict, reader)

        if state_dict['total_steps'] != total_steps:
            raise RuntimeError

        set_state_dict(
            shard_model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict)

        warmup_scheduler.load_state_dict(warump_state_dict)
        cosine_scheduler.load_state_dict(cosine_state_dict)

        start_step = state_dict['step']
        
        max_memory = torch.cuda.max_memory_allocated()
        logger.info('The peak GPU memory when resuming is '
                    f'{max_memory/1024**3:.1f}GB.')
        
        
        resume_cost_time = time.time() - start_resume_t
        logger.info(f'Resume cost {resume_cost_time}s')
    # --------------------------  Resume  End  ------------------------------ #
    
    ############################## 5. Training ###############################
    
    start_train_t = time.time()
    
    for step in range(start_step, total_steps):

        epoch = step // per_epoch_iters
        if step + 1 % per_epoch_steps == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.
            inner_step = step % per_epoch_steps
            train_dataloader.sampler.set_epoch(epoch, inner_step)
            data_iterator = iter(train_dataloader)

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_lr()[0]

        torch.cuda.reset_peak_memory_stats()

        step_losses = []
        step_data_time = 0
        step_start_t = time.time()
        step_consumed_tokens = 0
        for i in range(per_step_iters):
            if step * per_step_iters + i + 1 == per_epoch_iters:
                break

            _data_start_t = time.time()
            data = next(data_iterator)
            step_data_time += time.time() - _data_start_t

            input_ids = data['input_ids'].cuda()

            labels = data['labels'].cuda()
            position_ids = data['position_ids'].cuda()

            if data['chunk_sizes']:
                unpack_sizes = data['chunk_sizes'].cuda()
            else:
                unpack_sizes = None

            loss_scale = 1 / per_step_iters
            loss = packed_sequence_fwd_and_bwd(shard_model, input_ids,
                                               position_ids, labels,
                                               unpack_sizes, loss_scale)
            step_losses.append(loss)
            step_consumed_tokens += data['attention_mask'].sum()

        grad_norm = shard_model.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time / args.tp_size)
        max_memory = torch.cuda.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            step_loss = sum(step_losses) / len(step_losses)
            logger.info(f'(Epoch {epoch}) Step {step+1}/{total_steps}  '
                        f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                        f'grad_norm: {grad_norm:.2f}  '
                        f'max_memory: {(max_memory / 1024**3):.1f}GB  '
                        f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                        f'time: {step_time:.2f}s  '
                        f'eta: {eta}')

        if is_interval(step, total_steps, checkpoint_interval):
            # FSDP cannot be saved via torch.load
            # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
            model_state_dict, optimizer_state_dict = get_state_dict(
                shard_model, optimizer)

            state_dict = {
                'model': model_state_dict,
                'optimizer': optimizer_state_dict,
                'step': step,
                'total_steps': total_steps,
                'warmup_scheduler': warmup_scheduler.state_dict(),
                'cosine_scheduler': cosine_scheduler.state_dict()
            }

            num_digits = len(str(abs(total_steps)))
            work_dir = args.work_dir
            ckpt_dir = os.path.join(work_dir, f'ckpt-{step:0{num_digits}}')
            writer = dcp.FileSystemWriter(ckpt_dir)
            mkdir_or_exist(ckpt_dir)
            dcp.save(state_dict, writer)
        
    train_cost_time = time.time() - start_train_t
    logger.info(f'Training cost {train_cost_time}s')
    # ------------------------    Training  End  ---------------------------- #
if __name__ == '__main__':

    args = parse_args()
    # breakpoint()
    sft(args)
