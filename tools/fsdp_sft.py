# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta

import torch
import torch.distributed.checkpoint as dcp
from mmengine import mkdir_or_exist
from mmengine.dist import init_dist
from torch.distributed.checkpoint.state_dict import (get_state_dict,
                                                     set_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

# from xtuner.model import  TextFinetune
from xtuner._lite import AutoModelForCausalLM, AutoTokenizer, get_logger
from xtuner._lite.accelerate import packed_sequence_fwd_and_bwd
from xtuner._lite.chat import ChatTemplate
from xtuner._lite.datasets import FinetuneDataset
from xtuner._lite.parallel import ParallelSampler

# from transformers import AutoModelForCausalLM

logger = get_logger()


def parallel_formatter(dp_rank, tp_rank, debug=False):

    formatter = f'[DP_RANK {dp_rank}][TP_RANK {tp_rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += '<level>{message}</level>'
    return formatter


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Group 1 description')
    model_args.add_argument('-m', '--model', help='config file name or path.')
    model_args.add_argument('-t', '--tokenizer', default=None)

    data_args = parser.add_argument_group('data', 'Group 1 description')
    data_args.add_argument('--dataset', help='')
    data_args.add_argument(
        '--dataset-format',
        default='openai',
        help='the dir to save logs and models')
    data_args.add_argument(
        '--dataset-cache', help='the dir to save logs and models')
    data_args.add_argument('--max-length', type=int, default=2048, help='')
    data_args.add_argument('--mirco-batch-size', type=int, default=1, help='')
    data_args.add_argument('--num-workers', type=int, default=8, help='')

    dist_args = parser.add_argument_group('dist', 'Group 1 description')
    dist_args.add_argument('--tp-size', type=int, default=1, help='')
    dist_args.add_argument('--sp-size', type=int, default=1, help='')

    optim_args = parser.add_argument_group('optimizer', 'Group 1 description')
    optim_args.add_argument(
        '--global-batch-size', type=int, default=16, help='')
    optim_args.add_argument(
        '--lr',
        '--learning-rate',
        default=4e-5,
        type=float,
        help='the dir to save logs and models')
    optim_args.add_argument('--wd', '--weight-decay', default=0, type=float)
    optim_args.add_argument('--max-grad-norm', default=1, type=float)
    optim_args.add_argument('-e', '--epochs', default=1, type=int)
    optim_args.add_argument('--warmup-ratio', default=0.03, type=float)

    # engine_args = parser.add_argument_group('engine', 'Group 1 description')
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and models')
    parser.add_argument('--checkpoint-interval', default=10, type=int)
    parser.add_argument('--save-optimizer', default=10, type=int)
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=0, help='Random seed for the training')
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


def sft(args):

    init_dist('slurm')

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
    logger.add(log_file, format=formatter, level='INFO')

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32)
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.model,
        trust_remote_code=True,
        padding_side='right')

    shard_model = FSDP(
        model,
        device_mesh=dp_mesh,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16),
        use_orig_params=True)
    optimizer = AdamW(shard_model.parameters(), lr=args.lr, foreach=True)
    # For TP, input needs to be same across all TP ranks.
    # while for SP, input can be different across all ranks.
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader.
    dp_rank = dp_mesh.get_local_rank()

    chat_template = ChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>\n',
        stop_words=['<|im_end|>'])

    dataset = FinetuneDataset(
        tokenizer,
        chat_template,
        max_length=args.max_length,
        data_files=args.dataset,
        pack_to_max_length=True)

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        sampler=ParallelSampler(dataset, dp_mesh, shuffle=True),
        collate_fn=FinetuneDataset.dataloader_collate_fn,
        persistent_workers=True)

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    per_step_iters = global_batch_size // mirco_batch_size // dp_size
    per_epoch_iters = len(train_dataloader)
    per_epoch_steps = math.ceil(per_epoch_iters / per_step_iters)

    total_epochs = args.epochs
    total_steps = per_epoch_steps * total_epochs

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x <= warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=0)

    start_step = 0
    if args.resume:

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

        max_memory = torch.cuda.max_memory_allocated()

        step_losses = []
        data_time = 0
        _step_start_t = time.time()
        for i in range(per_step_iters):
            if step * per_step_iters + i + 1 == per_epoch_iters:
                break

            _data_start_t = time.time()
            data = next(data_iterator)
            data_time += time.time() - _data_start_t

            input_ids = data['input_ids'].cuda()
            labels = data['labels'].cuda()
            position_ids = data['position_ids'].cuda()
            unpack_sizes = data['chunk_sizes'].cuda()

            loss = packed_sequence_fwd_and_bwd(shard_model, input_ids,
                                               position_ids, labels,
                                               unpack_sizes)
            step_losses.append(loss)
        grad_norm = shard_model.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()

        step_time = time.time() - _step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        if is_interval(step, total_steps, args.log_interval):
            step_loss = sum(step_losses) / len(step_losses)
            logger.info(f'(Epoch {epoch}) Step {step+1}/{total_steps}  '
                        f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                        f'grad_norm: {grad_norm:.2f}  '
                        f'max_memory: {(max_memory / 1024**3):.1f}GB  '
                        f'data_time: {data_time:.2f}s  '
                        f'time: {step_time:.2f}s  '
                        f'eta: {eta}')

        if is_interval(step, total_steps, args.checkpoint_interval):
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

        torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    args = parse_args()
    sft(args)
