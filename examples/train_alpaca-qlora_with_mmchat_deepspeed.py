# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import deepspeed
import torch
import transformers
from data_utils import get_train_dataloader
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from xtuner.models import SupervisedFinetune


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path', type=str, default='internlm/internlm-7b')
    parser.add_argument('--use_qlora', type=bool, default=True)
    parser.add_argument('--use_lora', type=bool, default=False)
    parser.add_argument(
        '--dataset_cfg_path',
        type=str,
        default='../configs/_base_/datasets/alpaca.py',
        help='Path to xtuner dataset config')
    parser.add_argument(
        '--deepspeed_config',
        type=str,
        default='./deepspeed_config.json',
        help='Path to deepspeed config')
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=3,
        help='Number of epochs to train')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='work_dirs',
        help='Dir to store checkpoint files')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Reserved for deepspeed framework')
    return parser


arg_parser = get_argument_parser()
args = arg_parser.parse_args()
if args.local_rank == -1:
    device = torch.device('cuda')
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    deepspeed.init_distributed()


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except Exception:
            output[k] = v
    return output


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def train():
    # build model
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM')
    elif not args.use_qlora and args.use_lora:
        quantization_config = None
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM')
    else:
        quantization_config = None
        lora_config = None

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)

    model = SupervisedFinetune(llm=llm, lora=lora_config, tokenizer=tokenizer)

    # build deepspeed engine
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=None,
        model_parameters=model_parameters)

    # build dataloader
    train_dataloader = get_train_dataloader(args.dataset_cfg_path, tokenizer)

    # training
    model_engine.train()
    for epoch in range(args.num_train_epochs):
        for i, inputs in enumerate(tqdm(train_dataloader)):
            inputs = to_device(inputs, device)
            loss = model_engine(**inputs)['loss']
            model_engine.backward(loss)
            model_engine.step()
            print_rank_0(
                f'Epoch [{epoch+1}/{args.num_train_epochs}], '
                f'Step [{i}/{len(train_dataloader)}], Loss: {loss.item()}')

        save_dir = os.path.join(args.output_dir, f'epoch_{epoch+1}')
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = model_engine.module.llm if hasattr(
            model_engine, 'module') else model_engine.llm
        model_to_save.save_pretrained(
            save_dir, state_dict=model_to_save.state_dict())


if __name__ == '__main__':
    train()
