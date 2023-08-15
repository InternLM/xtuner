import torch
from bitsandbytes.optim import PagedAdamW32bit
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.model import BaseDataPreprocessor
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.datasets import process_hf_dataset
from xtuner.datasets.collate_fns import default_collate_fn
from xtuner.datasets.map_fns import oasst1_map_fn
from xtuner.engine import LogSampleHook, SampleGenerateHook
from xtuner.models import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE

pretrained_model_name_or_path = 'internlm/internlm-7b'

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    padding_side='right',
    trust_remote_code=True)

model = dict(
    type=SupervisedFinetune,
    data_preprocessor=dict(type=BaseDataPreprocessor),
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'),
    tokenizer=tokenizer)

oasst1 = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='timdettmers/openassistant-guanaco'),
    tokenizer=tokenizer,
    max_length=2048,
    map_fn=oasst1_map_fn,
    concat_to_max_length=True)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=oasst1,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0.01
max_norm = 1
accumulative_counts = 16

# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=PagedAdamW32bit, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

max_epochs = 3
# learning policy
param_scheduler = dict(
    type=CosineAnnealingLR,
    eta_min=lr * 0.1,
    by_epoch=True,
    T_max=max_epochs,
    convert_to_iter_based=True)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

custom_hooks = [
    dict(type=LogSampleHook, tokenizer=tokenizer),  # noqa: F405
    dict(
        type=SampleGenerateHook,
        tokenizer=tokenizer,  # noqa: F405
        every_n_iters=500,
        stop_word='###',
        sample_inputs=[
            '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
        ],
        instruction=PROMPT_TEMPLATE.openassistant.INSTRUCTION_START)
]

# defaults to use registries in mmpretrain
default_scope = 'xtuner'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),

    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=10),

    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),

    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
