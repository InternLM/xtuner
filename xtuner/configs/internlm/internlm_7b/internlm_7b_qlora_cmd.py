import torch
from bitsandbytes.optim import PagedAdamW32bit
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.datasets import process_hf_dataset
from xtuner.datasets.collate_fns import default_collate_fn
from xtuner.datasets.map_fns import cmd_map_fn
from xtuner.engine import LogSampleHook, SampleGenerateHook
from xtuner.models import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          STEP 1  Settings                           #
#######################################################################
# path
pretrained_model_name_or_path = 'internlm/internlm-7b'
data_url = 'https://github.com/Toyhom/Chinese-medical-dialogue-data/raw/master/Data_数据/'  # noqa: E501
all_csv = [
    'Andriatria_男科/男科5-13000.csv', 'IM_内科/内科5000-33000.csv',
    'OAGD_妇产科/妇产科6-28000.csv', 'Oncology_肿瘤科/肿瘤科5-10000.csv',
    'Pediatric_儿科/儿科5-14000.csv', 'Surgical_外科/外科5-14000.csv'
]
all_csv = [data_url + csv for csv in all_csv]

# data
batch_size = 1
accumulative_counts = 16
dataloader_num_workers = 0
max_epochs = 1
# optim
optim_type = PagedAdamW32bit
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0.01
max_norm = 1  # grad clip

#######################################################################
#                      STEP 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
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
        task_type='CAUSAL_LM'))

#######################################################################
#                      STEP 4  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(
        type=load_dataset,
        path='csv',
        data_files=dict(train=all_csv),
        encoding='GB18030'),
    tokenizer=tokenizer,
    max_length=2048,
    map_fn=cmd_map_fn,
    pack_to_max_length=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                            STEP 5  Scheduler                        #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = dict(
    type=CosineAnnealingLR,
    eta_min=lr * 0.1,
    by_epoch=True,
    T_max=max_epochs,
    convert_to_iter_based=True)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

#######################################################################
#                           STEP 6  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process，optional
custom_hooks = [
    dict(type=LogSampleHook, tokenizer=tokenizer),  # noqa: F405
    dict(
        type=SampleGenerateHook,
        tokenizer=tokenizer,  # noqa: F405
        every_n_iters=500,
        sample_inputs=[
            '我有家族遗传性的过敏，请问可以可以献血吗？', '我爷爷有高血压，请问他可以喝咖啡吗？',
            '我女儿今年3岁了，从昨天晚上九点开始腹泻，到现在已经八个小时了，请问应该怎么办？'
        ],
        instruction=PROMPT_TEMPLATE.medical.INSTRUCTION_START)
]

# defaults to use registries in xtuner
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
