from datasets import load_dataset
from mmchat.datasets import process_hf_dataset, DataCollatorForCausalLM
from mmengine.dataset import DefaultSampler
from transformers import AutoModel, AutoTokenizer
from mmchat.models import SupervisedFinetune
from mmchat.models.utils import DataProcesorForCausalLM
from mmchat.visualization import AttentionScoreVisualizer



"""
------------ Dataset Example (after `load_dataset`) ------------

DatasetDict({
    train: Dataset({
        features: ['instruction', 'input', 'output', 'text'],
        num_rows: 52002
    })
})

------------ Dataset Example (after `process_hf_dataset`) ------------

DatasetDict({
    train: Dataset({
        features: ['text', 'input', 'output'],
        num_rows: 9846
    })
    test: Dataset({
        features: ['text', 'input', 'output'],
        num_rows: 518
    })
})

"""

alpaca = dict(
    type = process_hf_dataset,
    dataset = dict(
        type = load_dataset,
        path = 'tatsu-lab/alpaca',
    ),
    # map_fn = extract_alpaca_dataset,
    prompt_input_format = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    prompt_no_input_format= (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
    remove_columns=['instruction'],
)



oasst1 = dict(
    type = process_hf_dataset,
    dataset = dict(
        type = load_dataset,
        path = 'timdettmers/openassistant-guanaco',
    ),
    map_fn = lambda x: {'input': '', 'output': x['text']},
)


train_dataloader = dict(
    batch_size = 32, 
    num_workers = 8,
    dataset = oasst1,
    sampler = dict(type=DefaultSampler, shuffle=True),
    persistent_workers = True,
)

model = dict(
    type = SupervisedFinetune,
    data_preprocessor = dict(
        type=DataProcesorForCausalLM,
        tokenizer=dict(
            type=AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path = '/share/gaojianfei/merged_chinese_lora_7b',
            use_fast = False,
        ),
        source_max_len = 512,
        target_max_len = 512,
        train_on_source = False,
        predict_with_generate = False,
    ),
    llm = dict(
        type=AutoModel.from_pretrained,
        pretrained_model_name_or_path = '/share/gaojianfei/merged_chinese_lora_7b',
    ),

)


# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[100, 150], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
# val_cfg = dict()
# test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)


# defaults to use registries in mmpretrain
default_scope = 'mmchat'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    # visualization=dict(type='VisualizationHook', enable=False),
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

# # set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)