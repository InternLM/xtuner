# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel)

from xtuner.dataset import InternVL_V1_5_LLaVADataset
from xtuner.dataset.collate_fns import mm_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import InternVL_v1_5_LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset import ConcatDataset
from xtuner.dataset.utils import internvl_1_5_encode_fn
from xtuner.dataset.samplers import LengthGroupedSampler

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = '/mnt/hwfile/xtuner/gaojianfei/Phi-3-mini-4k-instruct/models--microsoft--Phi-3-mini-4k-instruct/snapshots/3a811845d89f3c1b3f41b341d0f9f05104769f35'
visual_encoder_name_or_path = '/mnt/hwfile/xtuner/linzhihao/model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'

# Data
share_data_root = '/mnt/hwfile/xtuner/huanghaian/data/sharegpt4v/'
sharegpt4v_data_path = share_data_root + 'share-captioner_coco_lcs_sam_1246k_1107_llava.json'
sharegpt4v_image_folder = '/mnt/hwfile/xtuner/linzhihao/dataset/sharegpt4v/data'

data_root = '/mnt/hwfile/xtuner/huanghaian/data/ALLaVA-4V/'
allava_laion_data_path = data_root + 'allava_laion/ALLaVA-Caption-LAION-4V_llava.json'
allava_laion_image_folder = '/mnt/hwfile/openmmlab/zhaoxiangyu/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/624bd4c5fedc2209cf952eedf75712413d8d912c/'

data_root = '/mnt/hwfile/xtuner/huanghaian/data/ALLaVA-4V/'
allava_vflan_data_path = data_root + 'allava_vflan/ALLaVA-Caption-VFLAN-4V_llava.json'
allava_vflan_image_folder = '/mnt/hwfile/openmmlab/zhaoxiangyu/'

allava_text_data_path = data_root + 'allava_text/Evol-Instruct-GPT4-Turbo-143K_llava.json'

laion_data_root = '/mnt/hwfile/xtuner/huanghaian/data/laion-coco/'
laion_data_path = laion_data_root + 'filter_rand_10m_llava.json'
laion_image_folder = 'public:s3://public-dataset/laion-coco/images/'

coyo_data_root = '/mnt/hwfile/xtuner/huanghaian/data/COYO-700M/'
coyo_data_path = coyo_data_root + 'filter_rand_20m_llava.json'
coyo_image_folder = 'public:s3://public-dataset/COYO-700M/'

prompt_template = PROMPT_TEMPLATE.phi3_chat

# Scheduler & Optimizer
batch_size = 16  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 1e-3
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 1000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 1000
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['Please describe this picture']

min_num = 1
max_num = 6
downsample_ratio = 0.5

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

model = dict(
    type=InternVL_v1_5_LLaVAModel,
    downsample_ratio=downsample_ratio,
    tokenizer=tokenizer,
    template=prompt_template,
    image_processor=image_processor,
    freeze_llm=True,
    freeze_visual_encoder=True,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
cache_2k_root = laion_data_root + 'phi3_mini_2k_offline/'
laion_coco_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    use_patch=False,  # 由于 image token 很少，所以可能也不需要 4k 上下文
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=cache_2k_root + 'laion_coco_dataset_10m',
    data_path=laion_data_path,
    image_folder=laion_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        min_num=min_num,
        max_num=max_num,
        use_patch=False),  # 核心参数
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=2048)

cache_2k_root = coyo_data_root + 'phi3_mini_2k_offline/'
coyo_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    use_patch=False,  # 由于 image token 很少，所以可能也不需要 4k 上下文
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=cache_2k_root + 'coyo_dataset_20m',
    data_path=coyo_data_path,
    image_folder=coyo_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        min_num=min_num,
        max_num=max_num,
        use_patch=False),  # 核心参数
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=2048)

cache_2k_root = share_data_root + 'phi3_mini_2k_offline/'
sharegpt4v_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    use_patch=False,  # 由于 image token 很少，所以可能也不需要 4k 上下文
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=cache_2k_root + 'sharegpt4v_dataset',
    data_path=sharegpt4v_data_path,
    image_folder=sharegpt4v_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        min_num=min_num,
        max_num=max_num,
        use_patch=False),  # 核心参数
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=2048)

cache_2k_root = data_root + 'phi3_mini_2k_offline/'
allava_laion_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    use_patch=False,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=cache_2k_root + 'allava_laion_dataset',
    data_path=allava_laion_data_path,
    image_folder=allava_laion_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        min_num=min_num,
        max_num=max_num,
        use_patch=False),  # 核心参数
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=2048)

cache_2k_root = data_root + 'phi3_mini_2k_offline/'
allava_vflan_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    use_patch=False,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=cache_2k_root + 'allava_vflan_dataset',
    data_path=allava_vflan_data_path,
    image_folder=allava_vflan_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        min_num=min_num,
        max_num=max_num,
        use_patch=False),  # 核心参数
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=2048)

allava_text_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    use_patch=False,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=cache_2k_root + 'allava_text_dataset',
    data_path=allava_text_data_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    image_folder=None,
    dataset_map_fn=llava_map_fn,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        min_num=min_num,
        max_num=max_num,
        use_patch=False),  # 核心参数
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=2048)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        laion_coco_dataset, coyo_dataset,
        sharegpt4v_dataset, allava_laion_dataset, allava_vflan_dataset,
        allava_text_dataset, allava_text_dataset
    ])

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=mm_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
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
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    # dict(
    #     type=EvaluateChatHook,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     every_n_iters=evaluation_freq,
    #     evaluation_inputs=evaluation_inputs,
    #     evaluation_images=evaluation_images,
    #     system=SYSTEM,
    #     prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=True,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
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

# set log processor
log_processor = dict(by_epoch=False)
