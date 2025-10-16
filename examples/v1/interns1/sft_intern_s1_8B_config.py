from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import InternS1VLTokenizeFnConfig
from xtuner.v1.model import InternS1MiniConfig
from xtuner.v1.model.compose.intern_s1 import InternS1VisionConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
import json
import os

# 路径配置
ceph_config = "/mnt/shared-storage-user/gaozhangwei/workspace_glx/petreloss.conf"
meta_data_path = '/mnt/shared-storage-user/gaozhangwei/workspace_ysl/reformat_new_data/export_meta_s1.json'
model_path = "/mnt/shared-storage-user/intern7shared/internvl_a4s/xpuyu/work_dir/InternS1/InternS1-8B-Runtu-cpt-science-data-slow-tokenize-data-0630-slow-tokenize-lr1e5-bs512-maxstep30000-rjob-h200-hf-16000" #转换后的权重（hf官方格式）
work_dir = "/mnt/shared-storage-user/intern7shared/internvl_a4s/xtuner_saved_model/interns1/interns1-mini-sft-bs512-maxsteps8000-lr8e-5"
tokenizer_cache_dir = "/mnt/shared-storage-user/intern7shared/internvl_a4s/xtuner_tokenizer_cache/interns1/slow_tokenize_sft_ml_32k_tokenizer"

# 训练超参数
sample_max_length = 32768
pack_max_length = 32768
min_num_frames=8
max_num_frame=36
global_batch_size = 512
total_step = 8000
hf_interval = 1000
checkpoint_interval = 1000
checkpoint_maxkeep = 10
lr = 8e-5
lr_min = 1e-6
weight_decay = 0.05
warmup_ratio = 0.1
recompute_ratio = 1.0
drop_path_rate = 0.0
loss_reduction = "square"

# model config
model_cfg = InternS1MiniConfig(vision_config=InternS1VisionConfig(drop_path_rate=drop_path_rate))

# dataset config
oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name='VLMJsonlDataset',
                                          cache_dir=tokenizer_cache_dir),
                 "tokenize_fn": InternS1VLTokenizeFnConfig(model_cfg=model_cfg,
                                                           max_length=sample_max_length,
                                                           max_dynamic_patch=_data.get('max_dynamic_patch',
                                                                                       None),
                                                           min_dynamic_patch=_data.get('min_dynamic_patch',
                                                                                       None),
                                                           min_num_frames=_data.get('min_num_frames', min_num_frames),
                                                           max_num_frames=_data.get('max_num_frames', max_num_frame),
                                                           data_augment=_data.get('data_augment', False),
                                                           system_message=_data.get('system_message', None),
                                                           hash=_data.get('hash', None),
                                                           oss_loader_cfg=oss_loader_cfg,
                                                           template_name="intern-s1"
                                                           )
                 }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    collator="intern_s1_vl_sft_collator",
    num_workers=8,
    pack_extra_buffer_size=20,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(sp_size=1, recompute_ratio=recompute_ratio, torch_compile=True)

resume_cfg = ResumeConfig(auto_resume=True)

# trainer config
trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker='tensorboard',
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_step=total_step,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    work_dir=work_dir,
)
