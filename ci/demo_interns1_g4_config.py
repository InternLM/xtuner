from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.module.rope.rope import RopeScalingConfig
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
import json
import os
import shutil
from xtuner.v1.model.compose.qwen3_vl.qwen3_vl_config import Qwen3VLMoE30BA3TimeSeriesConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config

# 路径配置
meta_data_path = '/mnt/shared-storage-user/brainllm-share/wuwen/times_v2/scp_xtuner/train_4task.json'
model_path = 'xxxxxxxx'
work_dir = "./work_dir/qwen3vl_sft_ts"
tokenizer_cache_dir = "./ts_tokeniz_cache"


# 将当前配置文件拷贝到work_dir
if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)
current_file = __file__
shutil.copy(current_file, work_dir)

# 训练超参数
sample_max_length = 8192
pack_max_length = 8192
processor_path = model_path
num_workers = 8
global_batch_size = 8
total_epoch = 1
hf_interval = 3000
hf_max_keep = 5
checkpoint_interval = 3000
checkpoint_maxkeep = 5
lr = 2e-5
lr_min = 2e-6
weight_decay = 0.05
warmup_ratio = 0.03
recompute_ratio = 1.0
loss_reduction = "square"
enable_3d_rope = False

# model config
model_cfg = Qwen3VLMoE30BA3TimeSeriesConfig(
                freeze_vision=True,
                freeze_projector=True,
                freeze_language=True,
                text_config=Qwen3MoE30BA3Config(max_position_embeddings=32768, n_routed_experts=512))
model_cfg.time_series_encoder_path = model_path
model_cfg.compile_cfg = False

...


ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          enable_sequential_sampler=True,
                                          class_name='VLMJsonlDataset',
                                          cache_tag='cache_tags_v1',
                                          cache_dir=tokenizer_cache_dir),
                 "tokenize_fn": Qwen3VLTokenizeFnConfig(
                                        max_length=sample_max_length,
                                        processor_path=processor_path,
                                        system_message=_data.get('system_message', None),
                                        hash=_data.get('hash', None),
                                        enable_3d_rope=enable_3d_rope
                                    )
                 }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(recompute_ratio=recompute_ratio,
                      ep_size=1,
                      torch_compile=False,
                      checkpoint_preserve_rng_state=False)

resume_cfg = ResumeConfig(auto_resume=False)

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
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    hf_max_keep=hf_max_keep,
    work_dir=work_dir,
)
