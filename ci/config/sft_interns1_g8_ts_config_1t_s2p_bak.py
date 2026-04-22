from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5TimeSeriesMoE35BA3Config

# 路径配置

current_file = __file__

# 训练超参数
sample_max_length = 8192
pack_max_length = 8192
processor_path = "/mnt/shared-storage-gpfs2/speechllm-share/wuwen/interns2_pv/InternS2Preview_with_TS"
model_path = "/mnt/shared-storage-gpfs2/speechllm-share/wuwen/interns2_pv/InternS2Preview_with_TS"
num_workers = 8
global_batch_size = 8
total_epoch = 1
hf_interval = 500
hf_max_keep = 5
checkpoint_interval = 500
checkpoint_maxkeep = 5
lr = 2e-5
lr_min = 2e-6
weight_decay = 0.05
warmup_ratio = 0.03
recompute_ratio = 1.0
loss_reduction = "square"
enable_3d_rope = False

# model config
model_cfg = Qwen3_5TimeSeriesMoE35BA3Config(
    freeze_vision=True,
    freeze_projector=True,
    freeze_language=True,
    only_llm_forward=True
)
model_cfg.time_series_encoder_path = processor_path
model_cfg.compile_cfg = False

import json
meta_data_path = '/mnt/shared-storage-user/brainllm-share/wuwen/times_v2/scp_xtuner/train_all_zh_en.json'
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {
        "dataset":
        DatasetConfig(name=name,
                      anno_path=_data['annotation'],
                      media_root=_data.get('media_root', ''),
                      sample_ratio=_data.get('sample_ratio', 1.0),
                      enable_sequential_sampler=True,
                      class_name='VLMJsonlDataset',
                      cache_tag='cache_tags_v1',
                      cache_dir=tokenizer_cache_dir),
        "tokenize_fn":
        Qwen3VLTokenizeFnConfig(max_length=sample_max_length,
                                processor_path=processor_path,
                                system_message=_data.get(
                                    'system_message', None),
                                hash=_data.get('hash', None),
                                enable_3d_rope=enable_3d_rope)
    }
    dataset_config.append(_data_cfg)
# dataset_config = [
#     {
#         "dataset": DatasetConfig(
#             name="test",
#             anno_path="/mnt/shared-storage-user/llmrazor-share/data/train_1000_samples.jsonl",
#             media_root="/mnt/shared-storage-user/llmrazor-share/data",
#             sample_ratio=1.0,
#             enable_sequential_sampler=True,
#             class_name='VLMJsonlDataset',
#             cache_tag='cache_tags_v1',
#             cache_dir="/tmp"
#         ),
#         "tokenize_fn": Qwen3VLTokenizeFnConfig(
#                 max_length=sample_max_length,
#                 processor_path=processor_path,
#                 system_message=None,
#                 hash=None,
#                 enable_3d_rope=enable_3d_rope
#             )
#     }
# ]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=0,
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
    work_dir="/tmp",
    strict_load=False,
)

