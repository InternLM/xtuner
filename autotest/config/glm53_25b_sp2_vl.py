import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import Glm53VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.compose.glm53 import Glm53BaseConfig
from xtuner.v1.train import TrainerConfig


MEDIA_ROOT = os.environ["MEDIA_ROOT"]
MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]

ep_size = 2

model_cfg = Glm53BaseConfig.from_hf(MODEL_PATH)
model_cfg.text_config.dispatcher = "all2all"
model_cfg.text_config.ep_size = ep_size
model_cfg.text_config.compile_cfg = False
model_cfg.compile_cfg = False
model_cfg.text_config.attention.sparse_mla_backend = "flash_mla_cudnn"
model_cfg.vision_config.attn_impl = "flash_attention"

optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
    ep_size=ep_size,
    tp_size=1,
)

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="glm53_vl",
            anno_path=DATA_PATH,
            class_name="VLMJsonlDataset",
            media_root=MEDIA_ROOT,
            sample_ratio=1.0,
        ),
        "tokenize_fn": Glm53VLTokenizeFnConfig(
            processor_path=MODEL_PATH,
            max_length=16384,
            max_pixels=1048576,
        ),
    },
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=16384,
    collator="glm53_vl_sft_collator",
    collator_kwargs={
        "image_token_id": model_cfg.image_token_id,
        "merge_unit": model_cfg.vision_config.spatial_merge_size**2,
    },
)

loss_cfg = CELossConfig(mode="chunk", chunk_size=1024)
model_cfg.text_config.lm_loss_cfg = loss_cfg

trainer = TrainerConfig(
    load_from=MODEL_PATH,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=MODEL_PATH,
    strict_load=False,
    global_batch_size=8,
    intra_layer_micro_batch=2,
    sp_size=2,
    total_step=20,
    work_dir=f"{os.environ['WORK_DIR']}",
    seed=0,
)
