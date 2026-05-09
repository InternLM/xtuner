import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.train import ResumeConfig, TrainerConfig


MEDIA_ROOT = os.environ["MEDIA_ROOT"]
MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]


moe_cfg = Qwen3_5_VLMoE35BA3Config(compile_cfg=False)

optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
    tp_size=2,
)

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="sft",
            anno_path=DATA_PATH,
            class_name="VLMJsonlDataset",
            media_root=MEDIA_ROOT,
            sample_ratio=1.0,
        ),
        "tokenize_fn": Qwen3VLTokenizeFnConfig(
            processor_path=MODEL_PATH,
            max_length=16384,
            add_vision_id=True,
        ),
    },
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=16384,
    collator="qwen3_vl_sft_collator",
)

loss_cfg = CELossConfig(mode="chunk", chunk_size=1024)

trainer = TrainerConfig(
    load_from=MODEL_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=MODEL_PATH,
    global_batch_size=16,
    total_epoch=1,
    work_dir=f"{os.environ['WORK_DIR']}",
    seed=0,
    resume_cfg=ResumeConfig(auto_resume=True),
    checkpoint_interval=10,
    checkpoint_maxkeep=2,
)
