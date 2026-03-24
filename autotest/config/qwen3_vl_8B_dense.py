import os

from xtuner.v1.config import AdamWConfig, LRConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3VLDense8BConfig
from xtuner.v1.train import TrainerConfig

MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]
MEDIA_ROOT = os.environ["MEDIA_ROOT"]

# model config
model_cfg = Qwen3VLDense8BConfig()


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
            max_length=8192,
            add_vision_id=True,
        ),
    },
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=8192,
    collator="qwen3_vl_sft_collator",
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=1e-5, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0.03, lr_min=1e-6)

# trainer config
trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=MODEL_PATH,
    tokenizer_path=MODEL_PATH,
    dataloader_cfg=dataloader_config,
    optim_cfg=optim_cfg,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    global_batch_size=16,
    total_epoch=1,
    work_dir=f"{os.environ['WORK_DIR']}",
    seed=0,
)
