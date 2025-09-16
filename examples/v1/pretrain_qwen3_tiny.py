from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig
from xtuner.v1.datasets.pt_tokenize_fn import PretrainTokenizeFunctionConfig
from xtuner.v1.model import Qwen3Dense4BConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig

# model config
model_cfg = Qwen3Dense4BConfig()

# dataset and dataloader config
sample_max_length = 4096
pack_max_length = 4096

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="source1",
            # jsonl file with {"content": "<text>"}
            # or directory containing such jsonl files
            anno_path="tests/resource/pretrain_example_data.jsonl",
            sample_ratio=10000.0,
        ),
        "tokenize_fn": PretrainTokenizeFunctionConfig(
            add_bos_token=False,
            add_eos_token=True,
        ),
    },
    {
        "dataset": DatasetConfig(
            name="source2",
            # jsonl file with {"content": "<text>"}
            # or directory containing such jsonl files
            anno_path="tests/resource/pretrain_example_data.jsonl",
            sample_ratio=20000.0,
        ),
        "tokenize_fn": PretrainTokenizeFunctionConfig(
            add_bos_token=False,
            add_eos_token=True,
        ),
    },
]
dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    num_workers=4,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=1e-4, foreach=False)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0)

# trainer config
trainer = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    # dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    tokenizer_path="Qwen/Qwen3-4B",
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    global_batch_size=8,
    total_epoch=1,
    work_dir="work_dir/pretrain_qwen3_tiny",
    # load_from="Qwen/Qwen3-4B",
)
