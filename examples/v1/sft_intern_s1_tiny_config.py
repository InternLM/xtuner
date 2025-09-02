from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    LRConfig,
    TrainerConfig,
)
from xtuner.v1.datasets import InternS1TokenizeFnConfig
from xtuner.v1.model import Qwen3Dense8BConfig, InternS1MiniConfig
from xtuner.v1.loss import CELossConfig

# model config
text_cfg = Qwen3Dense8BConfig(num_hidden_layers=8,
                              hidden_size=1024,
                              intermediate_size=4096,
                              vocab_size=300)
model_cfg = InternS1MiniConfig(text_config=text_cfg,
                               # xtuner/v1/train/toy_tokenizer.py
                               image_token_id=260)  # fake tokenizer vocab size for tiny model

# dataset and dataloader config
sample_max_length = 4096
pack_max_length = 4096

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="pure_text",
            anno_path="tests/resource/mllm_sft_text_example_data.jsonl",
            sample_ratio=1.0,
            class_name="VLMJsonlDataset",
        ),
        "tokenize_fn": InternS1TokenizeFnConfig(
            model_cfg=model_cfg, max_length=sample_max_length
        ),
    },
    {
        "dataset": DatasetConfig(
            name="media",
            anno_path="tests/resource/mllm_sft_media_example_data.jsonl",
            media_root="tests/",
            sample_ratio=2.0,
            class_name="VLMJsonlDataset",
        ),
        "tokenize_fn": InternS1TokenizeFnConfig(
            model_cfg=model_cfg, max_length=sample_max_length
        ),
    },
]
dataloader_config = DataloaderConfig(
    pack_max_length=pack_max_length, collator="sft_vllm_collator"
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=1e-4, foreach=False)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0)

# trainer config
trainer = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    global_batch_size=1,
    epoch_num=1,
    work_dir="work_dirs",
)
