from xtuner.v1.model.interns1 import InternS1MiniConfig
from xtuner.v1.config import TrainerConfig
from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    LRConfig,
)
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets import InternS1TokenizeFnConfig

# model config
model_cfg = InternS1MiniConfig()

# dataset and dataloader config
sample_max_length = 8192
pack_max_length = 16384

dataset_config = [
    {
        "dataset": DatasetConfig(name='pure_text',
                                 anno_path='tests/resource/mllm_sft_text_example_data.jsonl',
                                 sample_ratio=5.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": InternS1TokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
    {
        "dataset": DatasetConfig(name='media',
                                 anno_path='tests/resource/mllm_sft_media_example_data.jsonl',
                                 media_root='tests/',
                                 sample_ratio=20.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": InternS1TokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
]
dataloader_config = DataloaderConfig(pack_max_length=pack_max_length,
                                     num_workers=8,
                                     pack_level="expand_soft",
                                     collator='sft_vllm_collator')

# optimizer and lr config
optim_cfg = AdamWConfig(lr=1e-6, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0)

hf_model_path = '/cpfs01/shared/llm_ddd/huanghaian/interns1/interns1-mini-remote'

# trainer config
trainer = TrainerConfig(
    load_from=hf_model_path,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    tokenizer_path=hf_model_path,
    global_batch_size=8,
    epoch_num=2,
    work_dir='work_dirs'
)
