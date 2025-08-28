from xtuner.v1.model.interns1 import InternS1Config, InternS1VisionConfig, InternS1ProjectorConfig
from xtuner.v1.model.dense.qwen3 import Qwen3_8BConfig
from xtuner.v1.config import TrainerConfig
from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    LRConfig,
)
from xtuner.v1.datasets import InternS1TokenizeFnConfig

# model config
vision_cfg = InternS1VisionConfig()
projector_cfg = InternS1ProjectorConfig()
llm_cfg = Qwen3_8BConfig(vocab_size=153216)
model_cfg = InternS1Config(vision_config=vision_cfg,
                           text_config=llm_cfg,
                           projector_config=projector_cfg,
                           freeze_vision=True,
                           freeze_projector=True
                           )

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
    tokenizer_path=hf_model_path,
    global_batch_size=8,
    epoch_num=2,
    chunked_loss=True,
    work_dir='work_dirs'
)
