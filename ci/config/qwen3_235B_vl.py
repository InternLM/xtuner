import shutil
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLMoE235BA22Config
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config
from xtuner.v1.train import ResumeConfig, TrainerConfig


model_path = "/mnt/llm-razor/model/Qwen3-235B-A22B-VL"
tokenizer_cache_dir = "/mnt/llm-razor/yehaochen/.cache"
work_dir = "/mnt/llm-razor/yehaochen/241/vl"

seq_len = 8192
pack_len = 8192
intra_layer_micro_batch = 1
ep_size = 1
sp_size = 1

num_workers = 8
# num_hidden_layers = 2
lr = 1e-4
lr_min = 1e-6
weight_decay = 1e-2
warmup_ratio = 0.1
recompute_ratio = 1.0
loss_chunk_size = 1024

Path(work_dir).mkdir(parents=True, exist_ok=True)
shutil.copy2(__file__, work_dir)

DP_SIZE = 512

print(f"DP_SIZE: {DP_SIZE}, intra_layer_micro_batch: {intra_layer_micro_batch}, sp_size: {sp_size}")

model_cfg = Qwen3VLMoE235BA22Config(
    text_config=Qwen3MoE235BA22Config(
        # num_hidden_layers=num_hidden_layers,
        hf_key_mapping={r"^model\.": "model.language_model."},
    )
)
model_cfg.text_config.attention.sliding_window = None
model_cfg.text_config.ep_size = ep_size
if ep_size != 1:
    model_cfg.text_config.dispatcher = "all2all"

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="test",
            anno_path=str(Path(__file__).parent.parent.parent / "tests/resource/mllm_pretrain_image_example_data.jsonl"),
            media_root=str(Path(__file__).parent.parent.parent / "tests"),
            sample_ratio=1000.,
            class_name="VLMJsonlDataset",
            enable_sequential_sampler=True,
            cache_tag="cache_tags_v1",
            cache_dir=tokenizer_cache_dir,
        ),
        "tokenize_fn": Qwen3VLTokenizeFnConfig(
            processor_path=model_path,
            max_length=seq_len,
            enable_3d_rope=False,
            video_min_frames=8,
            video_max_frames=36,
        ),
    }
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_len,
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
)

optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=True)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(
    recompute_ratio=recompute_ratio,
    torch_compile=False,
    checkpoint_preserve_rng_state=False,
    ep_size=ep_size,
)
resume_cfg = ResumeConfig(auto_resume=False)
loss_cfg = CELossConfig(
    mode="chunk",
    chunk_size=loss_chunk_size,
    loss_reduction="token",
)

trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker="tensorboard",
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    global_batch_size=DP_SIZE,
    total_step=4,
    work_dir=work_dir,
    sp_size=sp_size,
    intra_layer_micro_batch=intra_layer_micro_batch,
    strict_load=False,
)

