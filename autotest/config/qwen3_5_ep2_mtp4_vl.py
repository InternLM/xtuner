import os

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.train import ResumeConfig, TrainerConfig


MEDIA_ROOT = os.environ["MEDIA_ROOT"]
MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]
WORK_DIR = os.environ["WORK_DIR"]

sample_max_length = 128 * 1024
pack_max_length = 128 * 1024
ep_size = 2

moe_cfg = Qwen3_5_VLMoE35BA3Config(only_llm_forward=False, compile_cfg=True)
moe_cfg.text_config.ep_size = ep_size
moe_cfg.text_config.dispatcher = "deepep"
moe_cfg.text_config.mtp_config = MTPConfig(
    num_layers=4,
    share_weights=True,
    loss_scaling_factor=1.0,
)

optim_cfg = MuonConfig(lr=2e-5, weight_decay=0.05)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0.1, lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    recompute_ratio=1.0,
    torch_compile=True,
    cpu_offload=False,
    ep_size=ep_size,
    checkpoint_preserve_rng_state=False,
)

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="multimodal",
            anno_path=DATA_PATH,
            class_name="VLMJsonlDataset",
            media_root=MEDIA_ROOT,
            sample_ratio=1.0,
            cache_dir=os.path.join(WORK_DIR, "jsonl_cache"),
            cache_tag=f"qwen3p5_vl_{sample_max_length}",
        ),
        "tokenize_fn": Qwen3VLTokenizeFnConfig(
            processor_path=MODEL_PATH,
            chat_template="qwen3.5-vl",
            llm_pack_weight=-3.2,
            visual_pack_weight=5.0,
            max_length=sample_max_length,
            rand_video_max_frames=24,
            max_pixels=16384 * 32 * 32,
        ),
    },
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_level="soft",
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    pack_chunk_size=10000,
    pack_workers=4,
    global_pack=True,
    group_by_length=True,
    collator="qwen3_vl_sft_collator",
    pack_extra_buffer_size=20,
    num_workers=4,
)

loss_cfg = CELossConfig(mode="chunk", chunk_size=1024, loss_reduction="square")

trainer = TrainerConfig(
    load_from=MODEL_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=MODEL_PATH,
    resume_cfg=ResumeConfig(auto_resume=True),
    global_batch_size=16,
    total_step=20,
    sp_size=4,
    work_dir=WORK_DIR,
    seed=0,
)

