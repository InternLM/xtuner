# Smoke SFT config for Qwen3.5-VL Dense 4B's text tower — drives the full
# from_hf -> model.forward -> loss -> backward -> optimizer step -> FSDP shard/reduce
# chain on a real checkpoint so the port can be verified end-to-end. Loss should drop
# monotonically into a plausible SFT range within the first ~50 steps.
#
# Usage (single node, 8 GPUs):
#
#   export QWEN3_5_DENSE_4B_PATH=/path/to/Qwen3.5-4B
#   export ALPACA_PATH=/path/to/alpaca
#   torchrun --nproc-per-node=8 -m xtuner.v1.train.cli.sft --config ci/config/qwen3_5_dense4B.py
#
# This smokes the text-only path on alpaca (matches the convention of
# ``ci/config/qwen3_moe_30BA3.py``); the full VLM compose forward — image embed
# injection through ``_prepare_llm_inputs`` — is owned by
# ``tests/model/test_qwen3_5_dense.py::test_model_forward_bitwise_reduced_layers``.
import os

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense4BConfig
from xtuner.v1.train import TrainerConfig


QWEN3_5_DENSE_4B_PATH = os.environ["QWEN3_5_DENSE_4B_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]


model_cfg = Qwen3_5_VLTextDense4BConfig()
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=16384),
    },
]

dataloader_config = DataloaderConfig(
    pack_max_length=16384,
)

loss_cfg = CELossConfig()


trainer = TrainerConfig(
    load_from=QWEN3_5_DENSE_4B_PATH,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=QWEN3_5_DENSE_4B_PATH,
    global_batch_size=16,
    total_epoch=1,
    work_dir="/tmp/qwen3_5_dense4B",
    seed=0,
    # XTuner defers Qwen3.5 MTP weights, so the ckpt has ~15 ``mtp.*`` keys with no
    # matching XTuner params; loosen strict load to skip them. Same reasoning as the
    # save_hf round-trip test scoping out ``mtp.*``.
    strict_load=False,
)
