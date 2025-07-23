import os
import random

import torch
from mmengine.dist import infer_launcher, init_dist

from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    Float8Config,
    FSDPConfig,
    LRConfig,
    MoEEngineConfig,
    MoELossConfig,
)
from xtuner.v1.float8.float8_tensor import ScalingGranularity
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.train.trainer import Trainer

QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
ALPACAL_PATH = os.environ["ALPACAL_PATH"]


def main():
    random.seed(99)

    dist_launcher = infer_launcher()
    init_dist(dist_launcher, backend="nccl")
    os.environ["DG_CACHE_DIR"] = f"/tmp/.deep_gemm-{torch.distributed.get_rank()}"

    moe_cfgs = [
        Qwen3MoE30BA3Config(),
        # Qwen3MoE30BA3Config(ep_size=8, dispatcher="all2all"),
        Qwen3MoE30BA3Config(
            ep_size=1,
            float8_cfg=Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TILEWISE,
                scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            ),
        ),
    ]
    for moe_cfg in moe_cfgs:

        optim_cfg = AdamWConfig()
        lr_cfg = LRConfig(total_steps=1000)
        fsdp_cfg = FSDPConfig(
            torch_compile=False,
            cpu_offload=False,
            ep_size=moe_cfg.ep_size,
            # hsdp_sharding_size=4,
        )
        engine_config = MoEEngineConfig(
            model=moe_cfg,
            optim=optim_cfg,
            lr=lr_cfg,
            fsdp=fsdp_cfg,
            moe_loss=MoELossConfig(balancing_loss_type="sigmoid"),
        )
        dataset_config = DatasetConfig(
            meta_datas={
                "alpaca": {
                    "annotation": ALPACAL_PATH,
                    "sample_ratio": 1,
                }
            },
            dataset_args={},
        )

        dataloader_config = DataloaderConfig(
            pack_max_length=512,
            max_length=512,
        )
        trainer = Trainer(
            model_path=QWEN3_MOE_PATH,
            engine_config=engine_config,
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
            tokenizer=QWEN3_MOE_PATH,
            global_batch_size=16,
            epoch_num=1,
            work_dir="/tmp/qwen3_moe_test",
        )
        trainer.fit()
        del trainer


if __name__ == "__main__":
    main()
