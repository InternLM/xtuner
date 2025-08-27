from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    FSDPConfig,
    LRConfig,
    BalancingLossConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.loss import CELossContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.train.trainer import Trainer
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Test SFT Trainer")
    parser.add_argument(
        "work_dir",
        type=str,
        help="The directory to save the training results.",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="huggingface model path",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="dataset dir or jsonl file path",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_path = args.model_path
    dataset_path = args.dataset_path

    moe_cfg = Qwen3MoE30BA3Config(balancing_loss_cfg=BalancingLossConfig(), vocab_size=153216)
    optim_cfg = AdamWConfig(lr=6e-05)
    lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
    fsdp_cfg = FSDPConfig(
        torch_compile=True,
        cpu_offload=False,
        ep_size=moe_cfg.ep_size,
        # hsdp_sharding_size=4,
    )
    dataset_config = [
        {
            "dataset": DatasetConfig(name="alpaca", anno_path=dataset_path, sample_ratio=1.0),
            "tokenize_fn": FTDPTokenizeFnConfig(max_length=32768),
        },
    ]

    dataloader_config = DataloaderConfig(
        pack_max_length=32768,
        pack_level="soft",
        num_workers=0,
    )
    loss_ctx = CELossContext(loss_class="liger_cross_entropy")
    trainer = Trainer(
        load_from=model_path,
        model_cfg=moe_cfg,
        optim_cfg=optim_cfg,
        fsdp_cfg=fsdp_cfg,
        dataset_cfg=dataset_config,
        dataloader_cfg=dataloader_config,
        loss_ctx=loss_ctx,
        lr_cfg=lr_cfg,
        tokenizer_path=args.model_path,
        global_batch_size=16,
        epoch_num=1,
        work_dir=args.work_dir,
        hf_interval=30,
        seed=0,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
