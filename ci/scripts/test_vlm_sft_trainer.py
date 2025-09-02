import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
import json
from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    FSDPConfig,
    LRConfig
)
from xtuner.v1.datasets import InternS1TokenizeFnConfig
from xtuner.v1.model.interns1 import InternS1MiniConfig
from xtuner.v1.loss import CELossContext
from xtuner.v1.train.trainer import Trainer
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.loss import CELossConfig
import argparse

INTERNS1_DENSE_PATH = os.environ["INTERNS1_DENSE_PATH"]
INTERNS1_DATA_META = os.environ["INTERNS1_DATA_META"]

lr = [
    0.000060,
    0.000059,
    0.000058,
    0.000056,
    0.000052,
    0.000048,
    0.000044,
    0.000039,
    0.000033,
    0.000028,
    0.000022,
    0.000017,
    0.000013,
    0.000009,
    0.000005,
    0.000003,
    0.000002,
]
reduced_llm_loss = [
    2.453,
    1.465,
    1.492,
    1.404,
    1.267,
    1.261,
    1.238,
    1.218,
    1.201,
    1.206,
    1.195,
    1.189,
    1.188,
    1.188,
    1.169,
    1.199,
    1.141,
]
grad_norm = [
    23.950,
    5.20,
    6.07,
    4.25,
    2.27,
    0.87,
    1.59,
    1.03,
    0.65,
    1.05,
    0.83,
    0.58,
    0.47,
    0.52,
    0.51,
    0.52,
    0.59,
]
max_memory = [
    34.5,
    37.5,
    37.5,
    37.4,
    37.5,
    37.4,
    37.4,
    37.5,
    37.5,
    37.4,
    37.4,
    37.5,
    37.5,
    37.4,
    37.5,
    37.5,
    37.4,
]
text_tokens = [
    16302.0,
    16364.0,
    16376.0,
    15874.0,
    16360.0,
    16328.0,
    16341.0,
    16377.0,
    16356.0,
    16254.0,
    16329.0,
    16347.0,
    16359.0,
    15998.0,
    16356.0,
    16371.0,
    16307.0,
]
tgs = [
    516,
    1369,
    2219,
    2373,
    2429,
    2517,
    2758,
    2795,
    2812,
    2591,
    2622,
    2676,
    2663,
    2628,
    2872,
    2734,
    2920,
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test VLLM SFT Trainer")
    parser.add_argument(
        "work_dir",
        type=str,
    )
    return parser.parse_args()


def extract_data_from_log(logfile: Path):
    pattern_str = r"\[XTuner\].*Step.*lr:\s(\d+.\d*)\s.*text_tokens:\s(\d+.\d*)\s.*reduced_llm_loss:\s(\d+.\d*)\s.*max_memory:\s(\d+.\d*)\s*GB\s.*grad_norm:\s(\d+.\d*)\s.*tgs:\s(\d+.\d*)"
    compiled_pattern = re.compile(pattern_str)

    cur_lr = []
    cur_reduced_llm = []
    cur_grad_norm = []
    cur_max_memory = []
    cur_text_tokens = []
    cur_tgs = []

    with open(logfile) as f:
        for data in f:
            if match := compiled_pattern.search(data):
                cur_lr.append(float(match.group(1)))
                cur_text_tokens.append(float(match.group(2)))
                cur_reduced_llm.append(float(match.group(3)))
                cur_max_memory.append(float(match.group(4)))
                cur_grad_norm.append(float(match.group(5)))
                cur_tgs.append(float(match.group(6)))
    return (
        cur_lr,
        cur_text_tokens,
        cur_reduced_llm,
        cur_max_memory,
        cur_grad_norm,
        cur_tgs,
    )


def plot_comparison_curves(history_data, current_data, title, output_root: Path):
    """
    Plot comparison curves between two sets of data.

    Args:
        history_data: List of historical data points
        current_data: List of current data points
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))

    # Create x-axis step values
    x_history = np.arange(len(history_data))
    x_current = np.arange(len(current_data))

    # Plot both lines
    plt.plot(x_history, history_data, "r--", label="History", marker="x", markersize=4)
    plt.plot(x_current, current_data, "b-", label="Current", marker="o", markersize=4)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_root / f"{title.replace(' ', '_')}_comparison.png")
    plt.close()

    print(f"Comparison plot saved as {title.replace(' ', '_')}_comparison.png")


def main():
    args = parse_args()
    maybe_compile.clear_compile_targets()

    model_cfg_1 = InternS1MiniConfig()

    model_cfgs = [
        (model_cfg_1, 1),
        (model_cfg_1, 2),
    ]

    ds_collections = json.loads(open(INTERNS1_DATA_META).read())

    for model_cfg, sp_size in model_cfgs:
        optim_cfg = AdamWConfig(lr=6e-05, foreach=False)
        lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
        fsdp_cfg = FSDPConfig(
            torch_compile=False,
            cpu_offload=False
            # hsdp_sharding_size=4,
        )

        dataset_config = []
        for name, _data in ds_collections.items():
            _data_cfg = {"dataset": DatasetConfig(name=name,
                                                  anno_path=_data['annotation'],
                                                  media_root=_data.get('media_root', ''),
                                                  sample_ratio=_data.get('sample_ratio', 1.0),
                                                  class_name='VLMJsonlDataset'),
                         "tokenize_fn": InternS1TokenizeFnConfig(model_cfg=model_cfg,
                                                                 max_dynamic_patch=_data.get('max_dynamic_patch', None),
                                                                 min_dynamic_patch=_data.get('min_dynamic_patch', None),
                                                                 min_num_frames=_data.get('min_num_frames', 4),
                                                                 max_num_frames=_data.get('max_num_frames', 24),
                                                                 data_augment=_data.get('data_augment', False),
                                                                 system_message=_data.get('system_message', None),
                                                                 hash=_data.get('hash', None)
                                                                 )
                         }
            dataset_config.append(_data_cfg)

        dataloader_config = DataloaderConfig(
            collator="sft_vllm_collator",
            pack_max_length=8192,
            max_length=8192,
        )
        work_dir = f"{args.work_dir}-{name}"
        loss_cfg = CELossConfig(mode="chunk", chunk_size=1024, ignore_idx=-100)
        trainer = Trainer(
            load_from=INTERNS1_DENSE_PATH,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
            dataset_cfg=dataset_config,
            dataloader_cfg=dataloader_config,
            loss_cfg=loss_cfg,
            lr_cfg=lr_cfg,
            sp_size=sp_size,
            tokenizer_path=INTERNS1_DENSE_PATH,
            global_batch_size=16,
            work_dir=work_dir,
            seed=0,
            total_step=5,
        )
        trainer.fit()
        if dist.get_rank() == 0:
            rank0_log_path = Path(work_dir) / "rank0.log"
            (
                cur_lr,
                cur_text_tokens,
                cur_reduced_llm,
                cur_max_memory,
                cur_grad_norm,
                cur_tgs,
            ) = extract_data_from_log(rank0_log_path)
            work_dir = Path(work_dir)
            plot_dir = work_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_comparison_curves(lr, cur_lr, "lr", output_root=plot_dir)
            plot_comparison_curves(
                reduced_llm_loss, cur_reduced_llm, "reduced-loss", output_root=plot_dir
            )
            plot_comparison_curves(
                grad_norm, cur_grad_norm, "grad_norm", output_root=plot_dir
            )
            plot_comparison_curves(
                max_memory, cur_max_memory, "max_memory", output_root=plot_dir
            )
            plot_comparison_curves(
                text_tokens, cur_text_tokens, "text_tokens", output_root=plot_dir
            )
            plot_comparison_curves(tgs, cur_tgs, "tgs", output_root=plot_dir)
        # del trainer


if __name__ == "__main__":
    main()
