import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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
from xtuner.v1.model import InternS1MiniConfig
from xtuner.v1.train.trainer import Trainer
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.loss import CELossConfig
import argparse

INTERNS1_DENSE_PATH = os.environ["INTERNS1_DENSE_PATH"]
INTERNS1_DATA_META = os.environ["INTERNS1_DATA_META"]

lr = [
    0.000060,
    0.000059,
    0.000057,
    0.000053,
    0.000047,
    0.000041,
    0.000034,
    0.000027,
    0.000020,
    0.000014,
    0.000008,
    0.000004,
    0.000002,
]
reduced_llm_loss_sp1 = [
    2.118,
    1.991,
    2.061,
    1.959,
    1.985,
    1.979,
    1.870,
    1.943,
    1.797,
    1.815,
    1.881,
    1.906,
    1.490,
]
reduced_llm_loss_sp2 = [
    2.077,
    1.968,
    2.138,
    2.079,
    1.981,
    1.929,
    1.912,
    1.903,
    1.828,
    1.902,
    1.856,
    1.832,
    1.449,
]
grad_norm_sp1 = [
    6.966,
    2.800,
    11.252,
    5.580,
    2.915,
    2.135,
    3.095,
    2.321,
    2.023,
    1.198,
    2.579,
    2.132,
    3.048,
]
grad_norm_sp2 = [
    6.724,
    3.378,
    14.701,
    2.938,
    2.848,
    1.837,
    1.979,
    1.623,
    2.565,
    4.422,
    1.780,
    2.792,
    1.905,
]
max_memory_sp1 = [
    23.46,
    31.79,
    31.49,
    31.43,
    31.43,
    31.43,
    31.85,
    31.43,
    31.79,
    31.79,
    31.43,
    31.85,
    31.43,
]
max_memory_sp2 = [
    25.58,
    33.28,
    29.35,
    29.58,
    29.44,
    29.42,
    29.44,
    29.43,
    29.18,
    29.58,
    29.56,
    29.51,
    29.57,
]
text_tokens_sp1 = [
    13677.0,
    14123.0,
    11721.0,
    12677.0,
    13060.0,
    11776.0,
    14724.0,
    12467.0,
    12731.0,
    15550.0,
    13933.0,
    13479.0,
    13193.0,
]

text_tokens_sp2 = [
    16384.0,
    16384.0,
    16384.0,
    16384.0,
    16384.0,
    16384.0,
    16384.0,
    16384.0,
    16384.0,
    16384.0,
    16384.0,
    16384.0,
    16384.0,
]

tgs_sp1 = [
    1649.5,
    2301.6,
    2514.0,
    2682.9,
    2820.4,
    2869.1,
    3003.4,
    3037.3,
    3076.0,
    3169.7,
    3216.4,
    3248.6,
    3269.7,
]
tgs_sp2 = [
    1743.4,
    2049.7,
    2371.3,
    2565.2,
    2703.9,
    2804.6,
    2878.6,
    2938.1,
    2986.9,
    3025.1,
    3057.5,
    3085.5,
    3109.9,
]


# Note: export XTUNER_DETERMINISTIC=true
def parse_args():
    parser = argparse.ArgumentParser(description="Test MLLM SFT Trainer")
    parser.add_argument(
        "--work_dir",
        type=str,
        default='work_dirs'
    )
    return parser.parse_args()


def extract_data_from_log(logfile: Path):
    pattern_str = r"\[XTuner\].*Step.*lr:\s(\d+.\d*)\s.*text_tokens:\s(\d+.\d*)\s.*reduced_llm_loss:\s(\d+.\d*)\s.*max_memory:\s(\d+.\d*)\s*GB\s.*grad_norm:\s(\d+.\d*)\s.*e2e_tgs:\s(\d+.\d*)"
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

    exp_paths = []
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
            num_workers=8,
            pack_max_length=8192
        )
        work_dir = f"{args.work_dir}-sp{sp_size}-intern-s1-mini"
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
            epoch_num=1,
        )
        trainer.fit()
        if dist.get_rank() == 0:
            exp_paths.append(trainer.exp_dir)

        dist.barrier()
        del trainer
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    sp_sizes = [sp_size for _, sp_size in model_cfgs]
    for exp_path, sp_size in zip(exp_paths, sp_sizes):
        rank0_log_path = Path(exp_path) / "rank0.log"
        (
            cur_lr,
            cur_text_tokens,
            cur_reduced_llm,
            cur_max_memory,
            cur_grad_norm,
            cur_tgs,
        ) = extract_data_from_log(rank0_log_path)
        plot_dir = Path(exp_path) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison_curves(lr, cur_lr, "lr", output_root=plot_dir)
        plot_comparison_curves(
            eval(f'reduced_llm_loss_sp{sp_size}'), cur_reduced_llm, "reduced-loss", output_root=plot_dir
        )
        plot_comparison_curves(
            eval(f'grad_norm_sp{sp_size}'), cur_grad_norm, "grad_norm", output_root=plot_dir
        )
        plot_comparison_curves(
            eval(f'max_memory_sp{sp_size}'), cur_max_memory, "max_memory", output_root=plot_dir
        )
        plot_comparison_curves(
            eval(f'text_tokens_sp{sp_size}'), cur_text_tokens, "text_tokens", output_root=plot_dir
        )
        plot_comparison_curves(eval(f'tgs_sp{sp_size}'), cur_tgs, "tgs", output_root=plot_dir)


if __name__ == "__main__":
    main()
