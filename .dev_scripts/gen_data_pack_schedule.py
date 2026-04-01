"""
Generate preset pack config and sampler order from XTuner config.

This script:
1. Reads an XTuner config file (config.py)
2. Runs hard packing algorithm to generate pack_infos
3. Converts pack_infos to the preset pack config format (NPY directory)
4. Generates sampler order (optional: sequential, shuffle, or length-grouped)

Output files:
- pack_config_dir/
  - boundaries.npy: (num_packs+1,) int64, CSR boundary array
  - samples.npy: (total_slices, 6) int64, each row is [path_id, sample_idx, char_start, char_end, token_start, token_end]
  - paths.json: list[str], path_id -> dataset_path mapping
- sampler_order.npy: (num_samples,) int64, global pack consumption order

Usage:
    python gen_data_pack_schedule.py --config path/to/config.py --output-dir ./pack_output
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from transformers import AutoTokenizer
from xtuner.v1.datasets import build_datasets
from xtuner.v1.datasets.packing import get_pack_infos_by_hard_split
from xtuner.v1.datasets.utils import (
    concat_cumulative_sizes_from_lengths,
    get_pack_config_from_pack_infos_by_hard_split,
    get_sampler_config,
)
from xtuner.v1.utils import Config, get_logger
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache
from xtuner.v1.utils.profile import profile_time


logger = get_logger()


@contextmanager
def suppress_logger(target_level: str = "WARNING"):
    """Temporarily suppress logger output below target_level.

    Args:
        target_level: Minimum log level to display (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Get the global logger instance
    from xtuner.v1.utils.logger import _LOGGER

    if _LOGGER is None:
        yield
        return

    # Store original handlers configuration
    original_handlers = {}
    for handler_id, handler in _LOGGER._core.handlers.items():
        original_handlers[handler_id] = handler.levelno

    try:
        # Temporarily raise all handlers to target_level
        _LOGGER.remove()
        import sys

        from xtuner.v1.utils.logger import log_format

        _LOGGER.add(sys.stderr, level=target_level, format=log_format(), enqueue=True)

        yield

    finally:
        # Restore original configuration
        _LOGGER.remove()
        import os
        import sys

        from xtuner.v1.utils.logger import log_format

        log_level = os.environ.get("XTUNER_LOG_LEVEL", "INFO").upper()
        _LOGGER.add(sys.stderr, level=log_level, format=log_format(debug=log_level == "DEBUG"), enqueue=True)


def main():
    parser = argparse.ArgumentParser(description="Generate pack config and sampler order from XTuner config")
    parser.add_argument("--config", type=str, required=True, help="Path to XTuner config.py")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for pack config and sampler order"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for packing and sampling")
    parser.add_argument("--pack-workers", type=int, default=8, help="Number of workers for packing computation")
    parser.add_argument(
        "--sampler-strategy",
        type=str,
        choices=["sequential", "shuffle", "length_grouped"],
        default="length_grouped",
        help="Sampler order generation strategy",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="World size (number of DP ranks) for the target training job",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Generating Pack Config and Sampler Order")
    logger.info("=" * 80)

    # Load config
    config = Config.fromfile(args.config)
    dataset_config = config.dataset_config
    dataloader_config = config.dataloader_config
    tokenizer_path = config.trainer.tokenizer_path
    pack_max_length = dataloader_config.pack_max_length
    global_pack = dataloader_config.global_pack
    global_batch_size = config.trainer.global_batch_size

    if global_batch_size is None:
        raise ValueError("config.trainer.global_batch_size must be specified for sampler order generation.")

    if not global_pack:
        raise ValueError("Only support global_pack=True!")

    logger.info(f"Config: {args.config}")
    logger.info(f"Pack max length: {pack_max_length}")
    logger.info(f"Global batch size: {global_batch_size}")
    logger.info(f"World size: {args.world_size}")
    logger.info(f"Sampler strategy: {args.sampler_strategy}")
    logger.info(f"Seed: {args.seed}")

    sample_ratios = {}
    seq_sampler_switches = {}

    print(f"len of ds: {len(dataset_config)}")

    # patch so that dataset sample ratio is fixed to one
    for idx, ds_cfg_dict in enumerate(dataset_config):
        ds_cfg = ds_cfg_dict["dataset"]
        ds_name = f"ds_{idx}"
        ds_cfg.name = ds_name
        sample_ratios[ds_name] = float(ds_cfg.sample_ratio)
        ds_cfg.sample_ratio = 1.0
        seq_sampler_switches[ds_name] = ds_cfg.enable_sequential_sampler

    # Build tokenizer
    monkey_patch_hf_modules_cache()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Build datasets
    with profile_time("Building datasets"):
        with suppress_logger("WARNING"):
            datasets = build_datasets(dataset_config, tokenizer)

    print(f"len of ds after build: {len(datasets)}")

    # Collect dataset paths, num_tokens, and build indices for each dataset
    dataset_paths: list[str] = []
    all_num_tokens: list[np.ndarray] = []
    dataset_lengths: list[int] = []
    all_flat_inds: list[np.ndarray] = []

    total_samples = 0
    total_sampled = 0
    total_tokens = 0

    rng = np.random.RandomState(args.seed)
    offset = 0

    for ds in datasets:
        dataset_paths.append(ds.path)
        sample_ratio: float = sample_ratios[ds.name]
        num_tokens = ds.num_tokens
        enable_sequential_sampler: bool = seq_sampler_switches[ds.name]

        if num_tokens is None:
            raise ValueError(f"Dataset {ds.path} has no num_tokens cache. Please run tokenization first.")

        ds_len = len(ds)
        target_num_samples = int(ds_len * sample_ratio)

        # Build local indices based on enable_sequential_sampler and sample_ratio
        if enable_sequential_sampler:
            # Sequential: take first target_num_samples
            local_inds = np.arange(target_num_samples, dtype=np.int64)
        else:
            # Shuffle then take first target_num_samples
            local_inds = np.arange(ds_len, dtype=np.int64)
            rng.shuffle(local_inds)
            local_inds = local_inds[:target_num_samples]

        # Convert to flat concat indices
        flat_inds = local_inds + offset
        all_flat_inds.append(flat_inds)

        all_num_tokens.append(num_tokens)
        dataset_lengths.append(ds_len)
        total_samples += ds_len
        total_sampled += target_num_samples
        total_tokens += int(num_tokens.sum())
        offset += ds_len

    logger.info(f"Total datasets: {len(datasets)}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total sampled (after sample_ratio): {total_sampled}")
    logger.info(f"Total tokens: {total_tokens:,}")

    # Build concat arrays
    concat_num_tokens = np.concatenate(all_num_tokens)
    concat_cumulative_sizes = concat_cumulative_sizes_from_lengths(dataset_lengths)
    concat_inds = np.concatenate(all_flat_inds)

    # Global shuffle of all flat indices
    rng.shuffle(concat_inds)

    # Run packing once with ConcatDataset-style flat indices
    with profile_time("Running packing"):
        pack_infos = get_pack_infos_by_hard_split(
            inds=concat_inds,
            dataset_id=0,  # Single virtual dataset
            num_tokens=concat_num_tokens,
            pack_max_length=pack_max_length,
            pack_workers=args.pack_workers,
        )

    num_packs = len(pack_infos["dataset_id"])
    logger.info(f"Generated {num_packs} packs")

    # Convert to preset format with concat_cumulative_sizes
    with profile_time("Converting to preset format"):
        preset_config = get_pack_config_from_pack_infos_by_hard_split(
            pack_infos,
            path_id=0,
            num_tokens=concat_num_tokens,
            paths=dataset_paths,
            concat_cumulative_sizes=concat_cumulative_sizes,
        )

    # Save pack config
    output_dir = Path(args.output_dir)
    pack_config_dir = output_dir / "pack_config"
    pack_config_dir.mkdir(parents=True, exist_ok=True)

    with profile_time("Saving pack config"):
        np.save(pack_config_dir / "boundaries.npy", preset_config["boundaries"])
        np.save(pack_config_dir / "samples.npy", preset_config["samples"])

        with open(pack_config_dir / "paths.json", "w", encoding="utf-8") as f:
            json.dump(dataset_paths, f, ensure_ascii=False, indent=2)

    # Generate sampler order
    with profile_time("Generating sampler order"):
        longest = pack_infos["longest"]
        sampler_order_path = output_dir / "sampler_order.npy"

        sampler_order = get_sampler_config(
            order_path=sampler_order_path,
            mode=args.sampler_strategy,
            num_packs=num_packs,
            longest=longest,
            global_batch_size=global_batch_size,
            world_size=args.world_size,
            seed=args.seed,
            epoch=0,
        )

    # Print summary
    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info(f"  Total packs: {num_packs}")
    logger.info(f"  Total slices: {len(preset_config['samples'])}")
    logger.info(f"  Sampler length: {len(sampler_order)}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
