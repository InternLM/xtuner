"""Integration test: JsonlDataset + PresetPackDataset + PresetSampler via DataloaderConfig.build."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig

tokenizer_path = os.environ["QWEN3_MOE_PATH"]


def test_preset_dataloader_build_is_deterministic(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "data.jsonl"
    line = {"messages": [{"role": "pretrain", "content": "hello preset dataloader.\n"}]}
    jsonl_path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    anno = str(jsonl_path.resolve())

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tok_fn = PretrainTokenizeFunctionConfig().build(tokenizer)
    n_tokens = tok_fn(line)["num_tokens"]

    # TODO: use hard pack logic
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    np.save(pack_dir / "boundaries.npy", np.array([0, 1], dtype=np.int64))
    np.save(
        pack_dir / "samples.npy",
        np.array([[0, 0, -1, -1, 0, n_tokens]], dtype=np.int64),
    )
    (pack_dir / "paths.json").write_text(json.dumps([anno]), encoding="utf-8")

    # TODO: a dist version of this test
    # TODO: use group by length logic
    order_path = tmp_path / "order.npy"
    np.save(order_path, np.array([0, 0], dtype=np.int64))

    dataloader_cfg = DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(name="preset", anno_path=anno),
                "tokenize_fn": PretrainTokenizeFunctionConfig(),
            }
        ],
        pack_level="preset",
        pack_config_path=str(pack_dir),
        sampler_config_path=str(order_path),
        pack_max_length=n_tokens,
        num_workers=0,
    )

    def _build(seed: int, shuffle: bool = False):
        return dataloader_cfg.build(
            tokenizer,
            dp_mesh=None,
            global_batch_size=2,
            micro_batch_size=1,
            seed=seed,
            shuffle=shuffle,
        )

    # seed and shuffle should not affect the output in preset mode
    dl_a = _build(1, shuffle=False)
    dl_b = _build(2, shuffle=True)
    for batch_a, batch_b in zip(dl_a, dl_b):
        assert len(batch_a) == len(batch_b)
        for x, y in zip(batch_a, batch_b):
            assert torch.equal(x["seq_ctx"].input_ids, y["seq_ctx"].input_ids)
            assert torch.equal(x["shifted_labels"], y["shifted_labels"])
