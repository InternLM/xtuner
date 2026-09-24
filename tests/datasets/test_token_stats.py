"""Tests for token length recording and statistics."""

import copy
import csv
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from mmengine.runner import set_random_seed
from pydantic import BaseModel
from tokenizers import Tokenizer, models, pre_tokenizers

from transformers import PreTrainedTokenizerFast
from xtuner.tools.token_stats import (
    _build_packed_dataset,
    main,
    save_reports,
    summarize_cache_manifest,
    summarize_datasets,
    summarize_packing,
    summarize_samples,
)
from xtuner.v1.datasets import (
    DataloaderConfig,
    FtdpTokenizeFunction,
    JsonlDataset,
    OpenaiTokenizeFunction,
    PretrainTokenizeFunction,
)
from xtuner.v1.datasets.collator import sft_llm_collator
from xtuner.v1.datasets.token_stats import TokenStatsTokenizeFunction


def _tokenizer() -> PreTrainedTokenizerFast:
    words = ["[UNK]", "[PAD]", "a", "b", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "[BOS]"]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|endoftext|>",
        additional_special_tokens=words[5:],
    )


def _annotation(path: Path, lengths: list[int]) -> Path:
    rows = [{"dialogs": [{"role": "pretrain_content", "content": "a " * length}]} for length in lengths]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _dataset(path: Path, cache: Path, record: bool, **kwargs: Any) -> JsonlDataset:
    fn = FtdpTokenizeFunction(_tokenizer(), chat_template="qwen", max_length=8, tokenizer_hash="local-test")
    if record:
        fn = TokenStatsTokenizeFunction(fn)
    return JsonlDataset(path, cache_dir=cache, tokenize_fn=fn, name="subset", **kwargs)


def _training_fields(data: dict) -> dict:
    return {key: value for key, value in data.items() if key != "original_num_tokens"}


def _assert_collation_equal(left: list, right: list, limit: int = 32) -> None:
    left_batch = sft_llm_collator(copy.deepcopy(left), pack_max_length=limit, padding_token_idx=1)
    right_batch = sft_llm_collator(copy.deepcopy(right), pack_max_length=limit, padding_token_idx=1)
    for a, b in zip(left_batch, right_batch):
        assert a.keys() == b.keys() == {"seq_ctx", "shifted_labels"}
        torch.testing.assert_close(a["shifted_labels"], b["shifted_labels"], rtol=0, atol=0)
        assert a["seq_ctx"].__dict__.keys() == b["seq_ctx"].__dict__.keys()
        for key, value in a["seq_ctx"].__dict__.items():
            other = b["seq_ctx"].__dict__[key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
        assert "original_num_tokens" not in a["seq_ctx"].__dict__


class TestTokenStatsRecording:
    @pytest.mark.parametrize("kind", ["openai", "glm52", "qwen35", "ftdp"])
    @pytest.mark.parametrize("relative_limit", [1, 0, -3])
    @pytest.mark.parametrize("state", ["runtime", "cache"])
    def test_cutoffs_preserve_tokens_labels_masks_and_encode_count(
        self, kind: str, relative_limit: int, state: str
    ) -> None:
        tokenizer = _tokenizer()
        tokenizer.bos_token = "[BOS]"
        messages = [{"role": "user", "content": "a b"}, {"role": "assistant", "content": "a b a b"}]
        cls = FtdpTokenizeFunction if kind == "ftdp" else OpenaiTokenizeFunction
        template = {"openai": "qwen3", "glm52": "glm5.2", "qwen35": "qwen3.5-vl", "ftdp": "qwen"}[kind]
        full_fn = cls(tokenizer, chat_template=template, max_length=1000, tokenizer_hash="test")
        full = full_fn(copy.deepcopy(messages))
        length = len(full["input_ids"])
        limit = length + relative_limit
        base = cls(tokenizer, chat_template=template, max_length=limit, tokenizer_hash="test")
        base.set_state(state)
        original_hash = base.hash()
        recorder = TokenStatsTokenizeFunction(base)
        recorder.set_state(state)
        with patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode:
            expected = base(copy.deepcopy(messages))
            calls = encode.call_count
            encode.reset_mock()
            actual = recorder(copy.deepcopy(messages))
            assert encode.call_count == calls
        assert _training_fields(actual) == expected
        assert actual["original_num_tokens"] == length
        assert actual["original_num_tokens"] - actual["num_tokens"] == max(length - limit, 0)
        assert actual["input_ids"] == full["input_ids"][:limit]
        assert base.max_length == limit and base.state == state and base.hash() == original_hash
        assert recorder.hash() != original_hash
        _assert_collation_equal([[expected]], [[actual]])

    @pytest.mark.parametrize("bos", [False, True])
    @pytest.mark.parametrize("eos", [False, True])
    def test_pretrain_special_tokens(self, bos: bool, eos: bool) -> None:
        tokenizer = _tokenizer()
        tokenizer.bos_token = "[BOS]"
        base = PretrainTokenizeFunction(tokenizer, add_bos_token=bos, add_eos_token=eos)
        actual = TokenStatsTokenizeFunction(base)({"content": "a b"})
        assert _training_fields(actual) == base({"content": "a b"})
        assert actual["original_num_tokens"] == 2 + bos + eos
        assert actual["original_num_tokens"] == actual["num_tokens"]

    def test_empty_conversation_is_unknown_and_limit_enters_cache_key(self) -> None:
        base = OpenaiTokenizeFunction(_tokenizer(), "qwen3", max_length=8, tokenizer_hash="test")
        recorder = TokenStatsTokenizeFunction(base)
        recorder.set_state("cache")
        assert recorder([]) == {"num_tokens": 0, "proxy_attn_flops": 0.0}
        other = TokenStatsTokenizeFunction(
            OpenaiTokenizeFunction(_tokenizer(), "qwen3", max_length=9, tokenizer_hash="test")
        )
        assert recorder.hash() != other.hash()

    @pytest.mark.parametrize("kind", ["openai", "ftdp"])
    @pytest.mark.parametrize("boundary", [1, 2, 3])
    def test_empty_think_patch_keeps_its_post_truncation_position(
        self, monkeypatch: pytest.MonkeyPatch, kind: str, boundary: int
    ) -> None:
        from xtuner.v1.datasets._hardcode_patch import _label_processor_wrapper, _SkipEmptyThink

        cls = OpenaiTokenizeFunction if kind == "openai" else FtdpTokenizeFunction
        base = cls(_tokenizer(), chat_template="qwen3" if kind == "openai" else "qwen", max_length=1000)
        messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b <think></think>a b"}]
        full = base(copy.deepcopy(messages))
        base.max_length = full["input_ids"].index(7) + boundary
        base._skip_seq = [7, 8, 2]
        base._skip_seq_fallback = [7, 8]
        base._think_token = 7
        # Apply the optional label patch without changing class inheritance.
        monkeypatch.setattr(cls, "process_labels", _SkipEmptyThink.process_labels, raising=False)
        monkeypatch.setattr(cls, "__call__", _label_processor_wrapper(cls.__call__))
        actual = TokenStatsTokenizeFunction(base)(copy.deepcopy(messages))
        assert _training_fields(actual) == base(copy.deepcopy(messages))


class TestTokenStatsCache:
    @pytest.mark.parametrize("as_model", [False, True])
    def test_cache_projection_accepts_callables_without_metadata_callbacks(self, as_model: bool) -> None:
        class CacheModel(BaseModel):
            num_tokens: int
            proxy_attn_flops: float
            input_ids: list[int]
            chunks: list[dict[str, int]]

        metadata = {
            "num_tokens": 5,
            "proxy_attn_flops": 3.0,
            "chunks": [{"char_start": 0, "char_end": 5, "token_start_offset": 0}],
        }
        tokenized = {**metadata, "input_ids": [2] * 5}
        if as_model:
            tokenized = CacheModel(**tokenized)
        assert JsonlDataset._tokenize_by_offset(b"{}", lambda _: tokenized) == metadata

    def test_cache_records_unknown_length_for_damaged_sample(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text('{"messages": []}\n')
        recorder = TokenStatsTokenizeFunction(
            OpenaiTokenizeFunction(_tokenizer(), "qwen3", max_length=8, tokenizer_hash="test")
        )
        dataset = JsonlDataset(path, cache_dir=tmp_path / "cache", tokenize_fn=recorder, disable_filter=True)
        assert dataset.num_tokens.tolist() == [0]
        assert dataset._meta["original_num_tokens"].tolist() == [-1]
        cached = np.load(next((tmp_path / "cache").glob("*/*/jsonl_meta/original_num_tokens.npy")))
        np.testing.assert_array_equal(cached, [-1])
        assert cached.dtype == np.dtype("int64")

    @pytest.mark.parametrize("template", ["qwen3", "glm5.2", "qwen3.5-vl"])
    @pytest.mark.parametrize("limits", [(8, 16), (16, 8), (8, None), (None, 8)])
    def test_openai_length_change_rebuilds_statistics_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, template: str, limits: tuple[int | None, int | None]
    ) -> None:
        monkeypatch.setenv("XTUNER_TOKENIZE_WORKERS", "1")
        tokenizer = _tokenizer()
        rows = [
            {"messages": [{"role": "user", "content": "a " * length}, {"role": "assistant", "content": "b " * length}]}
            for length in (4, 12, 40, 80)
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        full_fn = OpenaiTokenizeFunction(tokenizer, template, tokenizer_hash="test")
        original = [full_fn(copy.deepcopy(row))["num_tokens"] for row in rows]
        assert min(original) > 8 and max(original) > 16

        def build(limit: int | None, record: bool = False) -> JsonlDataset:
            fn = OpenaiTokenizeFunction(tokenizer, template, max_length=limit, tokenizer_hash="test")
            return JsonlDataset(
                path,
                cache_dir=tmp_path / ("cache" if record else f"training_{limit}"),
                name="subset",
                tokenize_fn=TokenStatsTokenizeFunction(fn) if record else fn,
            )

        for limit in limits:
            training = build(limit)
            expected = [length if limit is None else min(length, limit) for length in original]
            assert training.num_tokens.tolist() == expected
            stats = build(limit, record=True)
            assert stats.num_tokens.tolist() == expected
            assert stats._meta["original_num_tokens"].tolist() == original
            for index, length in enumerate(expected):
                assert training[index]["num_tokens"] == len(training[index]["input_ids"]) == length
                assert training[index] == _training_fields(stats[index])

        with patch.object(JsonlDataset, "count_tokens", side_effect=AssertionError("must reuse matching cache")):
            for limit in limits:
                training, stats = build(limit), build(limit, record=True)
                assert training.num_tokens.tolist() == [
                    length if limit is None else min(length, limit) for length in original
                ]
                np.testing.assert_array_equal(stats.num_tokens, training.num_tokens)
                for index in range(len(training)):
                    assert training[index]["num_tokens"] == training.num_tokens[index]
                    assert training[index] == _training_fields(stats[index])

    @pytest.mark.parametrize("limits", [(8, 16), (16, 8)])
    def test_ftdp_length_change_rebuilds_statistics_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limits: tuple[int, int]
    ) -> None:
        monkeypatch.setenv("XTUNER_TOKENIZE_WORKERS", "1")
        path = _annotation(tmp_path / "data.jsonl", [4, 12, 100])

        def build(limit: int, record: bool = False) -> JsonlDataset:
            fn = FtdpTokenizeFunction(_tokenizer(), chat_template="qwen", max_length=limit, tokenizer_hash="test")
            return JsonlDataset(
                path,
                cache_dir=tmp_path / ("cache" if record else f"training_{limit}"),
                name="subset",
                tokenize_fn=TokenStatsTokenizeFunction(fn) if record else fn,
            )

        for limit in limits:
            training = build(limit)
            expected = [min(length, limit) for length in (5, 13, 101)]
            assert training.num_tokens.tolist() == expected
            stats = build(limit, record=True)
            assert stats.num_tokens.tolist() == expected
            assert stats._meta["original_num_tokens"].tolist() == [5, 13, 101]
            for index, length in enumerate(expected):
                assert len(training[index]["input_ids"]) == length
                assert training[index] == _training_fields(stats[index])
        with patch.object(JsonlDataset, "count_tokens", side_effect=AssertionError("must reuse matching cache")):
            for limit in limits:
                assert build(limit).num_tokens.tolist() == [min(length, limit) for length in (5, 13, 101)]
                assert build(limit, record=True).num_tokens.tolist() == build(limit).num_tokens.tolist()

    @pytest.mark.parametrize("same_name", [False, True])
    def test_distinct_datasets_may_share_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_name: bool
    ) -> None:
        monkeypatch.setenv("XTUNER_TOKENIZE_WORKERS", "1")
        path = _annotation(tmp_path / "data.jsonl", [4, 12, 100])
        first = _dataset(path, tmp_path / "cache", True)
        second = JsonlDataset(
            path,
            cache_dir=tmp_path / "cache",
            name="subset" if same_name else "other",
            tokenize_fn=TokenStatsTokenizeFunction(
                FtdpTokenizeFunction(_tokenizer(), chat_template="qwen", max_length=16, tokenizer_hash="local-test")
            ),
        )
        rows = summarize_datasets([first, second])
        assert len(rows) == (1 if same_name else 2)
        assert sum(row["count"] for row in rows) == 6
        assert sum(row["truncated_tokens"] for row in rows) == 98 + 85
        if not same_name:
            assert {row["name"]: row["truncated_tokens"] for row in rows} == {"subset": 98, "other": 85}
        with pytest.raises(ValueError, match="Dataset object supplied more than once"):
            summarize_datasets([first, first])

    @pytest.mark.parametrize("workers", [1, 2])
    def test_cache_sampling_and_old_cache_compatibility(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
    ) -> None:
        monkeypatch.setenv("XTUNER_TOKENIZE_WORKERS", str(workers))
        path = _annotation(tmp_path / "data.jsonl", [4, 7, 11])  # including EOS: 5, 8, 12
        old = _dataset(path, tmp_path / "cache", False, sample_ratio=1.5, enable_sequential_sampler=True)
        new = _dataset(path, tmp_path / "cache", True, sample_ratio=1.5, enable_sequential_sampler=True)
        np.testing.assert_array_equal(old.num_tokens, new.num_tokens)
        np.testing.assert_array_equal(old._meta["offsets"], new._meta["offsets"])
        np.testing.assert_array_equal(new._meta["original_num_tokens"], [5, 8, 12, 5])
        assert new._meta.keys() - old._meta.keys() == {"original_num_tokens"}
        assert len(new) == len(old) == 4
        for index in range(len(old)):
            assert old[index] == _training_fields(new[index])
        with patch.object(JsonlDataset, "count_tokens", side_effect=AssertionError("must reuse old cache")):
            loaded = _dataset(path, tmp_path / "cache", False)
        assert "original_num_tokens" not in loaded._meta
        assert loaded[0] == old[0]
        old_row = summarize_datasets([loaded])[0]
        assert old_row["unknown_original_count"] == 3
        assert old_row["truncated_tokens"] is None
        row = summarize_datasets([new])[0]
        assert row["scope"] == "sampled_instances_before_packing"
        assert row["original_total_tokens"] == 30
        assert row["truncated_count"] == 1 and row["truncated_tokens"] == 4
        assert row["truncated_sample_ratio"] == 1 / 4
        assert row["truncated_token_ratio"] == 4 / 30
        metas = list((tmp_path / "cache").glob("*/*/jsonl_meta"))
        assert len(metas) == 2
        recorded_meta = next(meta for meta in metas if (meta / "original_num_tokens.npy").exists())
        old_meta = next(meta for meta in metas if meta != recorded_meta)
        assert {p.name for p in recorded_meta.iterdir()} - {p.name for p in old_meta.iterdir()} == {
            "original_num_tokens.npy"
        }
        manifest = [{"name": str(i), "meta_dirs": [str(meta)]} for i, meta in enumerate(metas)]
        assert summarize_cache_manifest(manifest, workers=1) == summarize_cache_manifest(manifest, workers=2)
        raw = [r for r in summarize_cache_manifest(manifest) if r["known_original_count"]][0]
        assert raw["count"] == 3 and raw["original_total_tokens"] == 25

    def test_distributed_cache_has_no_duplicate_samples_or_empty_rank_failure(self, tmp_path: Path) -> None:
        path = _annotation(tmp_path / "one.jsonl", [4])
        torch.multiprocessing.spawn(_distributed_cache_worker, args=(str(tmp_path), str(path)), nprocs=2, join=True)
        reports = list((tmp_path / "reports").glob("*/subset_token_stats.csv"))
        assert len(reports) == 1
        with reports[0].open() as stream:
            row = next(csv.DictReader(stream))
        assert row["count"] == "1" and row["original_total_tokens"] == "5"


class TestTokenStatsConfig:
    @pytest.mark.parametrize("packing", [False, True])
    @pytest.mark.parametrize("legacy", [False, True])
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_training_config_matches_factory_without_mutation_or_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool, wrapped: bool, packing: bool
    ) -> None:
        from xtuner.v1.datasets import DataloaderConfig
        from xtuner.v1.train import Trainer
        from xtuner.v1.utils import Config

        monkeypatch.setenv("XTUNER_TOKENIZE_WORKERS", "1")
        _annotation(tmp_path / "data.jsonl", [4, 12, 100])
        _tokenizer().save_pretrained(tmp_path / "tokenizer")
        config = tmp_path / "train_config.py"
        config.write_text(
            f"legacy = {legacy}\nwrapped = {wrapped}\n"
            + """
from pathlib import Path
from transformers import AutoTokenizer
from xtuner.v1.config import AdamWConfig, LRConfig
from xtuner.v1.datasets import DataloaderConfig, DatasetConfig, FTDPTokenizeFnConfig, build_datasets
from xtuner.v1.datasets.token_stats import TokenStatsConfig
from xtuner.v1.model import Glm52MoEConfig
from xtuner.v1.train import TrainerConfig

base = Path(__file__).parent
tokenize = FTDPTokenizeFnConfig(chat_template="qwen", max_length=8)
sources = [{
    "dataset": DatasetConfig(name="subset", anno_path=base / "data.jsonl",
        cache_dir=base / "cache", sample_ratio=1.5),
    "tokenize_fn": TokenStatsConfig(tokenize) if wrapped else tokenize,
}]
loader = DataloaderConfig(dataset_config_list=[] if legacy else sources,
    pack_level="hard", pack_max_length=8, pack_workers=1, pack_chunk_size=2,
    tokenizer_hash="reuse-training-hash")
trainer = TrainerConfig(model_cfg=Glm52MoEConfig(), optim_cfg=AdamWConfig(lr=1e-6), lr_cfg=LRConfig(),
    dataloader_cfg=loader, dataset_cfg=sources if legacy else None,
    tokenizer_path=base / "tokenizer", global_batch_size=1, seed=17)
seed = 999  # The direct entry must use trainer.seed.
"""
        )
        loaded = Config.fromfile(config)
        original = loaded["sources"][0]["tokenize_fn"]
        factory_config = tmp_path / "factory_config.py"
        factory_config.write_text(
            config.read_text()
            + """
seed = trainer.seed
def build_stats_inputs(work_dir):
    return {"datasets": build_datasets([{
        **entry,
        "tokenize_fn": entry["tokenize_fn"] if wrapped else TokenStatsConfig(entry["tokenize_fn"]),
    } for entry in sources], AutoTokenizer.from_pretrained(trainer.tokenizer_path),
        tokenizer_hash=loader.tokenizer_hash), "dataloader_config": loader}
"""
        )
        expected = snapshot = expected_packing = None
        for phase, selected in (("cold", config), ("warm", config), ("factory", factory_config)):
            output = tmp_path / phase
            arguments = ["token_stats", "--config", str(selected), "--output-dir", str(output)]
            monkeypatch.setattr(sys, "argv", arguments + (["--packing"] if packing else []))
            guard = (
                nullcontext()
                if phase == "cold"
                else patch.object(
                    JsonlDataset, "count_tokens", side_effect=AssertionError("must reuse matching cache")
                )
            )
            config_guard = (
                patch.object(Config, "fromfile", return_value=loaded) if selected == config else nullcontext()
            )
            with (
                guard,
                config_guard,
                patch.object(Trainer, "from_config", side_effect=AssertionError("no training")),
                patch.object(DataloaderConfig, "build", side_effect=AssertionError("no training dataloader")),
            ):
                main()
            rows = next(output.glob("*/subset_token_stats.csv")).read_text()
            packing_reports = list(output.glob("*/packing_token_stats.csv"))
            assert bool(packing_reports) == packing
            if packing:
                current_packing = packing_reports[0].read_text()
                if phase == "cold":
                    expected_packing = current_packing
                else:
                    assert current_packing == expected_packing
            current = {
                path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (tmp_path / "cache").rglob("*.npy")
            }
            if phase == "cold":
                expected, snapshot = rows, current
                assert current
            else:
                assert rows == expected and current == snapshot
            assert loaded["sources"][0]["tokenize_fn"] is original

    @pytest.mark.parametrize("seed,ratio", [(None, 1.5), (17, 1.8), (0, 2.5), (17, 0.5)])
    def test_config_sampling_matches_training_with_cold_and_warm_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int | None, ratio: float
    ) -> None:
        monkeypatch.setenv("XTUNER_TOKENIZE_WORKERS", "1")
        _annotation(tmp_path / "data.jsonl", [4, 12, 100])
        _tokenizer().save_pretrained(tmp_path / "tokenizer")
        effective_seed = 42 if seed is None else seed
        config = tmp_path / "stats_config.py"
        config.write_text(
            ("" if seed is None else f"seed = {seed}\n")
            + f"sample_ratio = {ratio}\n"
            + """
from pathlib import Path
from transformers import AutoTokenizer
from xtuner.v1.datasets import DatasetConfig, FTDPTokenizeFnConfig, build_datasets
from xtuner.v1.datasets.token_stats import TokenStatsConfig

def build_stats_inputs(work_dir):
    base = Path(__file__).parent
    datasets = build_datasets([{
        "dataset": DatasetConfig(name="subset", anno_path=base / "data.jsonl",
            cache_dir=base / "cache", sample_ratio=sample_ratio),
        "tokenize_fn": TokenStatsConfig(FTDPTokenizeFnConfig(chat_template="qwen", max_length=8)),
    }], AutoTokenizer.from_pretrained(base / "tokenizer"))
    return {"datasets": datasets}
"""
        )
        set_random_seed(effective_seed)
        training = _dataset(tmp_path / "data.jsonl", tmp_path / "training_cache", False, sample_ratio=ratio)
        expected_original = []
        with (tmp_path / "data.jsonl").open() as stream:
            for offset in training.offsets:
                stream.seek(offset)
                expected_original.append(len(json.loads(stream.readline())["dialogs"][0]["content"].split()) + 1)
        expected = summarize_samples(
            "subset",
            [
                {
                    "num_tokens": training.num_tokens,
                    "original_num_tokens": np.array(expected_original),
                }
            ],
            "sampled_instances_before_packing",
        )
        cache_snapshot = None
        for phase, ambient_seed in [("cold", 991), ("warm", 117)]:
            output = tmp_path / phase
            set_random_seed(ambient_seed)
            monkeypatch.setattr(sys, "argv", ["token_stats", "--config", str(config), "--output-dir", str(output)])
            guard = (
                patch.object(JsonlDataset, "count_tokens", side_effect=AssertionError("must reuse cache"))
                if phase == "warm"
                else nullcontext()
            )
            with guard:
                main()
            report = next(output.glob("token_stats_*"))
            with (report / "subset_token_stats.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            assert rows == [
                {
                    key: "" if value is None else str(value)
                    for key, value in {**expected, "seed": effective_seed}.items()
                }
            ]
            assert {path.name for path in report.iterdir()} == {"subset_token_stats.csv"}
            snapshot = {
                path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (tmp_path / "cache").rglob("*.npy")
            }
            if phase == "cold":
                cache_snapshot = snapshot
            else:
                assert snapshot == cache_snapshot

    def test_cache_only_report_has_no_sampling_seed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        np.save(tmp_path / "num_tokens.npy", np.array([8]))
        np.save(tmp_path / "original_num_tokens.npy", np.array([12]))
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([{"name": "subset", "meta_dirs": ["."]}]))
        output = tmp_path / "reports"
        monkeypatch.setattr(
            sys, "argv", ["token_stats", "--cache-manifest", str(manifest), "--output-dir", str(output)]
        )
        with patch("mmengine.runner.set_random_seed", side_effect=AssertionError("cache-only mode must not reseed")):
            main()
        with next(output.glob("*/subset_token_stats.csv")).open() as stream:
            row = next(csv.DictReader(stream))
        assert row["seed"] == ""
        assert row["truncated_tokens"] == "4"
        assert row["scope"] == "source_samples_before_filtering"


class TestTokenStatsReports:
    def test_partial_coverage_denominators_and_csv(self, tmp_path: Path) -> None:
        shard = {
            "num_tokens": np.array([5, 8, 8, 8, 0]),
            "original_num_tokens": np.array([5, 8, 12, -1, -1]),
        }
        row = summarize_samples("subset", [shard], "source_samples_before_filtering")
        assert row["count"] == 5 and row["complete_count"] == 3
        assert row["known_original_count"] == 3 and row["unknown_original_count"] == 2
        assert row["original_total_tokens"] == 25
        assert row["original_median_tokens"] == 8
        assert row["original_p99_tokens"] == pytest.approx(11.92)
        assert row["original_max_tokens"] == 12
        assert row["truncated_sample_ratio"] == 1 / 3
        assert row["truncated_token_ratio"] == 4 / 25
        first, message = save_reports([row], tmp_path)
        second, _ = save_reports([row], tmp_path)
        assert first != second
        assert "complete 3/5" in message and "4 tokens" in message
        with (first / "subset_token_stats.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and rows[0]["truncated_tokens"] == "4"
        assert "supervised_tokens_after_collator" not in rows[0]
        assert not (first / "packing_token_stats.csv").exists()

    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("unknown", [False, True])
    @pytest.mark.parametrize("split", [False, True])
    def test_readonly_shards_preserve_global_statistics(self, dtype: Any, unknown: bool, split: bool) -> None:
        original = np.array([100, 1, 7, 0, 12, 8], dtype=dtype)
        kept = np.minimum(original, 8)
        if unknown:
            original[-2] = -1
        original.flags.writeable = kept.flags.writeable = False
        boundaries = (0, 1, 1, 4, 6) if split else (0, 6)
        shards = [
            {"num_tokens": kept[start:end], "original_num_tokens": original[start:end]}
            for start, end in zip(boundaries, boundaries[1:])
        ]
        row = summarize_samples("subset", shards, "source")
        known = original >= 0
        assert row["num_shards"] == len(shards)
        assert row["known_original_count"] == int(known.sum())
        assert row["original_total_tokens"] == int(original[known].sum())
        assert row["original_median_tokens"] == np.median(original[known])
        assert row["original_p99_tokens"] == pytest.approx(np.percentile(original[known], 99))
        assert row["retained_tokens_all_samples"] == int(kept.sum())
        assert row["truncated_tokens"] == int((original[known] - kept[known]).sum())
        np.testing.assert_array_equal(original, [100, 1, 7, 0, -1 if unknown else 12, 8])
        np.testing.assert_array_equal(kept, [8, 1, 7, 0, 8, 8])

    @pytest.mark.parametrize("field", ["num_tokens", "original_num_tokens"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, 1.5, -2])
    def test_invalid_metadata_counts_are_rejected(self, field: str, value: float) -> None:
        shard = {"num_tokens": np.array([5]), "original_num_tokens": np.array([8])}
        shard[field] = np.array([value])
        with pytest.raises(ValueError, match="Invalid"):
            summarize_samples("bad", [shard], "source")

    def test_original_only_old_empty_and_invalid(self) -> None:
        old = summarize_samples("old", [{"num_tokens": np.array([8])}], "source")
        assert old["original_max_tokens"] is None
        assert old["truncated_sample_ratio"] is None
        measured = summarize_samples(
            "measured", [{"num_tokens": np.array([8]), "original_num_tokens": np.array([12])}], "source"
        )
        assert measured["known_original_count"] == measured["complete_count"] == 1
        assert measured["truncated_tokens"] == 4
        assert measured["truncated_sample_ratio"] == 1
        assert measured["truncated_token_ratio"] == 4 / 12
        empty = summarize_samples("empty", [{"num_tokens": np.array([])}], "source")
        assert empty["count"] == 0 and empty["original_mean_tokens"] is None
        with pytest.raises(ValueError, match="Inconsistent"):
            summarize_samples(
                "bad",
                [
                    {
                        "num_tokens": np.array([8]),
                        "original_num_tokens": np.array([7]),
                    }
                ],
                "source",
            )

    @pytest.mark.parametrize("workers", [1, 2])
    def test_distinct_subsets_may_share_cache(self, tmp_path: Path, workers: int) -> None:
        np.save(tmp_path / "num_tokens.npy", np.array([4, 8], dtype=np.int64))
        np.save(tmp_path / "original_num_tokens.npy", np.array([4, 12], dtype=np.int64))
        manifest = [{"name": name, "meta_dirs": [str(tmp_path)]} for name in ("a", "b")]

        rows = summarize_cache_manifest(manifest, workers=workers)

        assert [row["name"] for row in rows] == ["a", "b"]
        for row in rows:
            assert row["scope"] == "source_samples_before_filtering"
            assert row["num_shards"] == 1 and row["count"] == 2
            assert row["original_total_tokens"] == 16
            assert row["retained_tokens_all_samples"] == 12
            assert row["truncated_count"] == 1 and row["truncated_tokens"] == 4

    @pytest.mark.parametrize("workers", [1, 2])
    @pytest.mark.parametrize("duplicate", ["same_entry", "split_entries", "path_alias"])
    def test_duplicate_cache_is_rejected(self, tmp_path: Path, workers: int, duplicate: str) -> None:
        cache = tmp_path / "jsonl_meta"
        cache.mkdir()
        np.save(cache / "num_tokens.npy", np.array([4, 8], dtype=np.int64))
        repeated = cache
        if duplicate == "path_alias":
            repeated = tmp_path / "alias"
            repeated.symlink_to(cache, target_is_directory=True)
        if duplicate == "split_entries":
            manifest = [{"name": "a", "meta_dirs": [str(path)]} for path in (cache, repeated)]
        else:
            manifest = [{"name": "a", "meta_dirs": [str(cache), str(repeated)]}]

        with pytest.raises(ValueError, match="Cache supplied more than once for subset 'a'") as exc:
            summarize_cache_manifest(manifest, workers=workers)
        assert str(cache.resolve()) in str(exc.value)


class _PackingSamples:
    def __init__(self, name: str, lengths: list[int], token_id: int, originals: list[int] | None = None) -> None:
        self.name = name
        self.path = name
        self.num_tokens = np.asarray(lengths, dtype=np.int64)
        self.proxy_attn_flops = self.num_tokens**2
        self._meta = {
            "num_tokens": self.num_tokens,
            "original_num_tokens": np.asarray(originals if originals is not None else lengths, dtype=np.int64),
        }
        self.items = [
            {"num_tokens": count, "input_ids": [token_id] * count, "labels": [-100] + [token_id] * (count - 1)}
            for count in lengths
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class TestPackingTokenStats:
    @pytest.fixture(autouse=True)
    def isolated_mmap(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import xtuner.v1.datasets.utils as dataset_utils

        monkeypatch.setattr(dataset_utils, "_MMAP_DIR", tmp_path)

    @pytest.mark.parametrize("limit", [6, 8, 16])
    @pytest.mark.parametrize("padding", [False, True])
    def test_packed_length_changes_subset_loss_and_shares(self, limit: int, padding: bool) -> None:
        sources = [_PackingSamples("short", [5], 2), _PackingSamples("long", [8], 3, originals=[12])]
        original_items = copy.deepcopy([source.items for source in sources])
        config = DataloaderConfig(pack_level="none", pack_max_length=limit, pack_to_max_length=padding)
        rows = summarize_datasets(sources)
        result = summarize_packing(_build_packed_dataset(sources, config, 42), config, subset_rows=rows)
        by_name = {row["name"]: row for row in rows}
        short, long = by_name["short"], by_name["long"]
        retained = min(8, limit)
        assert long["truncated_tokens"] == 4  # First truncation: 12 -> 8.
        assert long["packing_collator_lost_tokens"] == 8 - retained
        assert long["packing_collator_loss_ratio"] == (8 - retained) / 8
        assert long["token_share_before_packing"] == 8 / 13
        assert long["token_share_after_packing"] == retained / (retained + 5)
        assert long["token_share_change"] == pytest.approx(retained / (retained + 5) - 8 / 13)
        assert long["effective_token_share"] == (retained - 1) / (retained + 3)
        assert short["packing_collator_lost_tokens"] == 0
        assert result["effective_total_tokens"] == retained + 3
        assert result["collator_truncated_tokens"] == 8 - retained
        assert result["packing_collator_loss_ratio"] == (8 - retained) / 13
        assert result["label_shift_tokens"] == 2
        assert result["padding_tokens"] == (limit * 2 - retained - 3 if padding else 0)
        assert result["padding_ratio"] == ((limit * 2 - retained - 3) / (limit * 2) if padding else 0)
        assert result["supervised_tokens_after_collator"] == retained + 3
        assert long["supervised_tokens_after_collator"] == retained - 1
        assert [source.items for source in sources] == original_items

    @pytest.mark.parametrize("counts", [(9, 1, 0), (0, 0, 0)])
    def test_supervision_shares_and_zero_supervision_packs(self, tmp_path: Path, counts: tuple[int, ...]) -> None:
        sources = [_PackingSamples(name, [10], 2) for name in ("dense", "sparse", "zero")]
        for source, count in zip(sources, counts):
            source.items[0]["labels"] = [-100] * (10 - count) + [2] * count
        config = DataloaderConfig(pack_level="none", pack_max_length=10, pack_to_max_length=True)
        rows = summarize_datasets(sources)
        result = summarize_packing(_build_packed_dataset(sources, config, 17), config, subset_rows=rows)
        total = sum(counts)
        for row, count in zip(rows, counts):
            assert row["effective_tokens"] == 9
            assert row["supervised_tokens_after_collator"] == count
            assert row["supervised_token_share"] == (count / total if total else None)
        assert result["supervised_tokens_after_collator"] == total
        assert result["supervision_ratio"] == total / 27
        assert result["zero_supervision_pack_count"] == counts.count(0)
        assert result["padding_ratio"] == 3 / 30
        report, _ = save_reports(rows, tmp_path, result)
        with (report / "packing_token_stats.csv").open() as stream:
            saved = next(csv.DictReader(stream))
        assert saved["supervised_tokens_after_collator"] == str(total)

    @pytest.mark.parametrize("padding", [False, True])
    def test_supervision_follows_target_source_at_shift_boundaries(self, padding: bool) -> None:
        from xtuner.v1.datasets.packing import _LegacySoftPackDataset

        sources = [_PackingSamples("a", [4], 2), _PackingSamples("b", [1], 3)]
        sources[0].items[0]["labels"] = [2] * 4
        sources[1].items[0]["labels"] = [3]
        packed = _LegacySoftPackDataset(sources, pack_max_length=8, global_pack=True, seed=0)
        config = DataloaderConfig(pack_level="__legacy", pack_max_length=8, pack_to_max_length=padding)
        rows = summarize_datasets(sources)
        result = summarize_packing(packed, config, subset_rows=rows)
        assert [row["effective_tokens"] for row in rows] == [4, 0]
        assert [row["supervised_tokens_after_collator"] for row in rows] == [3, 1]
        assert rows[1]["supervised_token_share"] == 1 / 4
        assert result["supervised_tokens_after_collator"] == 4
        assert result["supervision_ratio"] == 1

    @pytest.mark.parametrize("missing_total", [False, True])
    def test_subset_population_mismatch_does_not_modify_rows(self, missing_total: bool) -> None:
        sources = [_PackingSamples("a", [5], 2), _PackingSamples("b", [8], 3)]
        rows = summarize_datasets([sources[0], _PackingSamples("b", [5], 3)])
        if missing_total:
            rows[-1].pop("retained_tokens_all_samples")
        original_rows = copy.deepcopy(rows)
        config = DataloaderConfig(pack_level="none", pack_max_length=6)
        with pytest.raises(ValueError, match="Subset and packing populations do not match"):
            summarize_packing(_build_packed_dataset(sources, config, 17), config, subset_rows=rows)
        assert rows == original_rows

    @pytest.mark.parametrize("level", ["soft", "hard", "__legacy"])
    @pytest.mark.parametrize("global_pack", [False, True])
    @pytest.mark.parametrize("workers", [1, 2])
    def test_subset_effective_tokens_match_real_collator(self, level: str, global_pack: bool, workers: int) -> None:
        sources = [_PackingSamples("a", [5, 14, 9], 2), _PackingSamples("b", [8, 3, 7], 3)]
        config = DataloaderConfig(
            pack_level=level,
            pack_max_length=8,
            global_pack=global_pack,
            pack_workers=workers,
            pack_chunk_size=2,
            pack_extra_buffer_size=2,
        )
        packed = _build_packed_dataset(sources, config, 17)
        rows = summarize_datasets(sources)
        result = summarize_packing(packed, config, subset_rows=rows)
        expected_effective = {"a": 0, "b": 0}
        expected_packed = {"a": 0, "b": 0}
        expected_supervised = {"a": 0, "b": 0}
        for index in range(len(packed)):
            items = copy.deepcopy(packed[index])
            for item in items:
                for name, marker in [("a", 2), ("b", 3)]:
                    expected_packed[name] += item["input_ids"].count(marker)
            batch = sft_llm_collator([items], pack_max_length=8, padding_token_idx=0)[0]
            for name, marker in [("a", 2), ("b", 3)]:
                expected_effective[name] += int((batch["seq_ctx"].input_ids == marker).sum())
                expected_supervised[name] += int((batch["shifted_labels"] == marker).sum())
        for row in rows:
            assert row["effective_tokens"] == expected_effective[row["name"]]
            assert row["packing_retained_tokens"] == expected_packed[row["name"]]
            assert row["supervised_tokens_after_collator"] == expected_supervised[row["name"]]
            assert row["supervised_token_share"] == expected_supervised[row["name"]] / sum(
                expected_supervised.values()
            )
            assert row["packing_input_tokens"] == (
                row["effective_tokens"] + row["packing_collator_lost_tokens"] + row["label_shift_tokens"]
            )
        assert sum(row["effective_tokens"] for row in rows) == result["effective_total_tokens"]
        assert result["supervised_tokens_after_collator"] == sum(expected_supervised.values())
        assert sum(row["packing_collator_lost_tokens"] for row in rows) == result["packing_collator_lost_tokens"]
        assert sum(row["token_share_after_packing"] for row in rows) == pytest.approx(1)

    def test_hard_pack_tail_changes_mixture_without_counting_interior_slices_as_loss(self) -> None:
        sources = [_PackingSamples("a", [10], 2), _PackingSamples("b", [1, 1], 3)]
        config = DataloaderConfig(pack_level="hard", pack_max_length=8, global_pack=True, pack_workers=1)
        rows = summarize_datasets(sources)
        result = summarize_packing(_build_packed_dataset(sources, config, 1), config, subset_rows=rows)
        a, b = rows
        assert a["packing_truncated_instances"] == 1 and a["packing_truncated_tokens"] == 2
        assert b["packing_dropped_instances"] == 2 and b["packing_dropped_tokens"] == 2
        assert a["packing_collator_loss_ratio"] == 0.2
        assert b["packing_collator_loss_ratio"] == 1
        assert a["token_share_before_packing"] == 10 / 12 and a["token_share_after_packing"] == 1
        assert b["token_share_before_packing"] == 2 / 12 and b["token_share_after_packing"] == 0
        assert result["packing_unassigned_tokens"] == 4
        assert result["label_shift_tokens"] == 1
        assert result["collator_truncated_tokens"] == 0

        long = [_PackingSamples("long", [21], 2)]
        rows = summarize_datasets(long)
        result = summarize_packing(_build_packed_dataset(long, config, 1), config, subset_rows=rows)
        assert result["count"] == 2
        assert rows[0]["packing_truncated_instances"] == 1
        assert rows[0]["packing_truncated_tokens"] == 5
        assert rows[0]["effective_tokens"] == 14

    def test_whole_sequence_drop_and_same_name_shards(self) -> None:
        from xtuner.v1.datasets.packing import _LegacySoftPackDataset

        sources = [_PackingSamples("a", [4], 2), _PackingSamples("b", [6], 3)]
        packed = _LegacySoftPackDataset(sources, pack_max_length=16, global_pack=True, seed=0)
        config = DataloaderConfig(pack_level="__legacy", pack_max_length=8)
        rows = summarize_datasets(sources)
        result = summarize_packing(packed, config, subset_rows=rows)
        assert result["collator_dropped_tokens"] == 6
        assert result["collator_truncated_tokens"] == 0
        assert rows[0]["effective_tokens"] == 3
        assert rows[1]["collator_dropped_sequences"] == 1
        assert rows[1]["token_share_after_packing"] == 0
        assert [row["supervised_tokens_after_collator"] for row in rows] == [3, 0]
        sources[1].name = "a"
        rows = summarize_datasets(sources)
        summarize_packing(packed, config, subset_rows=rows)
        assert len(rows) == 1
        assert rows[0]["packing_input_tokens"] == 10
        assert rows[0]["packing_collator_loss_ratio"] == 0.6
        assert rows[0]["token_share_change"] == 0
        assert rows[0]["supervised_tokens_after_collator"] == 3
        assert rows[0]["supervised_token_share"] == 1

    def test_empty_distribution_and_report_export(self, tmp_path: Path) -> None:
        sources = [_PackingSamples("empty", [], 2)]
        config = DataloaderConfig(pack_level="none", pack_max_length=8)
        rows = summarize_datasets(sources)
        result = summarize_packing(_build_packed_dataset(sources, config, 42), config, subset_rows=rows)
        assert result["count"] == 0 and result["effective_mean_tokens"] is None
        assert rows[0]["packing_collator_loss_ratio"] is None
        assert rows[0]["token_share_change"] is None
        assert rows[0]["supervised_tokens_after_collator"] == 0
        assert rows[0]["supervised_token_share"] is None
        assert result["supervised_tokens_after_collator"] == 0
        assert result["supervision_ratio"] is None
        assert result["zero_supervision_pack_count"] == 0
        assert result["padding_ratio"] is None
        report, message = save_reports(rows, tmp_path, result, seed=42)
        assert "Packing/collator lost 0 tokens" in message
        with (report / "packing_token_stats.csv").open() as stream:
            saved = next(csv.DictReader(stream))
        assert saved["seed"] == "42"
        assert saved["packing_collator_loss_ratio"] == ""
        assert saved["supervision_ratio"] == ""
        assert saved["padding_ratio"] == ""

    @pytest.mark.parametrize(
        "sp_size,padding,factory,expected_padding",
        [(1, False, False, 0), (1, True, False, 4), (2, False, False, 4), (2, True, False, 4), (2, False, True, 0)],
    )
    def test_config_packing_uses_effective_padding(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sp_size: int,
        padding: bool,
        factory: bool,
        expected_padding: int,
    ) -> None:
        from xtuner.v1.utils import Config

        sources = [_PackingSamples("subset", [5], 2)]
        loader = DataloaderConfig(pack_level="none", pack_max_length=8, pack_to_max_length=padding)
        trainer = SimpleNamespace(dataloader_cfg=loader, sp_size=sp_size, seed=17)
        config = {"trainer": trainer}
        if factory:
            config["build_stats_inputs"] = lambda work_dir: {"datasets": sources, "dataloader_config": loader}
        monkeypatch.setattr(
            sys, "argv", ["token_stats", "--config", "train.py", "--packing", "--output-dir", str(tmp_path)]
        )
        with (
            patch.object(Config, "fromfile", return_value=config),
            patch("xtuner.tools.token_stats._build_training_datasets", return_value=sources),
        ):
            main()

        with next(tmp_path.glob("*/packing_token_stats.csv")).open() as stream:
            row = next(csv.DictReader(stream))
        assert int(row["padding_tokens"]) == expected_padding
        assert int(row["effective_total_tokens"]) == 4
        assert int(row["label_shift_tokens"]) == 1
        assert int(row["packing_collator_lost_tokens"]) == 0
        assert trainer.dataloader_cfg is loader
        assert loader.pack_to_max_length is padding

    def test_cache_only_cannot_claim_packing_statistics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys, "argv", ["token_stats", "--cache-manifest", "unused.json", "--packing", "--output-dir", "."]
        )
        with pytest.raises(SystemExit, match="2"):
            main()

    @pytest.mark.parametrize("mismatch", [None, "length", "instance", "repeated"])
    def test_factory_existing_packs_require_matching_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str | None
    ) -> None:
        from xtuner.v1.utils import Config

        sources = [_PackingSamples("subset", [5, 8], 2)]
        packed_sources = sources
        if mismatch == "length":
            packed_sources = [_PackingSamples("subset", [5, 5], 2)]
        elif mismatch == "instance":
            packed_sources = copy.deepcopy(sources)
        elif mismatch == "repeated":
            packed_sources = sources * 2
        loader = DataloaderConfig(pack_level="none", pack_max_length=6)
        inputs = {
            "datasets": sources,
            "packed_dataset": _build_packed_dataset(packed_sources, loader, 17),
            "dataloader_config": loader,
        }
        monkeypatch.setattr(sys, "argv", ["token_stats", "--config", "custom.py", "--output-dir", str(tmp_path)])
        with (
            patch.object(Config, "fromfile", return_value={"build_stats_inputs": lambda work_dir: inputs, "seed": 17}),
            pytest.raises(ValueError, match="datasets and packed_dataset must share sampled sources")
            if mismatch is not None
            else nullcontext(),
        ):
            main()
        if mismatch is not None:
            assert not list(tmp_path.rglob("*.csv"))
            return
        with next(tmp_path.glob("*/packing_token_stats.csv")).open() as stream:
            row = next(csv.DictReader(stream))
        assert row["collator_truncated_tokens"] == "2"
        assert row["seed"] == "17"
        with next(tmp_path.glob("*/subset_token_stats.csv")).open() as stream:
            row = next(csv.DictReader(stream))
        assert float(row["packing_collator_loss_ratio"]) == 2 / 13


def _distributed_cache_worker(rank: int, directory: str, path: str) -> None:
    import torch.distributed as dist

    os.environ["LOCAL_WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["XTUNER_TOKENIZE_WORKERS"] = "1"
    base = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{base / 'rendezvous'}", rank=rank, world_size=2)
    try:
        dataset = _dataset(Path(path), base / "cache", True)
        assert dataset._meta["original_num_tokens"].tolist() == [5]
        assert dataset._meta["original_num_tokens"].dtype == np.dtype("int64")
        assert "truncated_num_tokens" not in dataset._meta
        if rank == 0:
            save_reports(summarize_datasets([dataset]), base / "reports")
        dist.barrier()
    finally:
        dist.destroy_process_group()
