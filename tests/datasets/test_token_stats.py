"""Tests for token length recording and statistics."""

import copy
import csv
import json
import logging
import os
import random
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
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


@pytest.fixture(autouse=True)
def _single_process_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # DistributedTestBase leaves WORLD_SIZE in the parent process after earlier tests.
    monkeypatch.setenv("WORLD_SIZE", "1")


@pytest.fixture(autouse=True)
def _restore_sampling_rng() -> Iterator[None]:
    # Sampling comparisons must not leave seeds behind for later training tests.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def _set_sampling_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def _stats_inputs() -> dict[str, Any]:
    source = SimpleNamespace(
        name="subset", path="unused.jsonl", _meta={"num_tokens": np.array([2]), "original_num_tokens": np.array([3])}
    )
    return {"datasets": [source]}


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
    @pytest.mark.parametrize("fail", [False, True])
    def test_config_restores_sampling_rng(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
        from xtuner.v1.utils import Config

        def build_stats_inputs(work_dir):
            assert random.random() == random.Random(17).random()
            assert np.random.random() == np.random.RandomState(17).random()
            torch.testing.assert_close(torch.rand(3), torch.rand(3, generator=torch.Generator().manual_seed(17)))
            if fail:
                raise RuntimeError("factory failed")
            return _stats_inputs()

        def load_config(path):
            # Config loading can also consume randomness before the sampling seed is applied.
            random.random()
            np.random.random()
            torch.rand(3)
            return {"seed": 17, "build_stats_inputs": build_stats_inputs}

        monkeypatch.setattr(Config, "fromfile", load_config)
        monkeypatch.setattr(sys, "argv", ["token_stats", "--config", "unused.py", "--output-dir", str(tmp_path)])
        _set_sampling_seed(991)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with pytest.raises(RuntimeError, match="factory failed") if fail else nullcontext():
            main()
        assert random.getstate() == python_state
        current_numpy_state = np.random.get_state()
        np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
        assert current_numpy_state[0] == numpy_state[0] and current_numpy_state[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state)

    @pytest.mark.parametrize("source", ["config", "cache-manifest"])
    @pytest.mark.parametrize("fail", [False, True])
    def test_cli_preserves_logging_and_environment(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        source: str,
        fail: bool,
    ) -> None:
        from xtuner.v1.utils import Config, get_logger

        def build_stats_inputs(work_dir):
            if fail:
                raise FileNotFoundError("factory input missing")
            return _stats_inputs()

        monkeypatch.setattr(Config, "fromfile", lambda path: {"seed": 17, "build_stats_inputs": build_stats_inputs})
        path = tmp_path / "input.json"
        if not fail:
            np.save(tmp_path / "num_tokens.npy", np.array([2]))
            np.save(tmp_path / "original_num_tokens.npy", np.array([3]))
            path.write_text(json.dumps([{"name": "subset", "meta_dirs": ["."]}]))
        monkeypatch.setattr(sys, "argv", ["token_stats", f"--{source}", str(path), "--output-dir", str(tmp_path)])
        monkeypatch.delenv("TQDM_DISABLE", raising=False)
        log = get_logger()
        records = []
        sink_id = log.add(lambda message: records.append(str(message)), format="{message}")
        root_logger = logging.getLogger()
        monkeypatch.setattr(root_logger, "handlers", [])
        monkeypatch.setattr(root_logger, "level", root_logger.level)
        handlers, level = list(root_logger.handlers), root_logger.level
        environment = dict(os.environ)
        try:
            with pytest.raises(FileNotFoundError) if fail else nullcontext():
                main()
            log.info("training log still visible")
            assert any("training log still visible" in record for record in records)
            assert dict(os.environ) == environment
            assert root_logger.handlers == handlers and root_logger.level == level
            if not fail:
                assert "Token statistics:" in capsys.readouterr().err
        finally:
            try:
                log.remove(sink_id)
            except ValueError:
                pass  # Old implementations removed the caller's sink; keep the assertion failure visible.

    def test_config_does_not_seed_accelerators(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from xtuner.v1.utils import Config

        monkeypatch.setattr(
            Config, "fromfile", lambda path: {"seed": 17, "build_stats_inputs": lambda _: _stats_inputs()}
        )
        monkeypatch.setattr(sys, "argv", ["token_stats", "--config", "unused.py", "--output-dir", str(tmp_path)])
        with (
            patch.object(
                torch.cuda, "manual_seed", side_effect=AssertionError("offline statistics must not seed CUDA")
            ),
            patch.object(
                torch.cuda, "manual_seed_all", side_effect=AssertionError("offline statistics must not seed CUDA")
            ),
        ):
            main()

    @pytest.mark.parametrize("source", ["--config", "--cache-manifest"])
    @pytest.mark.parametrize("world_size", [2, 8])
    def test_cli_rejects_distributed_launch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        source: str,
        world_size: int,
    ) -> None:
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setattr(sys, "argv", ["token_stats", source, "unused.json", "--output-dir", str(tmp_path)])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        assert "Run this offline coordinator once, without torchrun" in capsys.readouterr().err
        assert not list(tmp_path.iterdir())

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
        _set_sampling_seed(effective_seed)
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
            _set_sampling_seed(ambient_seed)
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
                    for key, value in {**expected, "num_shards": 1, "seed": effective_seed}.items()
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
    @pytest.mark.parametrize("chunk_size", [1, 3, 65_536])
    def test_source_counts_and_metrics_are_independent_of_chunk_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk_size: int
    ) -> None:
        import xtuner.tools.token_stats as token_stats

        metadata = [
            {"num_tokens": np.array([0, 2, 8, 8, 8]), "original_num_tokens": np.array([-1, 2, 8, 12, 10**12])},
            {"num_tokens": np.array([3, 4])},
            {"num_tokens": np.array([], dtype=np.int64), "original_num_tokens": np.array([], dtype=np.int64)},
        ]
        sources = []
        directories = []
        for index, shard in enumerate(metadata):
            directory = tmp_path / str(index)
            directory.mkdir()
            for key, values in shard.items():
                values.flags.writeable = False
                np.save(directory / f"{key}.npy", values)
            sources.append(SimpleNamespace(name="subset", path=str(index), _meta=shard))
            directories.append(str(directory))
        expected = summarize_datasets(sources)[0]
        assert expected["num_shards"] == 3 and expected["count"] == 7
        assert expected["known_original_count"] == 4 and expected["unknown_original_count"] == 3
        monkeypatch.setattr(token_stats, "_STATS_CHUNK_SIZE", chunk_size)
        assert summarize_datasets(sources) == [expected]
        assert summarize_cache_manifest([{"name": "subset", "meta_dirs": directories}]) == [
            {**expected, "scope": "source_samples_before_filtering"}
        ]

    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("case", ["mixed", "boundary", "sparse"])
    def test_dense_and_sparse_length_counts_match_numpy(self, dtype: Any, case: str) -> None:
        from xtuner.tools.token_stats import _DENSE_LENGTH_LIMIT

        limit = _DENSE_LENGTH_LIMIT
        lengths = {
            "mixed": [0, 1, limit - 1, limit, 10**12, 1, limit - 1, limit + 1, limit, 4],
            "boundary": [limit - 1, limit],
            "sparse": [limit, 10**12, limit + 1, limit],
        }[case]
        original = np.asarray(lengths, dtype=dtype)
        kept = np.minimum(original, 8)
        original.flags.writeable = kept.flags.writeable = False
        shards = [
            {"num_tokens": k, "original_num_tokens": o}
            for k, o in zip(np.array_split(kept, 3), np.array_split(original, 3))
        ]
        row = summarize_samples("subset", shards, "source")
        losses = original - kept
        assert row["count"] == row["known_original_count"] == len(original)
        assert row["original_total_tokens"] == int(original.sum())
        assert row["original_mean_tokens"] == original.mean()
        assert row["original_median_tokens"] == np.median(original)
        assert row["original_p99_tokens"] == np.percentile(original, 99)
        assert row["original_max_tokens"] == int(original.max())
        assert row["retained_tokens_all_samples"] == int(kept.sum())
        assert row["truncated_count"] == int((losses > 0).sum())
        assert row["truncated_tokens"] == int(losses.sum())
        np.testing.assert_array_equal(original, lengths)
        np.testing.assert_array_equal(kept, np.minimum(lengths, 8))

    @pytest.mark.parametrize("dtype", [np.float64, ">i8"])
    @pytest.mark.parametrize("version", [(1, 0), (2, 0)])
    def test_cache_reader_preserves_numeric_dtype_and_npy_headers(
        self, tmp_path: Path, dtype: Any, version: tuple[int, int]
    ) -> None:
        shard = {
            "num_tokens": np.array([0, 1, 8, 8], dtype=dtype),
            "original_num_tokens": np.array([-1, 1, 8, 11], dtype=dtype),
        }
        for key, values in shard.items():
            with (tmp_path / f"{key}.npy").open("wb") as stream:
                np.lib.format.write_array(stream, values, version=version)
        manifest = [{"name": "subset", "meta_dirs": [str(tmp_path)]}]
        assert summarize_cache_manifest(manifest) == [
            {**summarize_samples("subset", [shard], "source_samples_before_filtering"), "num_shards": 1}
        ]

    @pytest.mark.parametrize("shape", [(2,), (3, 1), ()])
    @pytest.mark.parametrize("source", ["dataset", "cache"])
    def test_readers_reject_invalid_metadata_shapes(self, tmp_path: Path, shape: tuple[int, ...], source: str) -> None:
        shard = {"num_tokens": np.ones(3, dtype=np.int64), "original_num_tokens": np.ones(shape, dtype=np.int64)}
        for key, values in shard.items():
            np.save(tmp_path / f"{key}.npy", values)
        with pytest.raises(ValueError, match="Invalid"):
            if source == "dataset":
                summarize_datasets([SimpleNamespace(name="subset", path="unused.jsonl", _meta=shard)])
            else:
                summarize_cache_manifest([{"name": "subset", "meta_dirs": [str(tmp_path)]}])

    @pytest.mark.parametrize("workers", [1, 2])
    def test_multiple_caches_match_numpy_with_exact_quantiles(self, tmp_path: Path, workers: int) -> None:
        rng = np.random.default_rng(17)
        original = rng.integers(0, 1000, size=200_003, dtype=np.int64)
        original[1] = 10**12  # Sparse outliers must not require an array indexed by maximum length.
        kept = np.minimum(original, 128)
        original[::11] = -1
        boundaries = (0, 0, 70_001, 100_001, len(original))
        directories = []
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            directory = tmp_path / str(index)
            directory.mkdir()
            directories.append(str(directory))
            np.save(directory / "num_tokens.npy", kept[start:end])
            # Also cover a legacy shard with no original-length metadata.
            if index != 2:
                np.save(directory / "original_num_tokens.npy", original[start:end])
            else:
                original[start:end] = -1
        row = summarize_cache_manifest([{"name": "subset", "meta_dirs": directories}], workers=workers)[0]
        known = original >= 0
        losses = original[known] - kept[known]
        assert row["count"] == len(original)
        assert row["num_shards"] == 4
        assert row["damaged_or_empty_count"] == int((kept == 0).sum())
        assert row["known_original_count"] == int(known.sum())
        assert row["unknown_original_count"] == int((~known).sum())
        assert row["retained_tokens_all_samples"] == int(kept.sum())
        assert row["original_total_tokens"] == int(original[known].sum())
        assert row["original_mean_tokens"] == pytest.approx(original[known].mean())
        assert row["original_median_tokens"] == np.median(original[known])
        assert row["original_p99_tokens"] == pytest.approx(np.percentile(original[known], 99))
        assert row["original_max_tokens"] == int(original[known].max())
        assert row["truncated_count"] == int((losses > 0).sum())
        assert row["truncated_tokens"] == int(losses.sum())
        assert row["truncated_sample_ratio"] == (losses > 0).mean()
        assert row["truncated_token_ratio"] == losses.sum() / original[known].sum()

    @pytest.mark.parametrize("lengths", [[], [0], [9], [0, 100], [1, 1, 1, 100], list(range(101))])
    def test_streamed_shards_preserve_quantile_interpolation(self, lengths: list[int]) -> None:
        values = np.asarray(lengths, dtype=np.int64)
        shards = ({"num_tokens": part, "original_num_tokens": part} for part in np.array_split(values, 4))
        row = summarize_samples("subset", shards, "source")
        assert row["count"] == len(lengths)
        for metric, percentile in [("median", 50), ("p99", 99)]:
            expected = np.percentile(values, percentile) if lengths else None
            assert row[f"original_{metric}_tokens"] == expected

    def test_partial_coverage_denominators_and_csv(self, tmp_path: Path) -> None:
        shard = {
            "num_tokens": np.array([5, 8, 8, 8, 0]),
            "original_num_tokens": np.array([5, 8, 12, -1, -1]),
        }
        row = summarize_samples("subset", [shard], "source_samples_before_filtering")
        assert row["count"] == 5
        assert row["known_original_count"] == 3 and row["unknown_original_count"] == 2
        assert row["original_total_tokens"] == 25
        assert row["original_median_tokens"] == 8
        assert row["original_p99_tokens"] == pytest.approx(11.92)
        assert row["original_max_tokens"] == 12
        assert row["truncated_sample_ratio"] == 1 / 3
        assert row["truncated_token_ratio"] == 4 / 25
        row["seed"] = 17
        before = dict(row)
        first, message = save_reports([row], tmp_path)
        second, _ = save_reports([row], tmp_path)
        assert first != second
        assert "original length known 3/5" in message and "4 tokens" in message
        with (first / "subset_token_stats.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and rows[0]["truncated_tokens"] == "4"
        assert rows[0]["seed"] == "17" and row == before
        assert {"complete_count", "incomplete_count", "complete_original_tokens"}.isdisjoint(rows[0])
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
        assert measured["known_original_count"] == 1
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

    @pytest.mark.parametrize("actual_level", ["none", "soft", "hard", "__legacy"])
    @pytest.mark.parametrize("declared_level", ["none", "soft", "hard", "__legacy"])
    def test_existing_pack_mode_must_match_config(self, actual_level: str, declared_level: str) -> None:
        sources = [_PackingSamples("subset", [5, 8, 13], 2)]
        config = DataloaderConfig(pack_level=actual_level, pack_max_length=8, pack_workers=1)
        packed = _build_packed_dataset(sources, config, 17)
        declared = config.model_copy(update={"pack_level": declared_level})
        if actual_level != declared_level:
            with pytest.raises(ValueError, match="pack_level"):
                summarize_packing(packed, declared)
        else:
            result, _ = summarize_packing(packed, declared)
            assert result["pack_level"] == actual_level

    @pytest.mark.parametrize("packing", [False, True])
    def test_factory_packing_requires_explicit_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packing: bool
    ) -> None:
        from xtuner.v1.utils import Config

        sources = [_PackingSamples("subset", [5, 8, 13], 2)]
        loader = DataloaderConfig(pack_level="hard", pack_max_length=8, pack_workers=1)
        inputs = {"datasets": sources, "dataloader_config": loader}
        monkeypatch.setattr(Config, "fromfile", lambda _: {"seed": 42, "build_stats_inputs": lambda work_dir: inputs})
        arguments = ["token_stats", "--config", "unused.py", "--output-dir", str(tmp_path)]
        monkeypatch.setattr(sys, "argv", arguments + (["--packing"] if packing else []))
        guard = (
            nullcontext()
            if packing
            else patch(
                "xtuner.tools.token_stats._build_packed_dataset", side_effect=AssertionError("packing disabled")
            )
        )
        with guard:
            main()
        reports = list(tmp_path.glob("*/packing_token_stats.csv"))
        assert bool(reports) == packing
        if packing:
            with reports[0].open() as stream:
                row = next(csv.DictReader(stream))
            assert row["pack_level"] == "hard"
            assert row["seed"] == "42"
        with next(tmp_path.glob("*/subset_token_stats.csv")).open() as stream:
            row = next(csv.DictReader(stream))
        assert row["seed"] == "42"
        assert ("effective_tokens" in row) == packing

    @pytest.mark.parametrize("limit", [6, 8, 16])
    @pytest.mark.parametrize("padding", [False, True])
    def test_packed_length_changes_subset_loss_and_shares(self, limit: int, padding: bool) -> None:
        sources = [_PackingSamples("short", [5], 2), _PackingSamples("long", [8], 3, originals=[12])]
        original_items = copy.deepcopy([source.items for source in sources])
        config = DataloaderConfig(pack_level="none", pack_max_length=limit, pack_to_max_length=padding)
        result, rows = summarize_packing(_build_packed_dataset(sources, config, 42), config)
        by_name = {row["name"]: row for row in rows}
        short, long = by_name["short"], by_name["long"]
        retained = min(8, limit)
        sample_rows = {row["name"]: row for row in summarize_datasets(sources)}
        assert sample_rows["long"]["truncated_tokens"] == 4  # First truncation: 12 -> 8.
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

    @pytest.mark.parametrize("padding", [False, True])
    def test_global_losses_include_collator_truncation_and_drops(self, tmp_path: Path, padding: bool) -> None:
        from xtuner.v1.datasets.packing import _LegacySoftPackDataset

        sources = [_PackingSamples("a", [4], 2), _PackingSamples("b", [6], 3)]
        packed = _LegacySoftPackDataset(sources, pack_max_length=16, global_pack=True, seed=0)
        config = DataloaderConfig(pack_level="__legacy", pack_max_length=3, pack_to_max_length=padding)
        result, rows = summarize_packing(packed, config)

        # The first sequence loses one token, the second is dropped; label shift is separate.
        assert result["sampled_input_tokens"] == result["tokens_before_collation"] == 10
        assert result["packing_unassigned_tokens"] == 0
        assert result["collator_truncated_tokens"] == 1
        assert result["collator_dropped_tokens"] == 6
        assert result["packing_collator_lost_tokens"] == 7
        assert result["packing_collator_loss_ratio"] == 0.7
        assert result["label_shift_tokens"] == 1
        assert result["effective_total_tokens"] == 2
        assert result["padding_tokens"] == int(padding)
        assert [row["packing_collator_lost_tokens"] for row in rows] == [1, 6]
        output, message = save_reports(summarize_datasets(sources), tmp_path, result)
        assert "Packing/collator lost 7 tokens" in message
        with (output / "packing_token_stats.csv").open() as stream:
            saved = next(csv.DictReader(stream))
        assert saved["packing_collator_lost_tokens"] == "7"
        assert saved["packing_collator_loss_ratio"] == "0.7"

    @pytest.mark.parametrize("counts", [(9, 1, 0), (0, 0, 0)])
    def test_supervision_shares_and_zero_supervision_packs(self, tmp_path: Path, counts: tuple[int, ...]) -> None:
        sources = [_PackingSamples(name, [10], 2) for name in ("dense", "sparse", "zero")]
        for source, count in zip(sources, counts):
            source.items[0]["labels"] = [-100] * (10 - count) + [2] * count
        config = DataloaderConfig(pack_level="none", pack_max_length=10, pack_to_max_length=True)
        result, rows = summarize_packing(_build_packed_dataset(sources, config, 17), config)
        total = sum(counts)
        for row, count in zip(rows, counts):
            assert row["effective_tokens"] == 9
            assert row["supervised_tokens_after_collator"] == count
            assert row["supervised_token_share"] == (count / total if total else None)
        assert result["supervised_tokens_after_collator"] == total
        assert result["supervision_ratio"] == total / 27
        assert result["zero_supervision_pack_count"] == counts.count(0)
        assert result["padding_ratio"] == 3 / 30
        report, _ = save_reports(summarize_datasets(sources), tmp_path, result)
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
        result, rows = summarize_packing(packed, config)
        assert [row["effective_tokens"] for row in rows] == [4, 0]
        assert [row["supervised_tokens_after_collator"] for row in rows] == [3, 1]
        assert rows[1]["supervised_token_share"] == 1 / 4
        assert result["supervised_tokens_after_collator"] == 4
        assert result["supervision_ratio"] == 1

    def test_packing_returns_independent_subset_results(self) -> None:
        sources = [_PackingSamples("a", [5], 2), _PackingSamples("b", [8], 3)]
        sample_rows = summarize_datasets(sources)
        original_rows = copy.deepcopy(sample_rows)
        original_items = copy.deepcopy([source.items for source in sources])
        config = DataloaderConfig(pack_level="none", pack_max_length=6)
        packed = _build_packed_dataset(sources, config, 17)
        result, rows = summarize_packing(packed, config)
        expected = copy.deepcopy((result, rows))
        result["effective_total_tokens"] = -1
        rows[0]["effective_tokens"] = -1
        assert summarize_packing(packed, config) == expected
        assert sample_rows == original_rows
        assert [source.items for source in sources] == original_items

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
        result, rows = summarize_packing(packed, config)
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
        result, rows = summarize_packing(_build_packed_dataset(sources, config, 1), config)
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
        result, rows = summarize_packing(_build_packed_dataset(long, config, 1), config)
        assert result["count"] == 2
        assert rows[0]["packing_truncated_instances"] == 1
        assert rows[0]["packing_truncated_tokens"] == 5
        assert rows[0]["effective_tokens"] == 14

    def test_whole_sequence_drop_and_same_name_shards(self) -> None:
        from xtuner.v1.datasets.packing import _LegacySoftPackDataset

        sources = [_PackingSamples("a", [4], 2), _PackingSamples("b", [6], 3)]
        packed = _LegacySoftPackDataset(sources, pack_max_length=16, global_pack=True, seed=0)
        config = DataloaderConfig(pack_level="__legacy", pack_max_length=8)
        result, rows = summarize_packing(packed, config)
        assert result["collator_dropped_tokens"] == 6
        assert result["collator_truncated_tokens"] == 0
        assert rows[0]["effective_tokens"] == 3
        assert rows[1]["collator_dropped_sequences"] == 1
        assert rows[1]["token_share_after_packing"] == 0
        assert [row["supervised_tokens_after_collator"] for row in rows] == [3, 0]
        sources[1].name = "a"
        _, rows = summarize_packing(packed, config)
        assert len(rows) == 1
        assert rows[0]["packing_input_tokens"] == 10
        assert rows[0]["packing_collator_loss_ratio"] == 0.6
        assert rows[0]["token_share_change"] == 0
        assert rows[0]["supervised_tokens_after_collator"] == 3
        assert rows[0]["supervised_token_share"] == 1

    def test_empty_distribution_and_report_export(self, tmp_path: Path) -> None:
        sources = [_PackingSamples("empty", [], 2)]
        config = DataloaderConfig(pack_level="none", pack_max_length=8)
        result, rows = summarize_packing(_build_packed_dataset(sources, config, 42), config)
        assert result["count"] == 0 and result["effective_mean_tokens"] is None
        assert rows[0]["packing_collator_loss_ratio"] is None
        assert rows[0]["token_share_change"] is None
        assert rows[0]["supervised_tokens_after_collator"] == 0
        assert rows[0]["supervised_token_share"] is None
        assert result["supervised_tokens_after_collator"] == 0
        assert result["supervision_ratio"] is None
        assert result["zero_supervision_pack_count"] == 0
        assert result["padding_ratio"] is None
        result["seed"] = 42
        report, message = save_reports(summarize_datasets(sources), tmp_path, result)
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

    def test_cache_only_cannot_claim_packing_statistics(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["token_stats", "--cache-manifest", "unused.json", "--packing", "--output-dir", "."]
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        assert "--packing requires --config" in capsys.readouterr().err

    @pytest.mark.parametrize("packing", [False, True])
    def test_factory_rejects_prebuilt_packs_with_migration_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], packing: bool
    ) -> None:
        from xtuner.v1.utils import Config

        inputs = {**_stats_inputs(), "packed_dataset": object()}
        monkeypatch.setattr(Config, "fromfile", lambda _: {"build_stats_inputs": lambda work_dir: inputs, "seed": 17})
        arguments = ["token_stats", "--config", "custom.py", "--output-dir", str(tmp_path)]
        monkeypatch.setattr(sys, "argv", arguments + (["--packing"] if packing else []))
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        assert "remove packed_dataset and use --packing" in capsys.readouterr().err
        assert not list(tmp_path.rglob("*.csv"))


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
