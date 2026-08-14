import json
from types import SimpleNamespace

import torch

from xtuner.v1.datasets.tokenize_debug import (
    _build_record,
    _sample_indices,
    maybe_dump_tokenize_debug_samples,
)


class _FakeTokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [f"token_{token_id}" for token_id in token_ids]

    def decode(self, token_ids, **kwargs):
        return "|".join(str(token_id) for token_id in token_ids)


class _FakeDataset:
    media_root = ""

    def __init__(self, path):
        self.path = str(path)
        self.offsets = [0]

    def __len__(self):
        return 1


class _FakeTokenizeFunction:
    chat_template_name = "qwen3.5-vl"
    chat_template = SimpleNamespace(default_system=None)
    add_vision_id = False
    tokenizer = _FakeTokenizer()
    state = "runtime"

    def set_state(self, state):
        self.state = state

    def __call__(self, raw_data, **kwargs):
        return {
            "input_ids": [10, 11, 12],
            "labels": [-100, 11, 12],
            "num_tokens": 3,
        }


class _FakeQwen36TokenizeFunction(_FakeTokenizeFunction):
    chat_template_name = "qwen3.6-vl"


def test_build_record_reverse_decodes_contiguous_loss_spans():
    record = _build_record(
        tokenizer=_FakeTokenizer(),
        raw_data={"messages": [{"role": "assistant", "content": "AB"}]},
        rendered_text="promptAB",
        character_loss_mask=[False] * 6 + [True, True],
        tokenized_data={
            "input_ids": [10, 11, 12, 13, 14],
            "labels": [-100, 11, 12, -100, 14],
            "position_ids": torch.zeros(3, 5),
            "num_tokens": 5,
        },
        dataset_name="demo",
        dataset_path="/tmp/demo.jsonl",
        sample_index=0,
    )

    assert record["rendered"]["loss_character_spans"] == [
        {"start": 6, "end": 8, "text": "AB"}
    ]
    assert record["loss"]["token_count"] == 3
    assert record["loss"]["token_spans"] == [
        {
            "start": 1,
            "end": 3,
            "token_ids": [11, 12],
            "token_pieces": ["token_11", "token_12"],
            "decoded_text": "11|12",
        },
        {
            "start": 4,
            "end": 5,
            "token_ids": [14],
            "token_pieces": ["token_14"],
            "decoded_text": "14",
        },
    ]
    assert record["tokenized"]["other_fields"]["position_ids"] == {
        "kind": "torch.Tensor",
        "shape": [3, 5],
        "dtype": "torch.float32",
    }


def test_sample_indices_are_seeded_unique_and_reproducible():
    first = _sample_indices(dataset_size=100, sample_count=10)
    second = _sample_indices(dataset_size=100, sample_count=10)

    assert first == second
    assert first != list(range(10))
    assert len(first) == len(set(first)) == 10
    assert first == sorted(first)


def test_single_environment_variable_dumps_one_sample(monkeypatch, tmp_path):
    data_path = tmp_path / "demo.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setenv("XTUNER_TOKENIZE_DEBUG_SAMPLES", "1")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work_dir"))

    maybe_dump_tokenize_debug_samples(
        dataset=_FakeDataset(data_path),
        tokenize_fn=_FakeTokenizeFunction(),
        dataset_name="demo",
    )

    sample_files = list((tmp_path / "work_dir" / "tokenize_debug_samples").rglob("sample_000000.json"))
    assert len(sample_files) == 1
    record = json.loads(sample_files[0].read_text())
    assert record["raw_data"]["messages"][0]["content"] == "question"
    assert record["rendered"]["text"].startswith("<|im_start|>user\nquestion")
    assert record["loss"]["token_spans"][0]["decoded_text"] == "11|12"

    manifest_files = list((tmp_path / "work_dir" / "tokenize_debug_samples").rglob("manifest.json"))
    assert len(manifest_files) == 1
    manifest = json.loads(manifest_files[0].read_text())
    assert manifest["sampling"] == "random_without_replacement"
    assert manifest["random_seed"] == 42
    assert manifest["sample_indices"] == [0]


def test_qwen36_debug_uses_template_specific_output_directory(monkeypatch, tmp_path):
    data_path = tmp_path / "demo.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setenv("XTUNER_TOKENIZE_DEBUG_SAMPLES", "1")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work_dir"))

    maybe_dump_tokenize_debug_samples(
        dataset=_FakeDataset(data_path),
        tokenize_fn=_FakeQwen36TokenizeFunction(),
        dataset_name="demo",
    )

    manifest_path = next(
        (tmp_path / "work_dir" / "tokenize_debug_samples" / "qwen3.6-vl").rglob("manifest.json")
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["chat_template_name"] == "qwen3.6-vl"
