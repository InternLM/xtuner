"""Dataset error handling must respect SFT and RL tokenizer contracts."""

import json
from unittest.mock import Mock

import pytest

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.datasets.rl_tokenize_fn.glm53_vl_tokenize_fn import RLGlm53VLTokenizeFunction
from xtuner.v1.datasets.rl_tokenize_fn.qwen3_vl_tokenize_fn import RLQwen3VLTokenizeFunction
from xtuner.v1.datasets.rl_tokenize_fn.text_tokenize_fn import RLTextTokenizeFn
from xtuner.v1.datasets.utils import CachableTokenizeFunction
from xtuner.v1.datasets.vlm_jsonl import VLMJsonlDataset


def make_dataset(tmp_path, tokenize_fn, line=None):
    path = tmp_path / "samples.jsonl"
    path.write_text(line if line is not None else json.dumps({"prompt": [], "data_source": "vl"}) + "\n")
    dataset = VLMJsonlDataset.__new__(VLMJsonlDataset)
    dataset.path = str(path)
    dataset._meta = {"offsets": [0]}
    dataset.media_root = str(tmp_path)
    dataset.tokenize_fn = tokenize_fn
    dataset.fake_data = {"messages": [{"role": "user", "content": "fallback"}]}
    return dataset


@pytest.mark.parametrize("tokenizer_cls", [RLGlm53VLTokenizeFunction, RLQwen3VLTokenizeFunction, RLTextTokenizeFn])
def test_rl_preserves_original_media_error(tmp_path, monkeypatch, tokenizer_cls):
    error = FileNotFoundError("missing-image.png")
    call = Mock(side_effect=error)
    monkeypatch.setattr(tokenizer_cls, "__call__", call)
    tokenizer = tokenizer_cls.__new__(tokenizer_cls)
    dataset = make_dataset(tmp_path, tokenizer)
    with pytest.raises(FileNotFoundError) as caught:
        dataset[0]
    assert caught.value is error
    call.assert_called_once()


def test_rl_preserves_annotation_read_error(tmp_path):
    tokenizer = Mock(supports_fake_data_fallback=False)
    dataset = make_dataset(tmp_path, tokenizer, line="not json\n")
    with pytest.raises(json.JSONDecodeError):
        dataset[0]
    tokenizer.assert_not_called()


@pytest.mark.parametrize("declared_capability", [True, False])
def test_sft_fallback_keeps_ignored_labels(tmp_path, declared_capability):
    calls = []

    def tokenize(item, **kwargs):
        calls.append(item)
        if len(calls) == 1:
            raise FileNotFoundError("missing-image.png")
        return {"input_ids": [1, 2], "labels": [1, 2]}

    if declared_capability:
        tokenize.supports_fake_data_fallback = CachableTokenizeFunction.supports_fake_data_fallback
    dataset = make_dataset(tmp_path, tokenize)
    assert dataset[0] == {"input_ids": [1, 2], "labels": [-100, -100]}
    assert calls[1] is dataset.fake_data


def test_successful_rl_sample_is_returned_unchanged(tmp_path):
    state = RolloutState(message=[], data_source={"vl": 1.0})
    tokenizer = Mock(supports_fake_data_fallback=False, return_value=state)
    dataset = make_dataset(tmp_path, tokenizer)
    assert dataset[0] is state
    tokenizer.assert_called_once()
