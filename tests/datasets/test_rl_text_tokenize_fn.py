# Copyright (c) OpenMMLab. All rights reserved.
"""RLTextTokenizeFn cache key must track tokenizer identity."""

import os

import pytest

from transformers import AutoTokenizer
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig


GLM5_2_TINY_MOE_PATH = os.environ["GLM5_2_TINY_MOE_PATH"]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH, trust_remote_code=True)


def _build(tokenizer):
    fn = RLTextTokenizeFnConfig(max_length=128).build(tokenizer)
    fn.set_state("runtime")
    return fn


def test_hash_is_stable_and_prefixed(tokenizer):
    first = _build(tokenizer).hash()
    second = _build(tokenizer).hash()
    assert first == second
    assert first.startswith("RLTextTokenizeFn_")


def test_hash_tracks_tokenizer_config(tokenizer, tmp_path):
    # A tokenizer whose serialized config differs (here: padding side, a proxy
    # for the real case of swapping in a custom tokenizer class via auto_map)
    # must produce a different cache key, so stale ids are never reused.
    base = _build(tokenizer).hash()
    tokenizer.padding_side = "right" if tokenizer.padding_side != "right" else "left"
    modified = _build(tokenizer).hash()
    tokenizer.save_pretrained(str(tmp_path))
    reloaded = _build(AutoTokenizer.from_pretrained(str(tmp_path), trust_remote_code=True)).hash()
    assert modified != base
    assert reloaded == modified


def test_tokenize_fn_still_runs(tokenizer):
    fn = _build(tokenizer)
    state = fn({"data_source": "x", "prompt": [{"role": "user", "content": "hi"}], "extra_info": {}})
    assert len(state.prompt_ids) > 0
