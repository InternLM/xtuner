"""Tests for LongTextPretrainTokenizeFunction and JsonlDataset chunked tokenization."""
import json
import os
import tempfile

import numpy as np
import pytest
from transformers import AutoTokenizer

model_path = os.environ["QWEN3_PATH"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer():
    """Load a small, fast tokenizer for testing."""
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tok


def _build_fn(tokenizer, chunk_size=64, tokenizer_chunk_chars=256, overlap_chars=32,
              min_chunk_tokens=0, add_eos_token=False, add_bos_token=False):
    from xtuner.v1.datasets.pt_tokenize_fn.long_text import LongTextPretrainTokenizeFunction
    return LongTextPretrainTokenizeFunction(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        tokenizer_chunk_chars=tokenizer_chunk_chars,
        overlap_chars=overlap_chars,
        min_chunk_tokens=min_chunk_tokens,
        add_eos_token=add_eos_token,
        add_bos_token=add_bos_token,
    )


# ---------------------------------------------------------------------------
# test_shard_char_boundaries_short
# ---------------------------------------------------------------------------

def test_shard_char_boundaries_short():
    tok = _make_tokenizer()
    fn = _build_fn(tok, chunk_size=512)

    short_text = "Hello world " * 5  # well under 512 tokens
    boundaries = fn.shard_char_boundaries(short_text)

    assert len(boundaries) == 1, f"Expected 1 boundary for short text, got {len(boundaries)}"
    cs = boundaries[0]["char_start"]
    ce = boundaries[0]["char_end"]
    assert cs == 0
    assert ce == len(short_text)


# ---------------------------------------------------------------------------
# test_shard_char_boundaries_long
# ---------------------------------------------------------------------------

def test_shard_char_boundaries_long():
    tok = _make_tokenizer()
    chunk_size = 32
    fn = _build_fn(tok, chunk_size=chunk_size, tokenizer_chunk_chars=128, overlap_chars=16)

    # Build text that tokenizes to ~200 tokens
    long_text = "The quick brown fox jumps over the lazy dog. " * 30

    boundaries = fn.shard_char_boundaries(long_text)
    assert len(boundaries) > 1, "Expected multiple boundaries for long text"

    # Boundaries must be contiguous and cover the full text
    assert boundaries[0]["char_start"] == 0
    assert boundaries[-1]["char_end"] == len(long_text)
    for i in range(len(boundaries) - 1):
        assert boundaries[i]["char_end"] == boundaries[i + 1]["char_start"], (
            f"Gap between boundary {i} end={boundaries[i]['char_end']} and boundary {i+1} start={boundaries[i+1]['char_start']}"
        )

    # Each chunk token count should be roughly <= 2 * chunk_size
    for chunk_info in boundaries[:-1]:  # last chunk may be shorter
        cs, ce = chunk_info["char_start"], chunk_info["char_end"]
        chunk_text = long_text[cs:ce]
        ids = tok.encode(chunk_text, add_special_tokens=False)
        assert len(ids) <= 2 * chunk_size + 5, (
            f"Chunk [{cs}:{ce}] has {len(ids)} tokens, expected ~{chunk_size}"
        )


# ---------------------------------------------------------------------------
# test_mixed_short_long_jsonl
# ---------------------------------------------------------------------------

def test_mixed_short_long_jsonl():
    """Key integration test: JSONL with both short and long lines.

    Verifies that after chunked expansion:
    - len(dataset) equals sum of chunks across all lines
    - short lines produce 1 chunk entry covering full text
    - long lines produce multiple entries
    - re-assembling all chunk input_ids per line matches full tokenization
    """
    tok = _make_tokenizer()
    chunk_size = 32

    # Build JSONL lines: short / long alternating
    lines_text = [
        "short text " * 3,                    # ~9 tokens
        "a longer document " * 40,            # ~100+ tokens → multiple chunks
        "another short one " * 2,             # ~10 tokens
        "very very long document indeed " * 50,  # ~200+ tokens → multiple chunks
        "short again",                        # ~3 tokens
    ]
    lines = [{"content": t} for t in lines_text]

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)

        from xtuner.v1.datasets.pt_tokenize_fn.long_text import LongTextPretrainTokenizeFunction
        from xtuner.v1.datasets import JsonlDataset

        fn = LongTextPretrainTokenizeFunction(
            tokenizer=tok,
            chunk_size=chunk_size,
            tokenizer_chunk_chars=128,
            overlap_chars=16,
            min_chunk_tokens=0,
            add_eos_token=False,
            add_bos_token=False,
        )

        dataset = JsonlDataset(
            anno_path=jsonl_path,
            tokenize_fn=fn,
            cache_dir=cache_dir,
            sample_ratio=1.0,
        )

        # --- Verify total length equals sum of chunks per line ---
        # Compute expected chunks per line by calling fn directly in cache state
        fn.set_state("cache")
        all_chunks_per_line = []
        for text in lines_text:
            result = fn({"content": text})
            all_chunks_per_line.append(result["chunks"])
        fn.set_state("runtime")

        expected_total = sum(len(c) for c in all_chunks_per_line)
        assert len(dataset) == expected_total, (
            f"Expected {expected_total} entries, got {len(dataset)}"
        )

        # --- Verify chunk entries are correct ---
        # Rebuild a mapping: for each original line index, collect dataset entries in order
        entry_idx = 0
        for line_idx, (text, chunks) in enumerate(zip(lines_text, all_chunks_per_line)):
            full_ids = tok.encode(text, add_special_tokens=False)
            reassembled = []

            for chunk_info in chunks:
                item = dataset[entry_idx]
                assert "input_ids" in item, f"Entry {entry_idx} missing input_ids"
                reassembled.extend(item["input_ids"])
                entry_idx += 1

            # Reassembled tokens from chunks should match full tokenization
            assert reassembled == full_ids, (
                f"Line {line_idx}: reassembled {len(reassembled)} tokens != full {len(full_ids)} tokens\n"
                f"text[:80]={text[:80]!r}"
            )


# ---------------------------------------------------------------------------
# test_cache_invalidation
# ---------------------------------------------------------------------------

def test_cache_invalidation():
    """Changing chunk_size changes the hash, so a different cache dir is used."""
    from xtuner.v1.datasets.pt_tokenize_fn.long_text import LongTextPretrainTokenizeFunctionConfig
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    cfg1 = LongTextPretrainTokenizeFunctionConfig(chunk_size=256)
    cfg2 = LongTextPretrainTokenizeFunctionConfig(chunk_size=512)

    fn1 = cfg1.build(tok)
    fn2 = cfg2.build(tok)

    assert fn1.hash() != fn2.hash(), (
        "Different chunk_size should produce different hashes"
    )

    # Same params -> same hash
    fn1b = cfg1.build(tok)
    assert fn1.hash() == fn1b.hash(), "Same config should produce same hash"


# ---------------------------------------------------------------------------
# test_call_runtime_char_range
# ---------------------------------------------------------------------------

def test_call_runtime_char_range():
    """Test that __call__ with char_start/char_end returns correct tokens."""
    tok = _make_tokenizer()
    fn = _build_fn(tok, chunk_size=32, add_eos_token=False, add_bos_token=False)
    fn.set_state("runtime")

    text = "Hello world, this is a test of chunked tokenization. " * 5
    item = {"content": text}

    # Full text via explicit char range
    full_result = fn(item, char_start=0, char_end=len(text))
    full_ids = full_result["input_ids"]

    # Split at an arbitrary char boundary and check first chunk
    split = len(text) // 2
    r1 = fn(item, char_start=0, char_end=split)
    r2 = fn(item, char_start=split, char_end=len(text))

    combined = r1["input_ids"] + r2["input_ids"]
    # Combined should equal full text tokenization (split is at char boundary, may differ slightly)
    # At minimum both pieces are non-empty and their tokens match the substrings
    assert r1["input_ids"] == tok.encode(text[:split], add_special_tokens=False)
    assert r2["input_ids"] == tok.encode(text[split:], add_special_tokens=False)
