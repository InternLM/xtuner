"""Heavy-scale checks for :func:`get_pack_config_from_pack_infos_by_hard_split`.

Default pytest runs skip the 100M-row case (multi‑GB RAM). Enable with::

    XTUNER_PACK_CONFIG_STRESS=1 bash run_test.sh

or::

    XTUNER_PACK_CONFIG_STRESS=1 pytest -v -s tests/datasets/test_pack_config_from_pack_infos_stress.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from xtuner.v1.datasets.utils import concat_cumulative_sizes_from_lengths, get_pack_config_from_pack_infos_by_hard_split


STRESS_ROWS = 100_000_000
STRESS_VOCAB = 4096  # num_tokens table size; indices reference this modulo range
_MIN_MEM_GIB = 12  # rough lower bound for ix + samples temporaries


def _mem_available_gib() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except OSError:
        pass
    return float("inf")


requires_stress = pytest.mark.skipif(
    os.environ.get("XTUNER_PACK_CONFIG_STRESS", "").strip().lower() not in ("1", "true", "yes"),
    reason="set XTUNER_PACK_CONFIG_STRESS=1 to run 100M-row stress test",
)


@requires_stress
def test_pack_config_from_pack_infos_hundred_million_rows():
    avail = _mem_available_gib()
    if avail < _MIN_MEM_GIB:
        pytest.skip(f"MemAvailable {avail:.1f} GiB < {_MIN_MEM_GIB} GiB (stress test needs RAM for ~8+ GiB peak)")

    rng = np.random.default_rng(12345)
    n = STRESS_ROWS
    num_tokens = rng.integers(8, 24, size=STRESS_VOCAB, dtype=np.int64)
    ix = rng.integers(0, STRESS_VOCAB, size=n, dtype=np.int64)

    pack_infos = {
        "dataset_id": np.zeros(1, dtype=np.int64),
        "indices": ix,
        "indices_cu_len": np.array([n], dtype=np.int64),
        "start_offset": np.zeros(1, dtype=np.int64),
        "end_offset": np.array([int(num_tokens[ix[-1]])], dtype=np.int64),
        "longest": np.array([int(num_tokens.max())], dtype=np.int64),
    }

    # single-path
    t0 = time.perf_counter()
    out = get_pack_config_from_pack_infos_by_hard_split(
        pack_infos, 0, num_tokens, paths=["stress.jsonl"]
    )
    elapsed = time.perf_counter() - t0

    assert out["samples"].shape == (n, 6)
    assert out["boundaries"].shape == (2,)
    np.testing.assert_array_equal(out["boundaries"], [0, n])
    assert out["paths"] == ["stress.jsonl"]

    head = 10_000
    np.testing.assert_array_equal(out["samples"][:head, 0], 0)
    np.testing.assert_array_equal(out["samples"][:head, 1], ix[:head])
    np.testing.assert_array_equal(out["samples"][:head, 2], -1)
    np.testing.assert_array_equal(out["samples"][:head, 3], -1)
    assert out["samples"][0, 4] == 0
    assert out["samples"][0, 5] == int(num_tokens[ix[0]])
    assert out["samples"][-1, 4] == 0
    assert out["samples"][-1, 5] == int(num_tokens[ix[-1]])

    # Concat path: two fake shards, flat indices still in [0, vocab)
    n0, n1 = STRESS_VOCAB // 2, STRESS_VOCAB - STRESS_VOCAB // 2
    cu = concat_cumulative_sizes_from_lengths([n0, n1])
    t1 = time.perf_counter()
    out_c = get_pack_config_from_pack_infos_by_hard_split(
        pack_infos, 0, num_tokens, paths=["a.jsonl", "b.jsonl"], concat_cumulative_sizes=cu
    )
    elapsed_c = time.perf_counter() - t1

    assert out_c["samples"].shape == (n, 6)
    for row in (0, n // 2, n - 1):
        flat = int(ix[row])
        pid, local = (0, flat) if flat < n0 else (1, flat - n0)
        assert int(out_c["samples"][row, 0]) == pid
        assert int(out_c["samples"][row, 1]) == local

    print(
        f"\n[stress] n={n:,} single-path {elapsed:.2f}s, concat {elapsed_c:.2f}s "
        f"(MemAvailable ~{avail:.1f} GiB)"
    )


def test_pack_config_vectorized_matches_reference_small():
    """Regression guard: ``mode='numpy'`` matches ``mode='loop'`` on tiny synthetic pack_infos."""

    rng = np.random.default_rng(7)
    num_tokens = rng.integers(5, 30, size=50).astype(np.int64)
    cu_lens = np.array([3, 7, 12], dtype=np.int64)
    ix_parts: list[np.ndarray] = []
    for k in cu_lens:
        ix_parts.append(rng.integers(0, 50, size=int(k), dtype=np.int64))
    ix = np.concatenate(ix_parts)
    cu = np.cumsum(cu_lens.astype(np.int64))
    pack_infos = {
        "dataset_id": np.arange(3, dtype=np.int64),
        "indices": ix,
        "indices_cu_len": cu,
        "start_offset": rng.integers(0, 3, size=3).astype(np.int64),
        "end_offset": rng.integers(10, 20, size=3).astype(np.int64),
        "longest": rng.integers(20, 40, size=3).astype(np.int64),
    }

    # single-path
    paths = ["only.jsonl"]
    got = get_pack_config_from_pack_infos_by_hard_split(pack_infos, 9, num_tokens, paths=paths, mode="numpy")
    expected = get_pack_config_from_pack_infos_by_hard_split(pack_infos, 9, num_tokens, paths=paths, mode="loop")
    np.testing.assert_array_equal(got["boundaries"], expected["boundaries"])
    np.testing.assert_array_equal(got["samples"], expected["samples"])
    np.testing.assert_array_equal(got["longest"], expected["longest"])
    assert got["paths"] == expected["paths"]

    # concat two paths
    cu2 = concat_cumulative_sizes_from_lengths([20, 30])
    paths2 = ["x.jsonl", "y.jsonl"]
    got2 = get_pack_config_from_pack_infos_by_hard_split(
        pack_infos, 0, num_tokens, paths=paths2, concat_cumulative_sizes=cu2, mode="numpy"
    )
    expected2 = get_pack_config_from_pack_infos_by_hard_split(
        pack_infos, 0, num_tokens, paths=paths2, concat_cumulative_sizes=cu2, mode="loop"
    )
    np.testing.assert_array_equal(got2["samples"], expected2["samples"])
