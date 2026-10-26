"""Tests for the node-local compile cache (`xtuner/compile_cache.py`).

Unit tests cover warmup, staging, and offline merge file semantics in-process.
Multi-process tests check that 8 local ranks share a cache and that two simulated
nodes produce artifacts which the offline merge preserves. GPU tests must run under
`zdev/gpu_lock.sh`.

Local mode activates whenever `XTUNER_COMPILE_CACHE_DIR` is set; subprocess workers patch
`cc.LOCAL_ROOT` from `TEST_LOCAL_ROOT` before activation so tests never touch the real
default `/tmp/xtuner-compile-cache`.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from xtuner import compile_cache as cc


REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_CLEAN = (
    cc.ENV_SHARED_DIR,
    cc.ENV_CACHE_DIR,
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    *cc.COMPILER_CACHE_ENVS,
)

# Worker code is flush-left (no dedent). Every worker starts with this snippet so the
# warmup never touches the real default local root.
_PATCH_LOCAL_ROOT = """
import os
import time
from pathlib import Path

from xtuner import compile_cache as cc

cc.LOCAL_ROOT = Path(os.environ["TEST_LOCAL_ROOT"])
"""


@pytest.fixture(autouse=True)
def _clean_env_and_state(monkeypatch, tmp_path):
    for name in _ENV_CLEAN:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cc, "_STATE", None)
    monkeypatch.setattr(cc, "LOCAL_ROOT", tmp_path / "local")


@pytest.fixture
def fixed_namespace(monkeypatch):
    monkeypatch.setattr(cc, "_namespace", lambda: "test-ns")
    return "test-ns"


def _use_shared(monkeypatch, tmp_path):
    shared = tmp_path / "shared"
    monkeypatch.setenv(cc.ENV_CACHE_DIR, str(shared))
    return shared


def _stage(shared, node, files, marker="READY"):
    node_dir = shared / ".staging" / f"node_{node}"
    (node_dir / "files").mkdir(parents=True)
    for rel, data in files.items():
        path = node_dir / "files" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (node_dir / marker).write_text("{}")
    return node_dir


def _state(shared, tmp_path, num_nodes=1, node_rank=0, global_rank=0):
    return cc._State(
        shared_dir=shared,
        namespace="test-ns",
        run_dir=tmp_path / f"local-{node_rank}",
        local_rank=0,
        global_rank=global_rank,
        node_rank=node_rank,
        num_nodes=num_nodes,
    )


def _worker_env(shared, local_root, rank, local_rank, world, local_world, extra=None):
    env = os.environ.copy()
    for name in _ENV_CLEAN:
        env.pop(name, None)
    env.update(
        {
            cc.ENV_CACHE_DIR: str(shared),
            "TEST_LOCAL_ROOT": str(local_root),
            "RANK": str(rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world),
            "LOCAL_WORLD_SIZE": str(local_world),
            "GROUP_RANK": str(rank // local_world),
            "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
    )
    env.update(extra or {})
    return env


def _run_worker(code, env, timeout=300):
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, f"worker failed with exit code {proc.returncode}:\n{proc.stdout}"
    return proc.stdout


# ---------------------------------------------------------------------------
# In-process unit tests
# ---------------------------------------------------------------------------


def test_no_shared_dir_is_noop():
    assert cc.activate_local_cache() is None
    assert cc.writeback() is None
    assert cc.ENV_CACHE_DIR not in os.environ


def test_rejects_compiler_cache_outside_local(monkeypatch, tmp_path, fixed_namespace):
    _use_shared(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(RuntimeError, match="TRITON_CACHE_DIR"):
        cc.activate_local_cache()


def test_allows_compiler_cache_inside_local(monkeypatch, tmp_path, fixed_namespace):
    _use_shared(monkeypatch, tmp_path)
    cache_dir = cc.LOCAL_ROOT / "cache"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_dir / "triton"))
    assert cc.activate_local_cache() is not None


def test_retargets_xtuner_derived_compiler_cache(monkeypatch, tmp_path, fixed_namespace):
    """`python -m`/module-import launches run `xtuner/v1/__init__.py` before activation, so
    the compiler cache dirs already point at the shared root's defaults; activation must
    retarget them to the local cache instead of failing."""
    shared = _use_shared(monkeypatch, tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(shared.resolve() / "triton"))
    state = cc.activate_local_cache()
    assert state is not None
    assert Path(os.environ["TRITON_CACHE_DIR"]) == state.cache_dir / "triton"


def test_single_process_lifecycle(monkeypatch, tmp_path, fixed_namespace):
    shared = _use_shared(monkeypatch, tmp_path)
    ns_root = shared / "test-ns"
    (ns_root / "torchinductor").mkdir(parents=True)
    (ns_root / "triton").mkdir()
    (ns_root / "torchinductor" / "graph_a").write_bytes(b"A" * 10)
    (ns_root / "triton" / "kern.bin").write_bytes(b"K")
    (ns_root / "triton" / "kern.bin.tmp").write_bytes(b"partial")  # incomplete: never copied
    (ns_root / "triton" / ".kern.bin.tmp.123").write_bytes(b"partial")  # interrupted merge temp
    (ns_root / "triton" / "inflight.lock").write_text("{}")  # lock file: never copied

    state = cc.activate_local_cache()
    assert state is not None
    local = cc.LOCAL_ROOT
    cache = local / "cache"
    assert Path(os.environ[cc.ENV_CACHE_DIR]) == cache
    assert os.environ[cc.ENV_SHARED_DIR] == str(shared.resolve())
    assert (cache / "torchinductor" / "graph_a").read_bytes() == b"A" * 10
    assert (cache / "triton" / "kern.bin").read_bytes() == b"K"
    assert not (cache / "triton" / "kern.bin.tmp").exists()
    assert not (cache / "triton" / ".kern.bin.tmp.123").exists()
    assert not (cache / "triton" / "inflight.lock").exists()
    assert (local / "ready").is_file()
    manifest = json.loads((local / "manifest.json").read_text())["files"]
    assert sorted(manifest) == ["torchinductor/graph_a", "triton/kern.bin"]

    # Training mutates the local cache: one new file, one modified, one unfinished.
    (cache / "torchinductor" / "graph_b").write_bytes(b"B" * 20)
    (cache / "triton" / "kern.bin").write_bytes(b"K2")
    (cache / "triton" / "inprogress.tmp").write_bytes(b"x")

    barriers = []
    staged_state = cc.writeback(barrier=lambda: barriers.append("staged"))

    assert barriers == ["staged"], "the slow merge must not run inside a training barrier"
    assert staged_state is state
    assert (shared / ".staging" / "node_0" / "READY").is_file()
    assert not (ns_root / "torchinductor" / "graph_b").exists()
    assert cc.merge_staging(staged_state)
    assert (ns_root / "torchinductor" / "graph_b").read_bytes() == b"B" * 20
    # A locally modified file collides with the existing shared entry; the original wins.
    assert (ns_root / "triton" / "kern.bin").read_bytes() == b"K"
    assert not (ns_root / "triton" / "inprogress.tmp").exists()
    assert (ns_root / "torchinductor" / "graph_a").read_bytes() == b"A" * 10
    report = json.loads((shared / ".merge-report.json").read_text())
    assert report["nodes"] == [0]
    assert report["published"] == {"0": 1}
    assert report["skipped_existing"] == {"0": 1}
    assert not local.exists(), "local run dir not cleaned"
    assert not (shared / ".staging").exists(), "staging path should be free for the next run"
    assert Path(report["staging_archive"]).is_dir()
    assert cc.main(["cleanup", "--shared-dir", str(shared)]) == 0
    assert not Path(report["staging_archive"]).exists()


def test_merge_conflict_keeps_existing(tmp_path):
    shared = tmp_path / "shared"
    (shared / "test-ns").mkdir(parents=True)
    (shared / "test-ns" / "same.txt").write_bytes(b"identical")
    _stage(shared, 0, {"a.txt": b"first", "same.txt": b"identical"})
    _stage(shared, 1, {"a.txt": b"second", "b.txt": b"only-node-1"})

    assert cc.main(["merge", "--shared-dir", str(shared), "--namespace", "test-ns", "--num-nodes", "2"]) == 0
    report = json.loads((shared / ".merge-report.json").read_text())

    assert (shared / "test-ns" / "a.txt").read_bytes() == b"first"
    assert (shared / "test-ns" / "b.txt").read_bytes() == b"only-node-1"
    assert (shared / "test-ns" / "same.txt").read_bytes() == b"identical"
    assert report["nodes"] == [0, 1]
    assert report["published"] == {"0": 1, "1": 1}
    assert report["skipped_existing"] == {"0": 1, "1": 1}


def test_merge_aborts_when_a_node_failed(tmp_path):
    shared = tmp_path / "shared"
    _stage(shared, 0, {"a.txt": b"x"})
    _stage(shared, 1, {"b.txt": b"y"}, marker="FAILED")
    state = _state(shared, tmp_path, num_nodes=2)

    assert cc._merge_staging(state) is None
    assert not (shared / "test-ns").exists(), "shared cache must stay untouched"
    assert (shared / ".staging" / "node_0").exists(), "staging kept for inspection"


def test_merge_aborts_when_a_node_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_STAGING_WAIT_S", 0.1)
    shared = tmp_path / "shared"
    _stage(shared, 0, {"a.txt": b"x"})
    state = _state(shared, tmp_path, num_nodes=2)

    assert cc._merge_staging(state) is None
    assert not (shared / "test-ns").exists()


def test_offline_merge_requires_namespace_without_gpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cc, "_namespace", lambda: "torch-test-nogpu")
    with pytest.raises(SystemExit) as exc:
        cc.main(["merge", "--shared-dir", str(tmp_path), "--num-nodes", "1"])
    assert exc.value.code == 2
    assert "requires --namespace" in capsys.readouterr().err


def test_merge_ignores_staging_beyond_num_nodes(tmp_path):
    """Leftover node dirs from a crashed larger run must not be merged."""
    shared = tmp_path / "shared"
    _stage(shared, 0, {"a.txt": b"x"})
    _stage(shared, 1, {"stale.txt": b"stale"})
    state = _state(shared, tmp_path, num_nodes=1)

    report = cc._merge_staging(state)
    assert report["nodes"] == [0]
    assert (shared / "test-ns" / "a.txt").read_bytes() == b"x"
    assert not (shared / "test-ns" / "stale.txt").exists()


def test_waiter_aborts_on_failed_marker(monkeypatch, tmp_path, fixed_namespace):
    _use_shared(monkeypatch, tmp_path)
    cc.LOCAL_ROOT.mkdir()
    (cc.LOCAL_ROOT / "failed").write_text(json.dumps({"error": "boom"}))
    monkeypatch.setenv("LOCAL_RANK", "1")
    with pytest.raises(RuntimeError, match="boom"):
        cc.activate_local_cache()


def test_waiter_ignores_stale_ready(monkeypatch, tmp_path, fixed_namespace):
    _use_shared(monkeypatch, tmp_path)
    monkeypatch.setattr(cc, "_WARMUP_TIMEOUT_S", 3)
    cc.LOCAL_ROOT.mkdir()
    ready = cc.LOCAL_ROOT / "ready"
    ready.write_text("{}")
    old = time.time_ns() - 3600 * 10**9
    os.utime(ready, ns=(old, old))
    monkeypatch.setenv("LOCAL_RANK", "1")
    with pytest.raises(RuntimeError, match="timed out"):
        cc.activate_local_cache()


# ---------------------------------------------------------------------------
# Multi-process tests
# ---------------------------------------------------------------------------


def test_eight_local_ranks_share_one_warmup(tmp_path):
    shared = tmp_path / "shared"
    namespace = cc._namespace()
    seed = shared / namespace / "torchinductor" / "seed.txt"
    seed.parent.mkdir(parents=True)
    seed.write_text("seed-data")
    local_root = tmp_path / "local"

    worker = (
        _PATCH_LOCAL_ROOT
        + """
state = cc.activate_local_cache()
assert state is not None
cache = os.environ["XTUNER_COMPILE_CACHE_DIR"]
expected = os.path.join(os.environ["TEST_LOCAL_ROOT"], "cache")
assert cache == expected, (cache, expected)
with open(os.path.join(cache, "torchinductor", "seed.txt")) as f:
    assert f.read() == "seed-data"
assert os.environ[cc.ENV_SHARED_DIR] == os.environ["EXPECTED_SHARED"]
print("RANK-OK", os.environ["RANK"], flush=True)
"""
    )
    procs = []
    for rank in range(8):
        env = _worker_env(shared, local_root, rank, rank, 8, 8, extra={"EXPECTED_SHARED": str(shared.resolve())})
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", worker],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    ok_ranks = 0
    for rank, proc in enumerate(procs):
        out, _ = proc.communicate(timeout=300)
        assert proc.returncode == 0, f"rank {rank} failed:\n{out}"
        assert f"RANK-OK {rank}" in out
        ok_ranks += 1
    assert ok_ranks == 8


_COMPILE_AND_WRITEBACK_WORKER = (
    _PATCH_LOCAL_ROOT
    + """
state = cc.activate_local_cache()
assert state is not None, "local compile cache not activated"

import torch

import xtuner.v1  # noqa: F401  derives per-compiler cache dirs from the local cache

tag = float(os.environ["NODE_TAG"])


@torch.compile
def f(x):
    return torch.sin(x) * tag + torch.cos(x)


x = torch.randn(2048, 2048, device="cuda")
for _ in range(3):
    f(x)
torch.cuda.synchronize()

barrier_dir = Path(os.environ["BARRIER_DIR"])
world = int(os.environ["WORLD_SIZE"])


def file_barrier():
    (barrier_dir / os.environ["RANK"]).touch()
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if len(list(barrier_dir.glob("*"))) >= world:
            return
        time.sleep(0.5)
    raise RuntimeError("file barrier timed out")


cc.writeback(barrier=file_barrier)
print("DONE rank=%s" % os.environ["RANK"], flush=True)
"""
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for real torch.compile")
def test_two_nodes_torch_compile_roundtrip(tmp_path):
    """Two simulated nodes (8 local ranks each) compile different graphs; local rank 0
    stages each node's artifacts before an offline merge. The write-back barrier is a
    file barrier over a shared directory, standing in for `dist.barrier`."""
    shared = tmp_path / "shared"
    namespace = cc._namespace()
    gpus = max(1, torch.cuda.device_count())
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()

    procs = []
    for rank in range(16):
        node = rank // 8
        env = _worker_env(
            shared,
            tmp_path / f"local{node}",
            rank,
            rank % 8,
            16,
            8,
            extra={
                "NODE_TAG": str(node + 1),
                "CUDA_VISIBLE_DEVICES": str(rank % gpus),
                "BARRIER_DIR": str(barrier_dir),
            },
        )
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", _COMPILE_AND_WRITEBACK_WORKER],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    for rank, proc in enumerate(procs):
        out, _ = proc.communicate(timeout=900)
        assert proc.returncode == 0, f"compile rank {rank} failed:\n{out}"
        assert f"DONE rank={rank}" in out

    staged = {
        str(node): {
            p.relative_to(shared / ".staging" / f"node_{node}" / "files")
            for p in (shared / ".staging" / f"node_{node}" / "files").rglob("*")
            if p.is_file()
        }
        for node in (0, 1)
    }
    assert cc.main(["merge", "--shared-dir", str(shared), "--namespace", namespace, "--num-nodes", "2"]) == 0
    report = json.loads((shared / ".merge-report.json").read_text())
    assert report["nodes"] == [0, 1]
    assert staged["0"] and staged["1"], "torch.compile produced no cache artifacts"
    assert staged["0"] - staged["1"] and staged["1"] - staged["0"], "nodes produced no distinct artifacts"
    # Same-path artifacts can differ byte-wise across processes; existing files win.
    ns_root = shared / namespace
    for rel in staged["0"] | staged["1"]:
        assert (ns_root / rel).is_file(), f"artifact lost in merge: {rel}"
    # The CUDA driver may recreate an empty cache/cuda skeleton when its cache flushes at
    # process exit (after write-back cleanup); no payload file may survive, though.
    for node in (0, 1):
        leftover = [p for p in (tmp_path / f"local{node}").rglob("*") if p.is_file()]
        assert leftover == [], f"local{node} not cleaned: {leftover}"
    assert not (shared / ".staging").exists()
    assert cc.main(["cleanup", "--shared-dir", str(shared)]) == 0


def test_sft_entry_activates_before_xtuner_import(tmp_path):
    """`sft.py` must repoint XTUNER_COMPILE_CACHE_DIR before `xtuner.v1` derives the
    per-compiler cache paths from it."""
    shared = tmp_path / "shared"
    env = _worker_env(shared, tmp_path / "local", 0, 0, 1, 1)
    out = _run_worker(
        _PATCH_LOCAL_ROOT
        + """
import xtuner.v1.train.cli.sft  # noqa: F401

cache = os.environ["XTUNER_COMPILE_CACHE_DIR"]
assert cache.endswith("/cache"), cache
assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == os.path.join(cache, "torchinductor")
assert os.environ["XTUNER_COMPILE_CACHE_SHARED_DIR"]
print("SFT-WIRING-OK", flush=True)
""",
        env,
        timeout=600,
    )
    assert "SFT-WIRING-OK" in out


def test_sft_script_mode_warms_local_cache(tmp_path):
    """Production launches `sft.py` as a script, so activation precedes every third-party
    import; warmup must populate the node-local cache before the CLI even parses args."""
    shared = tmp_path / "shared"
    namespace = cc._namespace()
    seed = shared / namespace / "torchinductor" / "seed.txt"
    seed.parent.mkdir(parents=True)
    seed.write_text("seed")
    env = _worker_env(
        shared,
        tmp_path / "local",
        0,
        0,
        1,
        1,
        extra={"SCRIPT": str(REPO_ROOT / "xtuner" / "v1" / "train" / "cli" / "sft.py")},
    )
    # Emulate `python sft.py --help`: patch LOCAL_ROOT, then execute the script top-level
    # (activation runs at its first line) exactly as the script launch path does.
    out = _run_worker(
        _PATCH_LOCAL_ROOT
        + """
import sys

sys.argv = ["sft.py", "--help"]
with open(os.environ["SCRIPT"]) as f:
    exec(f.read())
""",
        env,
        timeout=600,
    )
    assert "XTuner" in out  # cyclopts help text
    local = tmp_path / "local"
    assert (local / "ready").is_file()
    assert (local / "cache" / "torchinductor" / "seed.txt").read_text() == "seed"
