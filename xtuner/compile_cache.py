"""Node-local compile-cache warmup and centralized write-back.

Simplified implementation of ``exp256gpu/03_sp8_perf/xtuner_compile_cache.md``: during
training every compiler (Inductor, Triton, TileLang, CuTe DSL, CUDA) only touches a
node-local cache directory, while the cross-job shared cache under
``XTUNER_COMPILE_CACHE_DIR`` is read once at startup and written once at the end. Jobs are
assumed not to run concurrently on the same node or cache root.

Lifecycle (driven by ``xtuner/v1/train/cli/sft.py``):

1. Before importing torch or ``xtuner.v1``, every rank calls :func:`activate_local_cache`.
   The node's ``LOCAL_RANK=0`` copies the shared cache namespace to
   ``<local_root>/cache.partial``, records the starting file manifest, atomically renames
   it to ``cache`` and writes a ``ready`` marker; the other local ranks wait for the
   marker. Warmup failure aborts every rank on the node. Afterwards each process saves the
   shared path to ``XTUNER_COMPILE_CACHE_SHARED_DIR`` and repoints
   ``XTUNER_COMPILE_CACHE_DIR`` at the node-local cache, so ``xtuner/v1/__init__.py`` picks
   the local directory up when it sets the per-compiler cache paths.
2. After ``Trainer.fit()`` returns, :func:`writeback` runs before the process group is
   destroyed: each node's local rank 0 copies its cache delta to shared staging and
   publishes a ``READY`` marker. A global barrier confirms staging is complete. Merge
   runs on global rank 0 after the training process group is destroyed, so remote
   filesystem latency cannot time out a distributed barrier. Existing paths keep their
   original file; new files are published by temp-file-plus-rename. The merge can be
   retried safely.

The module only depends on the standard library so it can run before any third-party
import. Distributed barriers are injected by the caller.

Environment variables:

- ``XTUNER_COMPILE_CACHE_DIR`` (existing): shared cache root. Local mode activates
  whenever it is set.
- ``XTUNER_COMPILE_CACHE_SHARED_DIR``: written by :func:`activate_local_cache`; records the
  shared source after ``XTUNER_COMPILE_CACHE_DIR`` is repointed.
"""

import argparse
import concurrent.futures
import json
import logging
import math
import os
import re
import shutil
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path


logger = logging.getLogger("xtuner.compile_cache")

ENV_SHARED_DIR = "XTUNER_COMPILE_CACHE_SHARED_DIR"
ENV_CACHE_DIR = "XTUNER_COMPILE_CACHE_DIR"

# Node-local cache root; /tmp is per-node on every cluster machine. Tests patch this.
LOCAL_ROOT = Path("/tmp/xtuner-compile-cache")

# Per-compiler cache paths that `xtuner/v1/__init__.py` derives from XTUNER_COMPILE_CACHE_DIR.
COMPILER_CACHE_DIRS = (
    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ("TRITON_CACHE_DIR", "triton"),
    ("TILELANG_CACHE_DIR", "tilelang"),
    ("CUTE_DSL_CACHE_DIR", "cute_dsl"),
    ("CUDA_CACHE_PATH", "cuda"),
)
COMPILER_CACHE_ENVS = tuple(name for name, _ in COMPILER_CACHE_DIRS)

# Names of files a compiler has not finished publishing; never copy them in either direction.
INCOMPLETE_SUFFIXES = (".tmp", ".lock", ".part", ".partial", ".swp")

_COPY_WORKERS = 16
_READY_POLL_S = 1.0
_READY_CLOCK_SLACK_NS = 2 * 10**9  # tolerate mtime vs. process-start clock resolution
_WARMUP_TIMEOUT_S = 4 * 3600
_STAGING_WAIT_S = 600

_STATE = None  # process-local activation result; writeback() is a no-op without it


class _State:
    def __init__(
        self,
        *,
        shared_dir: Path,
        namespace: str,
        run_dir: Path,
        local_rank: int,
        global_rank: int,
        node_rank: int,
        num_nodes: int,
    ):
        self.shared_dir = shared_dir
        self.namespace = namespace
        self.shared_ns_dir = shared_dir / namespace
        self.run_dir = run_dir
        self.cache_dir = run_dir / "cache"
        self.manifest_path = run_dir / "manifest.json"
        self.ready_path = run_dir / "ready"
        self.failed_path = run_dir / "failed"
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.start_ns = time.time_ns()

    @property
    def staging_root(self) -> Path:
        return self.shared_dir / ".staging"

    @property
    def staging_dir(self) -> Path:
        return self.staging_root / f"node_{self.node_rank}"


def activate_local_cache():
    """Warm up the node-local cache and repoint ``XTUNER_COMPILE_CACHE_DIR`` to it.

    Must be called by every rank before importing torch or ``xtuner.v1``. Returns the
    activation state, or ``None`` when ``XTUNER_COMPILE_CACHE_DIR`` is not set. Raises on
    any warmup failure so the whole node aborts instead of training with an inconsistent
    cache.
    """
    global _STATE
    shared_raw = os.getenv(ENV_CACHE_DIR)
    if not shared_raw:
        return None
    _ensure_logging()

    state = _State(
        shared_dir=Path(shared_raw).expanduser().resolve(),
        namespace=_namespace(),
        run_dir=LOCAL_ROOT,
        local_rank=_env_int("LOCAL_RANK", 0),
        global_rank=_env_int("RANK", 0),
        node_rank=_node_rank(),
        num_nodes=_num_nodes(),
    )

    if state.local_rank == 0:
        try:
            stats = _warmup(state)
        except Exception as exc:
            _write_marker(state.failed_path, {"error": f"{type(exc).__name__}: {exc}"})
            raise
        logger.info(
            "compile-cache warmup done: %d files, %.1f MiB, %.1f s (namespace=%s, local=%s)",
            stats["files"],
            stats["bytes"] / 2**20,
            stats["duration_s"],
            state.namespace,
            state.cache_dir,
        )
    else:
        _await_ready(state)
        logger.info("compile-cache warmup waited for local rank 0 (local=%s)", state.cache_dir)

    _repoint_compiler_cache_envs(state)
    os.environ[ENV_SHARED_DIR] = str(state.shared_dir)
    os.environ[ENV_CACHE_DIR] = str(state.cache_dir)
    _STATE = state
    return state


def writeback(barrier=None):
    """Stage this node's cache delta, then release the training process group.

    Call after ``Trainer.fit()`` and before ``dist.destroy_process_group()``, passing
    ``dist.barrier`` (or ``None`` for a single-process run). Return the state so global
    rank 0 can merge staged files after the process group is destroyed.
    """
    global _STATE
    state = _STATE
    if state is None:
        return
    barrier = barrier or (lambda: None)

    if state.local_rank == 0:
        try:
            stats = _publish_staging(state)
            logger.info(
                "compile-cache staging done: %d files, %.1f MiB, %.1f s (node %d -> %s)",
                stats["files"],
                stats["bytes"] / 2**20,
                stats["duration_s"],
                state.node_rank,
                state.staging_dir,
            )
        except Exception as exc:
            logger.exception("compile-cache staging failed on node %d; merge will be skipped", state.node_rank)
            _write_marker(state.staging_dir / "FAILED", {"error": f"{type(exc).__name__}: {exc}"})
    barrier()

    if state.local_rank == 0:
        shutil.rmtree(state.run_dir, ignore_errors=True)
    _STATE = None
    return state


def merge_staging(state) -> bool:
    """Merge staged nodes outside the training process group; retain staging on failure."""
    try:
        report = _merge_staging(state)
    except Exception:
        logger.exception("compile-cache merge failed; staging retained for retry")
        return False
    if report is None:
        return False
    logger.info(
        "compile-cache merge done: %d published, %d existing, %.1f s",
        sum(report["published"].values()),
        sum(report["skipped_existing"].values()),
        report["duration_s"],
    )
    return True


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup(state: _State) -> dict:
    start = time.monotonic()
    if state.run_dir.exists():
        # Leftover of a previous run; never trust it.
        shutil.rmtree(state.run_dir, ignore_errors=True)
    state.run_dir.mkdir(parents=True)
    partial = state.run_dir / "cache.partial"

    entries = _scan_complete_files(state.shared_ns_dir) if state.shared_ns_dir.is_dir() else {}
    partial.mkdir()
    for rel in _copy_files(entries.keys(), state.shared_ns_dir, partial):
        entries.pop(rel, None)  # vanished or unreadable mid-copy; the compiler will rebuild it
    os.rename(partial, state.cache_dir)

    manifest = {
        "namespace": state.namespace,
        "shared_dir": str(state.shared_dir),
        "created": time.time(),
        "files": {rel: [size, mtime_ns] for rel, (size, mtime_ns) in entries.items()},
    }
    _write_marker(state.manifest_path, manifest)
    _write_marker(state.ready_path, {"host": socket.gethostname(), "pid": os.getpid()})
    total_bytes = sum(size for size, _ in entries.values())
    return {"files": len(entries), "bytes": total_bytes, "duration_s": time.monotonic() - start}


def _await_ready(state: _State) -> None:
    deadline = time.monotonic() + _WARMUP_TIMEOUT_S
    while time.monotonic() < deadline:
        for marker, is_error in ((state.failed_path, True), (state.ready_path, False)):
            if not marker.exists():
                continue
            # Ignore leftovers of a crashed previous run: only a fresh marker counts.
            if marker.stat().st_mtime_ns <= state.start_ns - _READY_CLOCK_SLACK_NS:
                continue
            if is_error:
                detail = _read_marker(marker)
                raise RuntimeError(f"compile-cache warmup failed on this node: {detail.get('error', 'unknown')}")
            return
        time.sleep(_READY_POLL_S)
    raise RuntimeError(f"timed out after {_WARMUP_TIMEOUT_S}s waiting for {state.ready_path}")


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


def _publish_staging(state: _State) -> dict:
    start = time.monotonic()
    manifest = _read_marker(state.manifest_path).get("files", {})
    staging_files = state.staging_dir / "files"
    if staging_files.exists():
        shutil.rmtree(staging_files, ignore_errors=True)
    staging_files.mkdir(parents=True)

    current = _scan_complete_files(state.cache_dir) if state.cache_dir.is_dir() else {}
    delta = [rel for rel, (size, mtime_ns) in current.items() if manifest.get(rel) != [size, mtime_ns]]
    failed = _copy_files(delta, state.cache_dir, staging_files)

    stats = {
        "node_rank": state.node_rank,
        "files": len(delta) - len(failed),
        "bytes": sum(current[rel][0] for rel in delta),
        "duration_s": time.monotonic() - start,
        "unchanged": len(current) - len(delta),
    }
    _write_marker(state.staging_dir / "node_stats.json", stats)
    _write_marker(state.staging_dir / "READY", {"host": socket.gethostname(), "pid": os.getpid()})
    return stats


def _merge_staging(state: _State) -> dict | None:
    start = time.monotonic()
    node_dirs = _await_staging_nodes(state.staging_root, state.num_nodes)
    if node_dirs is None:
        logger.error("compile-cache merge aborted; staging left at %s for inspection", state.staging_root)
        return None

    state.shared_ns_dir.mkdir(parents=True, exist_ok=True)
    published, skipped_existing = {}, {}
    seen = set()
    for node_dir in node_dirs:
        node = str(int(node_dir.name[5:]))
        files_root = node_dir / "files"
        published[node], skipped_existing[node] = 0, 0
        for dirpath, _dirnames, filenames in os.walk(files_root):
            for filename in filenames:
                src = Path(dirpath) / filename
                rel = src.relative_to(files_root)
                rel_text = rel.as_posix()
                if not _is_complete_name(rel_text):
                    continue
                if rel_text in seen:
                    skipped_existing[node] += 1
                    continue
                if src.is_symlink():
                    continue
                seen.add(rel_text)
                dest = state.shared_ns_dir / rel
                if dest.exists():
                    # Existing-path-wins already applies to content conflicts. No need
                    # to read both remote files just to distinguish identical bytes.
                    skipped_existing[node] += 1
                else:
                    _publish_file(src, dest)
                    published[node] += 1
        logger.info(
            "compile-cache merged node %s: %d published, %d existing", node, published[node], skipped_existing[node]
        )

    report = {
        "namespace": state.namespace,
        "nodes": [int(d.name[5:]) for d in node_dirs],
        "published": published,
        "skipped_existing": skipped_existing,
        "duration_s": time.monotonic() - start,
    }
    # Renaming releases the staging path immediately. Deleting hundreds of thousands
    # of files on shared storage can take longer than the merge itself.
    archive = state.shared_dir / f".staging.merged.{time.time_ns()}"
    os.rename(state.staging_root, archive)
    report["staging_archive"] = str(archive)
    _write_marker(state.shared_dir / ".merge-report.json", report)
    return report


def _await_staging_nodes(staging_root: Path, num_nodes: int) -> list[Path] | None:
    deadline = time.monotonic() + _STAGING_WAIT_S
    while True:
        if staging_root.is_dir():
            node_dirs = sorted(
                (d for d in staging_root.glob("node_*") if d.name[5:].isdigit()), key=lambda d: int(d.name[5:])
            )
        else:
            node_dirs = []
        if any((d / "FAILED").exists() for d in node_dirs):
            logger.error("a node reported staging failure under %s", staging_root)
            return None
        ready = [d for d in node_dirs if int(d.name[5:]) < num_nodes and (d / "READY").exists()]
        if len(ready) >= num_nodes:
            return ready
        if time.monotonic() > deadline:
            logger.error("expected %d staging READY markers under %s, found %d", num_nodes, staging_root, len(ready))
            return None
        time.sleep(2.0)


def _publish_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
    shutil.copy2(src, tmp)
    os.rename(tmp, dest)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _is_complete_name(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return (
        not name.startswith(".~")
        and not (name.startswith(".") and ".tmp." in name)
        and not any(name.endswith(suffix) for suffix in INCOMPLETE_SUFFIXES)
    )


def _scan_complete_files(root: Path) -> dict[str, tuple[int, int]]:
    """Map every complete file under ``root`` to ``(size, mtime_ns)``; paths are relative."""
    entries = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            if not _is_complete_name(rel) or path.is_symlink():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            entries[rel] = (stat.st_size, stat.st_mtime_ns)
    return entries


def _copy_files(rels, src_root: Path, dest_root: Path) -> list[str]:
    """Copy ``rels`` in parallel; returns the ones that failed (a cache can just rebuild them)."""

    def copy_one(rel: str) -> str | None:
        try:
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_root / rel, dest)
        except OSError as exc:
            logger.warning("compile-cache copy of %s failed (%s); skipping it", rel, exc)
            return rel
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=_COPY_WORKERS) as pool:
        return [rel for rel in pool.map(copy_one, rels) if rel is not None]


def _write_marker(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    os.rename(tmp, path)


def _read_marker(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except ValueError:
        return default


def _node_rank() -> int:
    group_rank = _env_int("GROUP_RANK", -1)
    if group_rank >= 0:
        return group_rank
    local_world = _env_int("LOCAL_WORLD_SIZE", 0)
    if local_world > 0:
        return _env_int("RANK", 0) // local_world
    return 0


def _num_nodes() -> int:
    world, local_world = _env_int("WORLD_SIZE", 1), _env_int("LOCAL_WORLD_SIZE", 0)
    if local_world > 0:
        return max(1, math.ceil(world / local_world))
    return 1


def _repoint_compiler_cache_envs(state: _State) -> None:
    """Ensure no compiler keeps writing outside the node-local cache.

    When `sft.py` runs as a script, activation happens before `xtuner.v1` is imported and
    none of these variables is set. When the module is imported instead (`python -m` or the
    package CLI), `xtuner/v1/__init__.py` has already derived them from the shared root; the
    compilers have not read them yet, so retarget the derived defaults to the local cache.
    An explicit value pointing anywhere else means that compiler would still write to the
    remote filesystem, which local mode rejects.
    """
    for name, subdir in COMPILER_CACHE_DIRS:
        value = os.getenv(name)
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if _is_relative_to(path, state.cache_dir):
            continue
        if path == state.shared_dir / subdir:
            os.environ[name] = str(state.cache_dir / subdir)
            continue
        raise RuntimeError(
            f"{name}={value} is set outside the node-local cache {state.cache_dir}; in local compile-cache "
            f"mode every compiler must write under the node-local directory, so unset {name} "
            "and let xtuner derive it"
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _namespace() -> str:
    """Environment namespace isolating the shared cache across compiler/GPU upgrades."""
    parts = []
    for dist_name in ("torch", "triton", "tilelang"):
        try:
            parts.append(f"{dist_name}{metadata.version(dist_name)}")
        except metadata.PackageNotFoundError:
            continue
    parts.append(_gpu_name())
    return _sanitize("-".join(parts))


def _gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return "nogpu"
    return out[0].strip() if out and out[0].strip() else "nogpu"


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", text).strip("-")


def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manage staged XTuner compiler cache files")
    parser.add_argument("command", choices=["merge", "cleanup"])
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--namespace", default=None, help="required when running without the training GPU")
    parser.add_argument("--num-nodes", type=int)
    args = parser.parse_args(argv)
    _ensure_logging()
    shared_dir = args.shared_dir.expanduser().resolve()
    if args.command == "cleanup":
        for archive in shared_dir.glob(".staging.merged.*"):
            logger.info("compile-cache removing merged staging archive %s", archive)
            # Each node directory can be removed independently; parallelism matters
            # on shared storage with hundreds of thousands of small cache files.
            with concurrent.futures.ThreadPoolExecutor(max_workers=_COPY_WORKERS) as pool:
                list(pool.map(shutil.rmtree, archive.iterdir()))
            archive.rmdir()
        return 0
    if args.num_nodes is None or args.num_nodes < 1:
        parser.error("merge requires --num-nodes >= 1")
    namespace = args.namespace or _namespace()
    if args.namespace is None and namespace.endswith("-nogpu"):
        parser.error("merge without a visible GPU requires --namespace from the training log")
    state = _State(
        shared_dir=shared_dir,
        namespace=namespace,
        run_dir=LOCAL_ROOT,
        local_rank=0,
        global_rank=0,
        node_rank=0,
        num_nodes=args.num_nodes,
    )
    return 0 if merge_staging(state) else 1


if __name__ == "__main__":
    raise SystemExit(main())
