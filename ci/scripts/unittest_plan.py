#!/usr/bin/env python3
"""Map PR changed paths to CI unit-test scope and cluster resources.

Used by .github/workflows/unit_test.yaml (detect_changes). Does not alter which
tests exist — only which paths pytest collects and job GPU sizing.
"""

from __future__ import annotations

import argparse
import os
import sys
# Long-running RL integration / E2E (8-GPU). Skipped on PR unless the PR touches them.
RL_E2E_TEST_PATHS = (
    "tests/rl/test_qwen35_vl_moe_recover_e2e.py",
    "tests/rl/test_qwen35_vl_moe_async_train_2step.py",
    "tests/rl/test_rl_colocate_trainer_integration.py",
)

# Prefix (under repo root) -> pytest directories to run when *all* changes stay in mapped areas.
_CODE_PREFIX_TARGETS: list[tuple[str, tuple[str, ...]]] = [
    ("xtuner/v1/ops/", ("tests/ops", "tests/module")),
    ("xtuner/v1/datasets/", ("tests/datasets",)),
    ("xtuner/v1/loss/", ("tests/loss",)),
    ("xtuner/v1/float8/", ("tests/float8",)),
    ("xtuner/v1/patch/", ("tests/patch",)),
    ("xtuner/v1/profiler/", ("tests/profiler",)),
    ("xtuner/v1/optim/", ("tests/optim",)),
    ("xtuner/v1/train/", ("tests/train", "tests/engine")),
    ("xtuner/v1/model/", ("tests/model", "tests/engine")),
    ("xtuner/v1/module/", ("tests/model", "tests/module", "tests/engine")),
    ("xtuner/v1/config/", ("tests/model", "tests/engine")),
    ("xtuner/v1/engine/", ("tests/engine", "tests/model")),
    ("xtuner/v1/chat_template/", ("tests/chat_template",)),
    ("xtuner/v1/rl/", ("tests/rl",)),
    ("tests/rl/", ("tests/rl",)),
    ("tests/ops/", ("tests/ops",)),
    ("tests/datasets/", ("tests/datasets",)),
    ("tests/model/", ("tests/model", "tests/engine")),
    ("tests/engine/", ("tests/engine", "tests/model")),
    ("tests/module/", ("tests/module",)),
    ("tests/loss/", ("tests/loss",)),
    ("tests/utils/", ("tests/utils",)),
    ("tests/train/", ("tests/train", "tests/engine")),
    ("tests/autotest/", ("tests/autotest",)),
    ("ci/scripts/", ("tests/autotest",)),
    (".github/workflows/unit_test.yaml", ("tests/autotest",)),
]

# Touching these always runs the full suite (cross-cutting).
_FULL_SUITE_PREFIXES = (
    "xtuner/v1/__init__.py",
    "pyproject.toml",
    "setup.cfg",
    "requirements",
    "Dockerfile",
)


def _posix(path: str) -> str:
    return path.replace("\\", "/")


def _only_rl(changed: list[str]) -> bool:
    if not changed:
        return False
    for path in changed:
        p = _posix(path)
        if p.startswith("tests/rl/") or p.startswith("xtuner/v1/rl/"):
            continue
        return False
    return True


def _targets_for_file(path: str) -> tuple[str, ...] | None:
    p = _posix(path)
    for prefix, targets in _CODE_PREFIX_TARGETS:
        if p == prefix.rstrip("/") or p.startswith(prefix):
            return targets
    return None


def _needs_full_suite(changed: list[str]) -> bool:
    for path in changed:
        p = _posix(path)
        for prefix in _FULL_SUITE_PREFIXES:
            if p == prefix or p.startswith(prefix):
                return True
        if p.startswith("xtuner/v1/") and _targets_for_file(p) is None:
            return True
        if p.startswith("tests/") and _targets_for_file(p) is None:
            return True
    return False


def _collect_targets(changed: list[str]) -> set[str]:
    targets: set[str] = set()
    for path in changed:
        mapped = _targets_for_file(path)
        if mapped:
            targets.update(mapped)
    return targets


def _run_rl_e2e(changed: list[str]) -> bool:
    changed_set = {_posix(p) for p in changed}
    if changed_set & set(RL_E2E_TEST_PATHS):
        return True
    rl_e2e_code_hints = (
        "xtuner/v1/rl/health_manager.py",
        "xtuner/v1/rl/rollout/health_manager.py",
        "xtuner/v1/rl/weight_update/",
        "xtuner/v1/rl/rollout/worker_registry.py",
    )
    for path in changed:
        p = _posix(path)
        for hint in rl_e2e_code_hints:
            if p == hint or p.startswith(hint):
                return True
    return False


def _pytest_ignores(changed: list[str], test_target: str) -> str:
    if _run_rl_e2e(changed):
        return ""
    if test_target != "tests" and "tests/rl" not in test_target.split():
        return ""
    parts = [f"--ignore={p}" for p in RL_E2E_TEST_PATHS]
    return " ".join(parts)


def _logical_gpus_for_targets(test_target: str) -> int:
    """GPUs a single pytest process needs if not using on-node sharding."""
    parts = test_target.split()
    if test_target == "tests" or "tests/rl" in parts:
        return 8
    if any(p in parts for p in ("tests/engine", "tests/model", "tests/train")):
        return 8
    if parts == ["tests/autotest"]:
        return 2
    if all(p in ("tests/datasets", "tests/utils", "tests/chat_template", "tests/patch", "tests/profiler") for p in parts):
        return 2
    return 2


# On-node: 8-GPU job, 4 workers × 2 GPUs (0,1 / 2,3 / 4,5 / 6,7).
_GPU_SHARD_WORKERS = 4
_GPUS_PER_SHARD_WORKER = 2
_CLUSTER_GPUS_WHEN_SHARDING = _GPU_SHARD_WORKERS * _GPUS_PER_SHARD_WORKER

# Per-worker sees 2 GPUs (XTUNER_TEST_WORLD_SIZE=2). These files need >2 visible devices
# or ignore XTUNER_TEST_WORLD_SIZE — skipped when sharding; PRs that touch them fall back
# to a single 8-GPU pytest process instead.
_GPU_SHARD_UNSAFE_FILES = frozenset(
    {
        "tests/datasets/test_dataloader.py",
        "tests/loss/test_grpo_loss.py",
        "tests/loss/test_oreal_loss.py",
        "tests/module/dispatcher/test_agrs_all2all.py",
        "tests/module/dispatcher/test_deepep.py",
        "tests/module/dispatcher/test_deepep_expert_tp.py",
        "tests/module/dispatcher/test_torch_all2all.py",
        "tests/module/dispatcher/test_torch_all2all_shared_expert_tp.py",
        "tests/optim/test_muon.py",
        "tests/patch/test_dcp_interleaved_planner.py",
        "tests/utils/test_interleaved_shard.py",
    }
)


def _changed_touches_shard_unsafe(changed: list[str]) -> bool:
    for path in changed:
        if _posix(path) in _GPU_SHARD_UNSAFE_FILES:
            return True
    return False


def _use_gpu_sharding(test_target: str, changed: list[str]) -> bool:
    if _logical_gpus_for_targets(test_target) != 2:
        return False
    if not test_target.split():
        return False
    if _changed_touches_shard_unsafe(changed):
        return False
    return True


def _gpu_shard_ignores() -> str:
    return " ".join(f"--ignore={p}" for p in sorted(_GPU_SHARD_UNSAFE_FILES))


def _wrap_pytest_gpu_shards(pytest_cmd: str) -> str:
    return (
        f"python ci/scripts/run_pytest_gpu_shards.py "
        f"--workers {_GPU_SHARD_WORKERS} "
        f"--gpus-per-worker {_GPUS_PER_SHARD_WORKER} "
        f"-- {pytest_cmd}"
    )


def _cpus_memory(gpus: int) -> tuple[int, str]:
    if gpus >= 8:
        return 120, "800"
    if gpus == 0:
        return 32, "64"
    return 64, "256"


def plan(changed_files: list[str]) -> dict[str, str]:
    changed = [line.strip() for line in changed_files if line.strip()]
    only_rl = _only_rl(changed)

    if not changed or _needs_full_suite(changed):
        test_target = "tests"
    else:
        targets = _collect_targets(changed)
        test_target = "tests" if not targets else " ".join(sorted(targets))

    logical_gpus = _logical_gpus_for_targets(test_target)
    shard = _use_gpu_sharding(test_target, changed)
    if shard:
        gpus = _CLUSTER_GPUS_WHEN_SHARDING
    elif logical_gpus == 2 and _changed_touches_shard_unsafe(changed):
        gpus = 8
    else:
        gpus = logical_gpus
    cpus, memory = _cpus_memory(gpus)

    ignore_parts: list[str] = []
    rl_ignores = _pytest_ignores(changed, test_target)
    if rl_ignores:
        ignore_parts.extend(rl_ignores.split())
    if shard:
        ignore_parts.extend(_gpu_shard_ignores().split())

    pytest_opts = "-ra --durations=25"
    ignore_str = " ".join(ignore_parts)
    if ignore_str:
        pytest_cmd = f"pytest {pytest_opts} {ignore_str} {test_target}"
    else:
        pytest_cmd = f"pytest {pytest_opts} {test_target}"

    if shard:
        pytest_cmd = _wrap_pytest_gpu_shards(pytest_cmd)

    return {
        "only_rl": "true" if only_rl else "false",
        "test_target": test_target,
        "gpu_sharding": "true" if shard else "false",
        "gpus_per_task": str(gpus),
        "cpus_per_task": str(cpus),
        "memory_per_task": memory,
        "pytest_cmd": pytest_cmd,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Append key=value lines for GITHUB_OUTPUT",
    )
    parser.add_argument("changed_files", nargs="*", help="Changed file paths (default: stdin lines)")
    args = parser.parse_args()

    if args.changed_files:
        changed = list(args.changed_files)
    else:
        changed = [line.strip() for line in sys.stdin if line.strip()]

    result = plan(changed)
    if args.github_output:
        out_path = os.environ.get("GITHUB_OUTPUT")
        if not out_path:
            print("GITHUB_OUTPUT is not set", file=sys.stderr)
            sys.exit(1)
        with open(out_path, "a", encoding="utf-8") as fh:
            for key, value in result.items():
                if "\n" in value:
                    fh.write(f"{key}<<EOF\n{value}\nEOF\n")
                else:
                    fh.write(f"{key}={value}\n")
    else:
        for key, value in result.items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
