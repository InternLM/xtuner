"""Suspend and resume NCCL process group backends via PyTorch native APIs.

PyTorch 2.12 exposes ProcessGroup backend methods backed by NCCL
``ncclCommSuspend`` / ``ncclCommResume``. This module keeps the train-side
memory release path on those public backend APIs instead of destroying and
replaying c10d process groups.
"""

import os
from typing import Any

import torch
import torch.distributed as dist


_SUSPENDED_BY_PID: dict[int, list[dict[str, Any]]] = {}


def suspend_nccl_process_groups(*, include_default: bool = False) -> list[dict[str, object]]:
    """Suspend NCCL backends for existing process groups.

    Backend type is usually "nccl", but backend instances and NCCL communicators are owned per ProcessGroup. Releasing
    all train-side NCCL communicator memory therefore requires visiting every relevant NCCL ProcessGroup.

    All ranks must call this in the same order and only after outstanding collectives have completed. By default the
    world/default process group is skipped because it may still be used by distributed control paths.
    """

    if not dist.is_available() or not dist.is_initialized():
        return []

    pid = os.getpid()
    if _SUSPENDED_BY_PID.get(pid):
        return []

    groups = _iter_nccl_process_groups(include_default=include_default)
    if not groups:
        return []

    details: list[dict[str, object]] = []
    records: list[dict[str, Any]] = []
    device = _current_cuda_device()
    try:
        # NCCL suspend requires the communicator to be idle. The caller already
        # invokes this after train offload; synchronize here to catch any
        # remaining async CUDA/NCCL work before touching communicator state.
        torch.cuda.synchronize(device)
    except Exception as exc:
        for group, ranks in groups:
            details.append(
                {
                    "ranks": ranks,
                    "backend": _backend_name(group),
                    "group_name": process_group_name(group),
                    "device": str(device),
                    "error": f"cuda_synchronize {type(exc).__name__}: {exc}",
                }
            )
        return details

    for group, ranks in groups:
        group_name = process_group_name(group)
        backend_name = _backend_name(group)
        try:
            backend = group._get_backend(device)
            _check_backend_api(backend)
            stats_before = _safe_memory_stats(backend)
            backend.suspend()  # type: ignore
            stats_after = _safe_memory_stats(backend)
        except Exception as exc:
            if _is_uninitialized_nccl_comm(exc):
                # Process groups can exist in c10d before their per-device
                # NCCL communicator is lazily initialized. Such groups do not
                # own NCCL GPU memory yet, so there is nothing to suspend.
                details.append(
                    {
                        "ranks": ranks,
                        "backend": backend_name,
                        "group_name": group_name,
                        "device": str(device),
                        "skipped": True,
                        "skip_reason": "nccl_communicator_not_initialized",
                    }
                )
                continue
            details.append(
                {
                    "ranks": ranks,
                    "backend": backend_name,
                    "group_name": group_name,
                    "device": str(device),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        # Resume only the groups that were actually suspended. Skipped or
        # failed groups must not be resumed later.
        records.append({"group": group, "device": device, "group_name": group_name, "ranks": ranks})
        details.append(
            {
                "ranks": ranks,
                "backend": backend_name,
                "group_name": group_name,
                "device": str(device),
                "memory_stats_before": stats_before,
                "memory_stats_after": stats_after,
            }
        )

    _SUSPENDED_BY_PID[pid] = records
    return details


def resume_nccl_process_groups() -> list[dict[str, object]]:
    """Resume NCCL backends previously suspended in this process.

    Resume failures are fatal for the next train phase, so let exceptions propagate instead of turning them into log-
    only errors.
    """

    pid = os.getpid()
    records = _SUSPENDED_BY_PID.get(pid, [])
    if not records:
        return []

    details: list[dict[str, object]] = []
    for record in records:
        group = record["group"]
        device = record["device"]
        group_name = record["group_name"]
        ranks = record["ranks"]
        backend_name = _backend_name(group)
        backend = group._get_backend(device)
        _check_backend_api(backend)
        stats_before = _safe_memory_stats(backend)
        backend.resume()
        stats_after = _safe_memory_stats(backend)

        details.append(
            {
                "ranks": ranks,
                "backend": backend_name,
                "group_name": group_name,
                "device": str(device),
                "memory_stats_before": stats_before,
                "memory_stats_after": stats_after,
            }
        )

    _SUSPENDED_BY_PID.pop(pid, None)
    return details


def nccl_process_group_status() -> dict[str, object]:
    records = _SUSPENDED_BY_PID.get(os.getpid(), [])
    return {"suspended": len(records)}


def process_group_name(group: dist.ProcessGroup) -> str:
    try:
        return str(group.group_name)
    except RuntimeError:
        return "unnamed"


def _iter_nccl_process_groups(*, include_default: bool) -> list[tuple[dist.ProcessGroup, list[int]]]:
    import torch.distributed.distributed_c10d as c10d

    default_group = dist.group.WORLD
    groups: list[tuple[dist.ProcessGroup, list[int]]] = []
    for group, rank_map in list(c10d._world.pg_group_ranks.items()):
        # Keep world/default PG out of the default path. It is a common
        # implicit target for dist.* calls and may be used by control code.
        if group is default_group and not include_default:
            continue
        backend = _backend_name(group)
        if "nccl" not in backend and "cuda" not in backend:
            continue
        ranks = [global_rank for global_rank, _ in sorted(rank_map.items(), key=lambda item: item[1])]
        groups.append((group, ranks))
    return groups


def _backend_name(group: dist.ProcessGroup) -> str:
    try:
        return str(dist.get_backend(group)).lower()
    except Exception:
        try:
            return str(group._get_backend_name()).lower()
        except Exception:
            return "unknown"


def _current_cuda_device() -> torch.device:
    try:
        return torch.device("cuda", torch.cuda.current_device())
    except Exception:
        local_rank = os.getenv("LOCAL_RANK")
        if local_rank is not None:
            return torch.device("cuda", int(local_rank))
        return torch.device("cuda")


def _check_backend_api(backend: object) -> None:
    missing = [name for name in ("suspend", "resume", "memory_stats") if not hasattr(backend, name)]
    if missing:
        raise RuntimeError(
            "NCCL backend does not expose native suspend/resume APIs; "
            f"missing {missing}. PyTorch 2.12+ with NCCL 2.29.7+ is required."
        )


def _safe_memory_stats(backend: object) -> dict[str, int] | None:
    try:
        stats = backend.memory_stats()  # type: ignore
    except Exception:
        return None
    return {str(key): int(value) for key, value in stats.items()}


def _is_uninitialized_nccl_comm(exc: Exception) -> bool:
    return "NCCL communicator not initialized" in str(exc)
