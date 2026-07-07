from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in _TRUE_VALUES


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / 1024**3


class MemoryStageProfiler:
    """Env-gated CUDA allocator profiler for coarse training-stage
    diagnosis."""

    def __init__(self) -> None:
        self.enabled = _env_bool("XTUNER_MEMORY_STAGE_PROFILE", False)
        self._history_started = False
        self._last_trace_len = 0
        self._rank: int | None = None
        self._output_file: Path | None = None
        self._last_allocated: int | None = None
        self._topk = int(os.environ.get("XTUNER_MEMORY_STAGE_PROFILE_TOPK", "12"))
        self._max_entries = int(os.environ.get("XTUNER_MEMORY_STAGE_PROFILE_MAX_ENTRIES", "2000000"))
        self._repo_root = Path.cwd().resolve()
        self._dump_snapshot = _env_bool("XTUNER_MEMORY_STAGE_PROFILE_DUMP_SNAPSHOT", False)

    def capture(self, stage: str, *, step: int | None = None, micro_batch: int | None = None) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        if not self._should_profile_rank():
            return

        self._ensure_started()
        torch.cuda.synchronize()
        snapshot = torch.cuda.memory._snapshot()
        memory_stats = torch.cuda.memory_stats()

        active_summary = self._summarize_active_blocks(snapshot)
        interval_summary = self._summarize_interval_allocations(snapshot)
        allocated = int(memory_stats.get("allocated_bytes.all.current", torch.cuda.memory_allocated()))
        reserved = int(memory_stats.get("reserved_bytes.all.current", torch.cuda.memory_reserved()))

        record = {
            "rank": self._rank,
            "device": torch.cuda.current_device(),
            "step": step,
            "micro_batch": micro_batch,
            "stage": stage,
            "allocated_gib": _bytes_to_gib(allocated),
            "reserved_gib": _bytes_to_gib(reserved),
            "max_allocated_gib": _bytes_to_gib(
                int(memory_stats.get("allocated_bytes.all.peak", torch.cuda.max_memory_allocated()))
            ),
            "max_reserved_gib": _bytes_to_gib(
                int(memory_stats.get("reserved_bytes.all.peak", torch.cuda.max_memory_reserved()))
            ),
            "delta_allocated_from_prev_gib": None
            if self._last_allocated is None
            else _bytes_to_gib(allocated - self._last_allocated),
            "active": active_summary,
            "interval": interval_summary,
        }
        self._last_allocated = allocated

        assert self._output_file is not None
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        with self._output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        if self._dump_snapshot:
            self._dump_snapshot_file(snapshot, stage=stage, step=step, micro_batch=micro_batch)

    def _should_profile_rank(self) -> bool:
        if self._rank is None:
            self._rank = _rank()
        ranks = os.environ.get("XTUNER_MEMORY_STAGE_PROFILE_RANKS", "0").strip().lower()
        if ranks == "all":
            return True
        return self._rank in {int(rank.strip()) for rank in ranks.split(",") if rank.strip()}

    def _ensure_started(self) -> None:
        if self._output_file is None:
            assert self._rank is not None
            output_dir_env = os.environ.get("XTUNER_MEMORY_STAGE_PROFILE_DIR")
            if output_dir_env:
                output_dir = Path(output_dir_env)
            else:
                output_dir = (
                    Path(os.environ.get("WORK_DIR", "work_dirs/memory_stage_profile")) / "memory_stage_profile"
                )
            self._output_file = output_dir / f"rank{self._rank}.jsonl"

        if self._history_started:
            return

        # This is intentionally opt-in: memory history synchronizes and walks allocator state.
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=self._max_entries,
            clear_history=True,
        )
        if self._output_file.exists():
            self._output_file.unlink()
        self._history_started = True

    def _dump_snapshot_file(
        self,
        snapshot: dict[str, Any],
        *,
        stage: str,
        step: int | None,
        micro_batch: int | None,
    ) -> None:
        assert self._output_file is not None
        snapshot_dir = self._output_file.parent / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        step_name = "none" if step is None else str(step)
        micro_name = "none" if micro_batch is None else str(micro_batch)
        path = snapshot_dir / f"rank{self._rank}_step{step_name}_micro{micro_name}_{stage}.pickle"
        with path.open("wb") as f:
            pickle.dump(snapshot, f)

    def _summarize_active_blocks(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        total_active = 0
        total_requested = 0
        total_inactive = 0
        by_stack: dict[str, dict[str, Any]] = {}

        for segment in snapshot.get("segments", []):
            for block in segment.get("blocks", []):
                state = block.get("state", "")
                size = int(block.get("size", 0))
                if state.startswith("active"):
                    requested = int(block.get("requested_size", size))
                    total_active += size
                    total_requested += requested
                    self._accumulate_stack(by_stack, size, requested, block.get("frames") or [])
                elif state == "inactive":
                    total_inactive += size

        return {
            "active_gib": _bytes_to_gib(total_active),
            "requested_gib": _bytes_to_gib(total_requested),
            "inactive_gib": _bytes_to_gib(total_inactive),
            "top_stacks": self._top_stacks(by_stack),
        }

    def _summarize_interval_allocations(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        device = torch.cuda.current_device()
        traces = snapshot.get("device_traces", [])
        trace = traces[device] if device < len(traces) else []
        if len(trace) < self._last_trace_len:
            self._last_trace_len = 0
        interval_events = trace[self._last_trace_len :]
        self._last_trace_len = len(trace)

        alloc_by_stack: dict[str, dict[str, Any]] = {}
        segment_by_stack: dict[str, dict[str, Any]] = {}
        alloc_bytes = 0
        segment_alloc_bytes = 0
        free_requested_bytes = 0

        for event in interval_events:
            action = event.get("action")
            size = int(event.get("size", 0))
            frames = event.get("frames") or []
            if action == "alloc":
                alloc_bytes += size
                self._accumulate_stack(alloc_by_stack, size, size, frames)
            elif action == "segment_alloc":
                segment_alloc_bytes += size
                self._accumulate_stack(segment_by_stack, size, size, frames)
            elif action == "free_requested":
                free_requested_bytes += size

        return {
            "event_count": len(interval_events),
            "alloc_gib": _bytes_to_gib(alloc_bytes),
            "segment_alloc_gib": _bytes_to_gib(segment_alloc_bytes),
            "free_requested_gib": _bytes_to_gib(free_requested_bytes),
            "top_alloc_stacks": self._top_stacks(alloc_by_stack),
            "top_segment_alloc_stacks": self._top_stacks(segment_by_stack),
        }

    def _accumulate_stack(
        self,
        by_stack: dict[str, dict[str, Any]],
        size: int,
        requested_size: int,
        frames: list[dict[str, Any]],
    ) -> None:
        key = self._select_frame_key(frames)
        item = by_stack.setdefault(
            key,
            {
                "frame": key,
                "stack": self._format_stack(frames),
                "bytes": 0,
                "requested_bytes": 0,
                "count": 0,
            },
        )
        item["bytes"] += size
        item["requested_bytes"] += requested_size
        item["count"] += 1

    def _top_stacks(self, by_stack: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        records = sorted(by_stack.values(), key=lambda item: item["bytes"], reverse=True)[: self._topk]
        return [
            {
                "frame": item["frame"],
                "stack": item["stack"],
                "gib": _bytes_to_gib(item["bytes"]),
                "requested_gib": _bytes_to_gib(item["requested_bytes"]),
                "count": item["count"],
            }
            for item in records
        ]

    def _select_frame_key(self, frames: list[dict[str, Any]]) -> str:
        for frame in frames:
            filename = str(frame.get("filename", ""))
            if self._is_project_frame(filename):
                return self._format_frame(frame)
        if frames:
            return self._format_frame(frames[0])
        return "<unknown>"

    def _format_stack(self, frames: list[dict[str, Any]]) -> list[str]:
        return [self._format_frame(frame) for frame in frames[:8]]

    def _format_frame(self, frame: dict[str, Any]) -> str:
        filename = str(frame.get("filename", ""))
        line = frame.get("line", 0)
        name = frame.get("name", "")
        try:
            filename = str(Path(filename).resolve().relative_to(self._repo_root))
        except (OSError, ValueError):
            pass
        return f"{filename}:{line}:{name}"

    def _is_project_frame(self, filename: str) -> bool:
        try:
            Path(filename).resolve().relative_to(self._repo_root)
            return True
        except (OSError, ValueError):
            return False


_MEMORY_STAGE_PROFILER: MemoryStageProfiler | None = None


def get_memory_stage_profiler() -> MemoryStageProfiler:
    global _MEMORY_STAGE_PROFILER
    if _MEMORY_STAGE_PROFILER is None:
        _MEMORY_STAGE_PROFILER = MemoryStageProfiler()
    return _MEMORY_STAGE_PROFILER
