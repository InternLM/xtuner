"""Opt-in training progress diagnostics, separate from performance
profiling."""

import faulthandler
import functools
import inspect
import itertools
import json
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiagnosticsConfig(BaseModel):
    """Configure bounded, per-rank training diagnostics.

    Step bounds are inclusive global optimizer steps, including resumed steps. Disabled diagnostics install no
    wrappers, tensor hooks or CUDA events.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    level: Literal["basic", "detailed"] = "basic"
    start_step: int = Field(default=1, ge=1)
    end_step: int | None = Field(default=None, ge=1)
    ranks: list[int] | None = None
    output_dir: Path | None = None
    stack_dump_interval_s: int = Field(default=0, ge=0)
    cuda_events: bool = False
    max_pending_cuda_events: int = Field(default=4096, ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "DiagnosticsConfig":
        """Validate the recording window and rank selection."""
        if self.end_step is not None and self.end_step < self.start_step:
            raise ValueError("end_step must be >= start_step")
        if self.ranks is not None and any(rank < 0 for rank in self.ranks):
            raise ValueError("ranks must contain non-negative global ranks")
        if self.cuda_events and self.level != "detailed":
            raise ValueError("cuda_events requires level='detailed'")
        return self


class TrainingDiagnostics:
    """Record CPU boundaries and optional nonblocking CUDA stream events.

    Args:
        config (DiagnosticsConfig): Recording settings.
        output_dir (Path): Default experiment-local output directory.
        rank (int): Global rank.
    """

    def __init__(self, config: DiagnosticsConfig, output_dir: Path, rank: int) -> None:
        self.config = config
        self.rank = rank
        self.enabled = config.enabled and (config.ranks is None or rank in config.ranks)
        self.directory = config.output_dir or output_dir
        self.active = False
        self.step = 0
        self.phase = "unknown"
        self.micro = 0
        self._local = threading.local()
        self._ids = itertools.count(1)
        self._file: Any = None
        self._lock = threading.RLock()
        self._pending: deque[tuple[Any, dict[str, Any]]] = deque()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._patches: list[tuple[Any, str, Any, bool]] = []
        self._patched: set[tuple[int, str]] = set()
        self._installed = False
        self._closed = False
        self._stack_armed = False
        self._overflow = False
        self._device: int | None = None
        self._hooks: list[Any] = []

    def set_step(self, step: int) -> None:
        """Select the global step and reset the optional Python stack timer."""
        if not self.enabled:
            return
        if self._stack_armed:
            faulthandler.cancel_dump_traceback_later()
            self._stack_armed = False
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self.step = step
        self.micro = 0
        self.active = step >= self.config.start_step and (self.config.end_step is None or step <= self.config.end_step)
        if self.active and self.config.stack_dump_interval_s:
            faulthandler.dump_traceback_later(self.config.stack_dump_interval_s, repeat=True)
            self._stack_armed = True

    def mark(self, phase: str, micro: int | None = None) -> None:
        """Record a CPU progress marker without synchronizing the GPU."""
        if not self.enabled:
            return
        self.phase = phase
        if micro is not None:
            self.micro = micro
        if self.active:
            self._emit("stage", **self._metadata())

    @contextmanager
    def span(self, operation: str, scope: str | None = None) -> Iterator[dict[str, Any] | None]:
        """Record a detailed call boundary on the calling thread."""
        if not self.active or self.config.level != "detailed":
            yield None
            return
        old = getattr(self._local, "scope", "unattributed")
        if scope is not None:
            self._local.scope = scope
        ctx = {**self._metadata(), "call": next(self._ids), "operation": operation}
        self._emit("cpu_begin", **ctx)
        self._event("before", ctx)
        try:
            yield ctx
        except BaseException as exc:
            self._emit("cpu_error", **ctx, error=repr(exc))
            raise
        else:
            self._emit("cpu_end", **ctx)
            self._event("after_current_stream", ctx)
        finally:
            self._local.scope = old

    def install(self, model: torch.nn.Module) -> None:
        """Install detailed wrappers once, preserving bound compiled
        methods."""
        if not self.enabled or self.config.level != "detailed" or self._installed:
            return
        self._installed = True
        # Use XTuner module families, including subclasses, rather than GLM class names.
        families = ("xtuner.v1.module.decoder_layer", "xtuner.v1.module.attention")
        for path, module in model.named_modules():
            if any(cls.__module__.startswith(families) for cls in type(module).__mro__):
                original = module.forward
                wrapped = self._wrap(original, "module.forward", scope=path, gradients=True)
                self._patch(module, "forward", torch.compiler.disable(wrapped))
            dispatcher = getattr(module, "dispatcher", None)
            if dispatcher is not None:
                for name in (
                    "dispatch_preprocess",
                    "dispatch",
                    "dispatch_postprocess",
                    "combine_preprocess",
                    "combine",
                    "combine_postprocess",
                ):
                    if hasattr(dispatcher, name):
                        self._patch(
                            dispatcher,
                            name,
                            self._wrap(getattr(dispatcher, name), "EP." + name, scope=path + ".dispatcher"),
                        )
        from xtuner.v1.utils.activation_offload import OffloadManager, SwapTensor

        for cls, names in (
            (
                SwapTensor,
                ("launch_d2h", "prefetch_launch_h2d", "launch_h2d", "wait_d2h_finished", "wait_h2d_finished", "load"),
            ),
            (OffloadManager, ("get_or_create_pin_memory", "clear_step", "get")),
        ):
            for name in names:
                if hasattr(cls, name):
                    self._patch(cls, name, self._wrap(getattr(cls, name), "offload." + name))
        # Patch already-loaded aliases too: modules commonly use `from ... import fn`.
        targets: dict[Callable[..., Any], str] = {}
        functions = {
            "ulysses_all_to_all",
            "gather_for_sequence_parallel",
            "sequence_causal_conv1d",
            "sequence_chunk_kda",
            "redistribute_indexer_queries",
        }
        modules = [m for name, m in list(sys.modules.items()) if name.startswith("xtuner.v1.") and m is not None]
        for module in modules:
            for name, value in list(vars(module).items()):
                if name in functions and inspect.isfunction(value):
                    targets[value] = "CP." + name
        replacements = {
            fn: torch.compiler.disable(self._wrap(fn, name, gradients=True)) for fn, name in targets.items()
        }
        seen_classes: set[type[Any]] = set()
        for module in modules:
            for name, value in list(vars(module).items()):
                if inspect.isfunction(value) and value in replacements:
                    self._patch(module, name, replacements[value])
                if (
                    inspect.isclass(value)
                    and value not in seen_classes
                    and issubclass(value, torch.autograd.Function)
                    and value.__module__.startswith(
                        ("xtuner.v1.module.dispatcher.deepep", "xtuner.v1.ops.kda", "xtuner.v1.ops.sparse_mla")
                    )
                ):
                    seen_classes.add(value)
                    for method in ("forward", "backward"):
                        fn = getattr(value, method)
                        self._patch(value, method, staticmethod(self._wrap_autograd(fn, value.__name__, method)))
        # DeepEP uses a stream outside ProcessGroupNCCL; do not infer its completion
        # from the caller's current stream or Flight Recorder.
        deep_ep = sys.modules.get("deep_ep")
        buffer_type = getattr(deep_ep, "Buffer", None)
        if buffer_type is not None and hasattr(buffer_type, "get_comm_stream"):
            for name in ("dispatch", "combine"):
                if hasattr(buffer_type, name):
                    self._patch(buffer_type, name, self._wrap_buffer(getattr(buffer_type, name), name))
        self._emit("installed", rank=self.rank, patched_methods=len(self._patches))

    def close(self) -> None:
        """Restore methods and close logs without waiting for GPU
        completion."""
        if self._closed:
            return
        self._closed = True
        self.active = False
        if self._stack_armed:
            faulthandler.cancel_dump_traceback_later()
            self._stack_armed = False
        self._stop.set()
        # The poller queries events only; it never waits for a CUDA stream.
        if self._thread is not None:
            self._thread.join(timeout=2)
        for target, name, original, owned in reversed(self._patches):
            if owned:
                setattr(target, name, original)
            else:
                delattr(target, name)
        self._patches.clear()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        with self._lock:
            if self._pending:
                self._emit(
                    "pending_at_close",
                    count=len(self._pending),
                    note="Unpolled events at exit are not proof of a GPU hang",
                )
            self._pending.clear()
            if self._file is not None:
                self._file.close()
                self._file = None

    def _metadata(self) -> dict[str, Any]:
        phase = self.phase
        if phase.startswith("backward"):
            from xtuner.v1.model.utils.checkpointing import is_checkpoint_replay

            if is_checkpoint_replay():
                phase = "recompute"
        return {
            "rank": self.rank,
            "step": self.step,
            "micro": self.micro,
            "phase": phase,
            "scope": getattr(self._local, "scope", "unattributed"),
            "grad_enabled": torch.is_grad_enabled(),
        }

    def _emit(self, kind: str, **fields: Any) -> None:
        with self._lock:
            if self._closed and kind != "pending_at_close":
                return
            if self._file is None:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._file = (self.directory / f"rank{self.rank}.jsonl").open("a", buffering=1)
            record = {
                "kind": kind,
                "wall_ns": time.time_ns(),
                "mono_ns": time.monotonic_ns(),
                "host": socket.gethostname(),
                **fields,
            }
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()

    def _patch(self, target: Any, name: str, replacement: Any) -> None:
        key = (id(target), name)
        if key in self._patched:
            return
        self._patched.add(key)
        owned = name in vars(target)
        original = inspect.getattr_static(target, name)
        self._patches.append((target, name, original, owned))
        setattr(target, name, replacement)

    def _wrap(
        self, fn: Callable[..., Any], name: str, scope: str | None = None, gradients: bool = False
    ) -> Callable[..., Any]:
        signature = inspect.signature(fn)

        # Do not copy _torchdynamo_orig_callable: it may be an unbound class method.
        @functools.wraps(fn, updated=())
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self.active:
                return fn(*args, **kwargs)
            with self.span(name, scope) as ctx:
                assert ctx is not None
                bound = signature.bind_partial(*args, **kwargs).arguments
                for key in ("mesh", "group", "process_group"):
                    group = bound.get(key)
                    if group is not None:
                        if hasattr(group, "get_group"):
                            group = group.get_group()
                        if isinstance(group, torch.distributed.ProcessGroup):
                            self._emit(
                                "group",
                                **ctx,
                                group_name=group.group_name,
                                ranks=torch.distributed.get_process_group_ranks(group),
                            )
                if gradients:
                    self._grad_hooks((args, kwargs), ctx, "grad_input_ready")
                result = fn(*args, **kwargs)
                if gradients:
                    self._grad_hooks(result, ctx, "grad_output_ready")
                for key in ("stream", "h2dstream"):
                    if bound.get(key) is not None:
                        self._event("after_copy_stream", ctx, bound[key])
                return result

        return wrapped

    def _wrap_buffer(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        @functools.wraps(fn, updated=())
        def wrapped(buffer: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.active:
                return fn(buffer, *args, **kwargs)
            with self.span("EP.buffer." + name) as ctx:
                result = fn(buffer, *args, **kwargs)
                assert ctx is not None
                self._event("deepep_comm_stream", ctx, buffer.get_comm_stream())
                return result

        return wrapped

    def _wrap_autograd(self, fn: Callable[..., Any], name: str, method: str) -> Callable[..., Any]:
        @functools.wraps(fn, updated=())
        def wrapped(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.active:
                return fn(ctx, *args, **kwargs)
            if method == "forward":
                ctx._xtuner_diagnostic_scope = getattr(self._local, "scope", "unattributed")
            scope = getattr(ctx, "_xtuner_diagnostic_scope", "unattributed")
            with self.span(name + "." + method, scope):
                return fn(ctx, *args, **kwargs)

        return wrapped

    def _grad_hooks(self, value: Any, ctx: dict[str, Any], label: str) -> None:
        for tensor in self._tensors(value):
            if not tensor.requires_grad:
                continue

            def hook(grad: torch.Tensor, ctx: dict[str, Any] = ctx, label: str = label) -> None:
                if self.active:
                    fields = {**ctx, "phase": self._metadata()["phase"], "boundary": label}
                    self._emit("gradient_boundary", **fields)
                    self._event(label, fields)

            self._hooks.append(tensor.register_hook(hook))

    def _tensors(self, value: Any, depth: int = 0) -> Iterator[torch.Tensor]:
        if depth > 3:
            return
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from self._tensors(item, depth + 1)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from self._tensors(item, depth + 1)

    def _event(self, label: str, ctx: dict[str, Any], stream: Any = None) -> None:
        if not self.config.cuda_events or not torch.cuda.is_initialized():
            return
        with self._lock:
            if len(self._pending) >= self.config.max_pending_cuda_events:
                if not self._overflow:
                    self._emit("event_capacity_reached", limit=self.config.max_pending_cuda_events)
                    self._overflow = True
                return
            if self._device is None:
                self._device = torch.cuda.current_device()
            stream = torch.cuda.current_stream() if stream is None else stream
            event = torch.cuda.Event(enable_timing=False)
            event.record(stream)
            fields = {**ctx, "label": label, "stream": stream.cuda_stream}
            self._pending.append((event, fields))
            self._emit("gpu_event_recorded", **fields)
            if self._thread is None:
                self._thread = threading.Thread(target=self._poll, name="xtuner-diagnostics", daemon=True)
                self._thread.start()

    def _poll(self) -> None:
        assert self._device is not None
        torch.cuda.set_device(self._device)
        last_heartbeat = time.monotonic()
        while not self._stop.wait(1):
            with self._lock:
                pending = list(self._pending)
            completed: set[int] = set()
            for event, fields in pending:
                if self._stop.is_set():
                    return
                try:
                    done = event.query()
                except RuntimeError as exc:
                    self._emit("gpu_query_error", **fields, error=repr(exc))
                    completed.add(id(event))
                    continue
                if done:
                    self._emit("gpu_event_completed", **fields)
                    completed.add(id(event))
            with self._lock:
                self._pending = deque((e, f) for e, f in self._pending if id(e) not in completed)
                if time.monotonic() - last_heartbeat >= 30:
                    self._emit(
                        "heartbeat",
                        pending=len(self._pending),
                        oldest=[fields for _, fields in list(self._pending)[:8]],
                    )
                    last_heartbeat = time.monotonic()
