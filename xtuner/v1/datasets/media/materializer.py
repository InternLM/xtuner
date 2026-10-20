import heapq
import threading
import time
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor, wait
from typing import Callable

import torch

from .config import MediaPipelineConfig
from .provider import MediaProvider, create_provider
from .schema import MaterializedMedia, MediaEntry, MediaPlan, TensorSpec, canonical_json


class MediaPipelineError(RuntimeError):
    pass


class MediaBudgetError(MediaPipelineError):
    pass


class MediaCancelledError(MediaPipelineError):
    pass


class MediaShutdownTimeout(MediaPipelineError):
    pass


def validate_cpu_media(media: MaterializedMedia, specs: tuple[TensorSpec, ...]) -> None:
    if not isinstance(media, MaterializedMedia):
        raise TypeError("provider must return MaterializedMedia")
    media.validate(specs)
    for tensor in media.tensors.values():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError("media providers must return CPU tensors")
        if tensor.layout != torch.strided or tensor.requires_grad or not tensor.is_contiguous():
            raise ValueError("provider outputs must be contiguous strided tensors without autograd")
    if media.storage_bytes > sum(spec.nbytes for spec in specs):
        raise MediaBudgetError("provider output retains backing storage larger than its declared compact outputs")
    canonical_json(media.metadata)
    canonical_json(media.metrics)


class MediaMaterializer:
    def __init__(self, config: MediaPipelineConfig):
        self.config = config
        self._lock = threading.Lock()
        self._local = threading.local()
        self._executor: ThreadPoolExecutor | None = None
        self._futures: set[Future] = set()
        self._slots = threading.BoundedSemaphore(config.max_pending_tasks or 1)
        self._closed = False
        self._poisoned = False
        self._thread_prefix = f"xtuner-media-{id(self):x}"

    @property
    def poisoned(self) -> bool:
        with self._lock:
            return self._poisoned

    def reservation_bytes(self, plan: MediaPlan) -> int:
        if not isinstance(plan, MediaPlan):
            raise TypeError("materialization requires an immutable MediaPlan")
        for specs in (plan.output_specs, *(entry.output_specs for entry in plan.entries)):
            if len(specs) > self.config.max_tensor_fields:
                raise MediaBudgetError("media field count exceeds configured limit")
            if any(len(spec.shape) > self.config.max_tensor_rank for spec in specs):
                raise MediaBudgetError("media tensor rank exceeds configured limit")
        if any(
            type(entry.estimated_peak_bytes) is not int or entry.estimated_peak_bytes < 0 for entry in plan.entries
        ):
            raise ValueError("entry peak estimates must be nonnegative integer byte counts")
        retained = sum(spec.nbytes for entry in plan.entries for spec in entry.output_specs)
        scratch = sum(
            heapq.nlargest(self.config.decode_threads, (entry.estimated_peak_bytes for entry in plan.entries))
        )
        return plan.output_bytes + retained + scratch + self.control_bytes(plan)

    def control_bytes(self, plan: MediaPlan | None) -> int:
        descriptor = len(plan.to_json().encode("utf-8")) if plan is not None else 0
        return 4 * descriptor + 4 * self.config.max_header_bytes

    def _provider(self, plan: MediaPlan) -> MediaProvider:
        identity = (plan.provider, plan.provider_version, plan.processor_config_json, plan.processor_fingerprint)
        if getattr(self._local, "identity", None) != identity:
            self._local.provider = create_provider(plan.provider, plan.processor_config_json)
            self._local.identity = identity
        provider = self._local.provider
        if provider.version != plan.provider_version:
            raise ValueError("provider implementation version differs from the frozen media plan")
        capabilities = provider.capabilities
        if self.config.materialization == "lazy" and not capabilities.metadata_exact:
            raise ValueError("lazy materialization requires a metadata-exact provider")
        if self.config.per_image and not capabilities.independent_images:
            raise ValueError("provider has not certified independent per-image processing")
        if not capabilities.worker_local and not capabilities.thread_safe:
            raise ValueError("provider must support worker-local instances or certified thread safety")
        return provider

    def _process(self, plan: MediaPlan, entry: MediaEntry, stop: threading.Event) -> MaterializedMedia:
        if stop.is_set():
            raise MediaCancelledError("media processing cancelled before execution")
        media = self._provider(plan).process(entry)
        validate_cpu_media(media, entry.output_specs)
        if stop.is_set():
            raise MediaCancelledError("media processing cancelled after provider completion")
        return media

    def _assemble(
        self, plan: MediaPlan, results: tuple[MaterializedMedia, ...], stop: threading.Event
    ) -> MaterializedMedia:
        if stop.is_set():
            raise MediaCancelledError("media assembly cancelled")
        media = self._provider(plan).assemble(plan, results)
        return self._finish_assembly(plan, media, stop)

    def _finish_assembly(self, plan: MediaPlan, media: MaterializedMedia, stop: threading.Event) -> MaterializedMedia:
        validate_cpu_media(media, plan.output_specs)
        if len(canonical_json(media.metadata).encode("utf-8")) > self.config.max_header_bytes:
            raise MediaBudgetError("assembled media metadata exceeds bounded transport header")
        if stop.is_set():
            raise MediaCancelledError("media assembly cancelled after provider completion")
        media.metrics = {
            **media.metrics,
            "processed_entries": len(plan.entries),
            "materializer_reserved_bytes": self.reservation_bytes(plan),
        }
        return media

    def _prepare_assembly(
        self, plan: MediaPlan, results: tuple[MaterializedMedia, ...], stop: threading.Event
    ) -> tuple[MaterializedMedia, bool]:
        if stop.is_set():
            raise MediaCancelledError("media assembly cancelled before allocation")
        allocate = getattr(self._provider(plan), "allocate_assembly", None)
        if allocate is None:
            return self._assemble(plan, results, stop), False
        for entry, result in zip(plan.entries, results):
            result.validate(entry.output_specs)
        for spec in plan.output_specs:
            fields = [result.tensors[spec.name] for result in results]
            if (
                spec.concat_dim != 0
                or any(tuple(field.shape[1:]) != spec.shape[1:] for field in fields)
                or any(str(field.dtype).removeprefix("torch.") != spec.dtype for field in fields)
                or sum(field.shape[0] for field in fields) != spec.shape[0]
            ):
                raise ValueError("parallel assembly requires compatible ordered axis-zero fields")
        media = allocate(plan, results)
        validate_cpu_media(media, plan.output_specs)
        retained = {
            tensor.untyped_storage().data_ptr()
            for result in results
            for tensor in result.tensors.values()
            if tensor.numel()
        }
        outputs = [tensor.untyped_storage().data_ptr() for tensor in media.tensors.values() if tensor.numel()]
        if retained.intersection(outputs) or len(outputs) != len(set(outputs)):
            raise ValueError("parallel assembly requires fresh nonoverlapping output storage")
        if stop.is_set():
            raise MediaCancelledError("media assembly cancelled after allocation")
        return media, True

    def _copy_assembly(
        self,
        media: MaterializedMedia,
        result: MaterializedMedia,
        offsets: dict[str, int],
        stop: threading.Event,
        deadline: float,
    ) -> None:
        for name, source in result.tensors.items():
            if stop.is_set():
                raise MediaCancelledError("media assembly copy cancelled")
            self._check_deadline(deadline)
            media.tensors[name].narrow(0, offsets[name], source.shape[0]).copy_(source, non_blocking=False)
        if stop.is_set():
            raise MediaCancelledError("media assembly copy cancelled after execution")
        self._check_deadline(deadline)

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("media materialization absolute deadline exceeded")

    def _submit(self, operation: Callable, arguments: tuple, stop: threading.Event, deadline: float) -> Future:
        while True:
            if stop.is_set():
                raise MediaCancelledError("media submission cancelled")
            self._check_deadline(deadline)
            if self._slots.acquire(timeout=max(0.0, min(0.05, deadline - time.monotonic()))):
                break
        try:
            with self._lock:
                self._check_deadline(deadline)
                if self._closed or self._poisoned or stop.is_set():
                    raise MediaCancelledError("media materializer is closed, poisoned or cancelled")
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self.config.decode_threads, thread_name_prefix=self._thread_prefix
                    )
                future = self._executor.submit(operation, *arguments)
                self._futures.add(future)
        except BaseException:
            self._slots.release()
            raise

        def finished(completed: Future) -> None:
            with self._lock:
                self._futures.discard(completed)
            self._slots.release()

        future.add_done_callback(finished)
        return future

    @staticmethod
    def _observe(futures) -> None:
        for future in futures:
            if future.done() and not future.cancelled():
                future.exception()

    def materialize(self, plan: MediaPlan, *, cancel_event: threading.Event | None = None) -> MaterializedMedia:
        if not self.config.enabled:
            raise MediaPipelineError("materialization requested while media pipeline is disabled")
        deadline = time.monotonic() + self.config.materialization_timeout_seconds
        required = self.reservation_bytes(plan)
        if required > self.config.max_inflight_bytes:
            raise MediaBudgetError(f"media plan requires {required} reserved bytes, exceeding its hard limit")
        stop = cancel_event if cancel_event is not None else threading.Event()
        self._check_deadline(deadline)
        submitted: list[Future] = []
        results: dict[int, MaterializedMedia] = {}
        pending: dict[Future, int] = {}
        window = min(self.config.decode_threads, self.config.max_pending_tasks, len(plan.entries))
        try:
            for ordinal in range(window):
                future = self._submit(self._process, (plan, plan.entries[ordinal], stop), stop, deadline)
                submitted.append(future)
                pending[future] = ordinal
            following = window
            while pending:
                if stop.is_set():
                    raise MediaCancelledError("materialization cancelled during ordered consumption")
                completed, _not_completed = wait(
                    pending, timeout=max(0.0, deadline - time.monotonic()), return_when=FIRST_COMPLETED
                )
                self._check_deadline(deadline)
                for future in sorted(completed, key=pending.__getitem__):
                    ordinal = pending.pop(future)
                    results[ordinal] = future.result()
                    self._check_deadline(deadline)
                while following < len(plan.entries) and len(pending) < window:
                    future = self._submit(self._process, (plan, plan.entries[following], stop), stop, deadline)
                    submitted.append(future)
                    pending[future] = following
                    following += 1
            ordered_results = tuple(results[ordinal] for ordinal in range(len(plan.entries)))
            future = self._submit(self._prepare_assembly, (plan, ordered_results, stop), stop, deadline)
            submitted.append(future)
            prepared = future.result(timeout=max(0.0, deadline - time.monotonic()))
            self._check_deadline(deadline)
            result, parallel = prepared
            if parallel:
                offsets = {spec.name: 0 for spec in plan.output_specs}
                following = 0
                while following < len(ordered_results) or pending:
                    if stop.is_set():
                        raise MediaCancelledError("media assembly cancelled during bounded copying")
                    self._check_deadline(deadline)
                    while following < len(ordered_results) and len(pending) < window:
                        entry_result = ordered_results[following]
                        future = self._submit(
                            self._copy_assembly, (result, entry_result, dict(offsets), stop, deadline), stop, deadline
                        )
                        submitted.append(future)
                        pending[future] = following
                        for name, tensor in entry_result.tensors.items():
                            offsets[name] += tensor.shape[0]
                        following += 1
                    completed, _not_completed = wait(
                        pending, timeout=max(0.0, deadline - time.monotonic()), return_when=FIRST_COMPLETED
                    )
                    self._check_deadline(deadline)
                    for future in completed:
                        pending.pop(future)
                        future.result()
                result = self._finish_assembly(plan, result, stop)
            self._check_deadline(deadline)
            return result
        except BaseException as error:
            stop.set()
            for future in submitted:
                future.cancel()
            done, unfinished = wait(submitted, timeout=self.config.shutdown_timeout_seconds)
            self._observe(done)
            if unfinished:
                with self._lock:
                    self._poisoned = True
                raise MediaShutdownTimeout(
                    "native media tasks did not stop; launcher must terminate this run"
                ) from error
            if isinstance(error, CancelledError):
                raise MediaCancelledError("pending media task cancelled") from error
            raise
        finally:
            results.clear()
            pending.clear()
            submitted.clear()
            future = completed = _not_completed = done = unfinished = None

    def close(self) -> None:
        self._close_until(time.monotonic() + self.config.shutdown_timeout_seconds)

    def _close_until(self, deadline: float) -> None:
        with self._lock:
            self._closed = True
            executor = self._executor
            futures = tuple(self._futures)
        if executor is None:
            return
        executor.shutdown(wait=False, cancel_futures=True)
        done, unfinished = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
        self._observe(done)
        for thread in tuple(threading.enumerate()):
            if thread.name.startswith(self._thread_prefix + "_"):
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if thread.is_alive():
                    unfinished = True
        if unfinished:
            with self._lock:
                self._poisoned = True
            raise MediaShutdownTimeout("media executor exceeded shutdown deadline; launcher termination required")
