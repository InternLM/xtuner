import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field

from .config import MediaPipelineConfig
from .materializer import (
    MediaBudgetError,
    MediaCancelledError,
    MediaMaterializer,
    MediaPipelineError,
    MediaShutdownTimeout,
)
from .schema import MaterializedMedia, MediaPlan
from .transport import MediaTransport, MediaTransportError


@dataclass
class _Prefetch:
    key: str
    fingerprint: str
    reserved_bytes: int
    stop: threading.Event = field(default_factory=threading.Event)
    future: Future = field(default_factory=Future)
    thread: threading.Thread | None = None


class MediaRuntime:
    """Own logical reservations for active consume scopes and one CPU lookahead.

    Callers must keep each scope open through backward and relevant offload work.
    Storage aliases retained past scope exit need external admission and ownership;
    this runtime does not measure their liveness or the allocator's physical frees.
    Integrations without an audited safe release point must reject that combination.
    """

    def __init__(self, config: MediaPipelineConfig):
        self.config = config
        self._owner = threading.get_ident()
        self._materializer = MediaMaterializer(config)
        self._transport = MediaTransport(config)
        self._pending: _Prefetch | None = None
        self._active: set[str] = set()
        self._host_reserved = 0
        self._transport_reserved = 0
        self._closed = False
        self._resources_closed = False
        self._poisoned = False
        self._counters = {
            "prefetch_accepted": 0,
            "prefetch_skipped": 0,
            "prefetch_rejected": 0,
            "prefetch_cancelled": 0,
            "prefetch_errors_observed": 0,
            "consumed": 0,
            "peak_host_reserved_bytes": 0,
            "peak_transport_reserved_bytes": 0,
        }

    @property
    def metrics(self) -> dict[str, int]:
        return {
            **self._counters,
            "host_reserved_bytes": self._host_reserved,
            "transport_reserved_bytes": self._transport_reserved,
            "active_consumptions": len(self._active),
            "lookahead_groups": int(self._pending is not None),
            "poisoned": int(self._poisoned or self._materializer.poisoned),
            "reservation_scope_only": 1,
        }

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise MediaPipelineError("media scheduling and collectives must run on their training owner thread")

    def _check_open(self) -> None:
        self._check_owner()
        if self._closed or self._poisoned or self._materializer.poisoned:
            raise MediaPipelineError("media runtime is closed or poisoned")

    def _add_host(self, amount: int) -> None:
        if self._host_reserved + amount > self.config.max_inflight_bytes:
            raise MediaBudgetError("current plus lookahead reservations exceed host media budget")
        self._host_reserved += amount
        self._counters["peak_host_reserved_bytes"] = max(
            self._counters["peak_host_reserved_bytes"], self._host_reserved
        )

    def prefetch(self, plan: MediaPlan | None, key: str, is_producer: bool) -> bool:
        """Decline invalid lookahead without interrupting the current SP group.

        Consume performs authoritative plan validation with shared SP status.
        Owner, closed and poisoned runtime errors remain fatal to the caller.
        """
        self._check_open()
        if not self.config.enabled or not self.config.prefetch_batches or plan is None or not is_producer:
            self._counters["prefetch_skipped"] += 1
            return False
        if self._pending is not None:
            self._counters["prefetch_skipped"] += 1
            return False
        try:
            if not isinstance(key, str) or not key or len(key.encode("utf-8")) > self.config.max_header_bytes:
                raise ValueError("prefetch requires a bounded nonempty consumption key")
            required = self._materializer.reservation_bytes(plan)
            fingerprint = plan.fingerprint
        except (MediaBudgetError, TypeError, ValueError):
            self._counters["prefetch_rejected"] += 1
            return False
        if required > self.config.max_inflight_bytes:
            self._counters["prefetch_rejected"] += 1
            return False
        if key in self._active or self._host_reserved + required > self.config.max_inflight_bytes:
            self._counters["prefetch_skipped"] += 1
            return False
        job = _Prefetch(key=key, fingerprint=fingerprint, reserved_bytes=required)
        self._add_host(required)
        self._pending = job

        def materialize(job: _Prefetch) -> None:
            try:
                result = self._materializer.materialize(plan, cancel_event=job.stop)
            except BaseException as error:
                job.future.set_exception(error)
            else:
                job.future.set_result(result)
            finally:
                del job

        job.thread = threading.Thread(
            target=materialize, args=(job,), name=f"xtuner-media-lookahead-{id(self):x}", daemon=False
        )
        try:
            job.thread.start()
        except BaseException:
            self._pending = None
            self._host_reserved -= required
            raise
        self._counters["prefetch_accepted"] += 1
        return True

    def _join(self, job: _Prefetch, deadline: float) -> None:
        job.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if job.thread.is_alive():
            self._poisoned = True
            raise MediaShutdownTimeout("CPU lookahead did not join; launcher must terminate this run")

    def cancel_pending(self) -> None:
        self._check_owner()
        self._cancel_pending_until(time.monotonic() + self.config.shutdown_timeout_seconds)

    def _cancel_pending_until(self, deadline: float) -> None:
        job = self._pending
        if job is None:
            return
        try:
            job.stop.set()
            self._join(job, deadline)
            try:
                job.future.result()
            except MediaCancelledError:
                pass
            except BaseException:
                self._counters["prefetch_errors_observed"] += 1
            if self._materializer.poisoned:
                self._poisoned = True
                raise MediaShutdownTimeout("lookahead has unjoined native tasks; reservation remains charged")
            self._pending = None
            self._host_reserved -= job.reserved_bytes
            self._counters["prefetch_cancelled"] += 1
        finally:
            del job

    @contextmanager
    def consume(self, plan: MediaPlan | None, key: str, sp_mesh=None, device="cpu"):
        """Lease a group; caller-owned aliases outside the scope are not budgeted."""
        self._check_open()
        if not self.config.enabled:
            if plan is not None:
                raise MediaPipelineError("unresolved media plan supplied to disabled runtime")
            yield None
            return
        host_charge = 0
        control_charge = 0
        transport_charge = 0
        delivered = None
        registered = False

        def produce() -> MaterializedMedia:
            nonlocal host_charge
            job = self._pending
            try:
                if job is not None and job.key == key:
                    if job.fingerprint != plan.fingerprint:
                        self.cancel_pending()
                        raise ValueError("prefetched plan fingerprint changed before consumption")
                    try:
                        result = job.future.result(timeout=self.config.materialization_timeout_seconds)
                        self._join(job, time.monotonic() + self.config.shutdown_timeout_seconds)
                    except BaseException:
                        self.cancel_pending()
                        raise
                    self._pending = None
                    host_charge += job.reserved_bytes
                    return result
                total = self._materializer.reservation_bytes(plan)
                if total > self.config.max_inflight_bytes:
                    raise MediaBudgetError("current plan exceeds the hard host media limit")
                required = total - control_charge
                if self._host_reserved + required > self.config.max_inflight_bytes and self._pending is not None:
                    self.cancel_pending()
                self._add_host(required)
                host_charge += required
                return self._materializer.materialize(plan)
            finally:
                del job

        def prepare_control(is_producer: bool, reserve_control: bool) -> None:
            nonlocal host_charge, control_charge, registered
            if not isinstance(key, str) or not key or len(key.encode("utf-8")) > self.config.max_header_bytes:
                raise ValueError("consumption requires a bounded nonempty key")
            if key in self._active:
                raise MediaPipelineError("consumption key already has an active lease")
            self._active.add(key)
            registered = True
            if not reserve_control:
                return
            if is_producer and self._pending is not None and self._pending.key == key:
                return
            required = self._materializer.control_bytes(plan)
            if self._host_reserved + required > self.config.max_inflight_bytes and self._pending is not None:
                self.cancel_pending()
            self._add_host(required)
            host_charge += required
            control_charge = required

        def reserve_transport(amount: int) -> None:
            nonlocal transport_charge
            if self._transport_reserved + amount > self.config.max_transport_bytes:
                raise MediaBudgetError("active per-rank payloads exceed the transport storage budget")
            self._transport_reserved += amount
            transport_charge += amount
            self._counters["peak_transport_reserved_bytes"] = max(
                self._counters["peak_transport_reserved_bytes"], self._transport_reserved
            )

        def reserve_host_payload(amount: int) -> None:
            nonlocal host_charge
            self._add_host(amount)
            host_charge += amount

        try:
            delivered = self._transport.deliver(
                plan, key, sp_mesh, device, produce, reserve_transport, prepare_control, reserve_host_payload
            )
            self._counters["consumed"] += 1
            yield delivered
        except MediaTransportError as error:
            if error.fail_stop:
                self._poisoned = True
            raise
        finally:
            delivered = None
            if registered:
                self._active.discard(key)
            if not self._materializer.poisoned and not self._poisoned:
                self._host_reserved -= host_charge
                self._transport_reserved -= transport_charge
            else:
                self._poisoned = True

    def close(self) -> None:
        self._check_owner()
        if self._resources_closed:
            return
        self._closed = True
        deadline = time.monotonic() + self.config.shutdown_timeout_seconds
        errors = []
        try:
            self._cancel_pending_until(deadline)
        except BaseException as error:
            errors.append(error)
        try:
            self._materializer._close_until(deadline)
        except BaseException as error:
            errors.append(error)
        if errors:
            self._poisoned = True
            message = "media runtime cleanup incomplete; launcher termination required"
            raise MediaShutdownTimeout(message) from errors[0]
        self._resources_closed = True
        if not self._active:
            self._host_reserved = 0
            self._transport_reserved = 0

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
