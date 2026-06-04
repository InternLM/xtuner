import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from xtuner.v1.utils import get_logger


logger = get_logger()


@dataclass
class AsyncSaveWatchItem:
    name: str
    future: Future
    path: Path
    step: int
    epoch: int | None


class AsyncSaveMonitor:
    def __init__(self, interval: float = 5.0):
        self._items: list[AsyncSaveWatchItem] = []
        self._failure: tuple[AsyncSaveWatchItem, BaseException] | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._interval = interval
        self._executor: ThreadPoolExecutor | None = None
        self._monitor_future: Future | None = None

    def start(self) -> None:
        if self._executor is not None:
            return
        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AsyncSaveMonitor")
        self._monitor_future = self._executor.submit(self._run)

    def register(self, item: AsyncSaveWatchItem) -> None:
        with self._lock:
            self._items.append(item)

    def record_failure(self, item: AsyncSaveWatchItem, exc: BaseException) -> None:
        with self._lock:
            self._record_failure_locked(item, exc)

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is None:
            return

        item, exc = failure
        raise RuntimeError(
            f"{item.name} failed at step={item.step}, epoch={item.epoch}, path={item.path}"
        ) from exc

    def stop(self) -> None:
        executor = self._executor
        monitor_future = self._monitor_future
        if executor is None:
            return
        self._stop_event.set()
        if monitor_future is not None:
            monitor_future.result()
        executor.shutdown(wait=True)
        self._executor = None
        self._monitor_future = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            self._poll_once()

    def _poll_once(self) -> None:
        with self._lock:
            items = list(self._items)

        finished: list[AsyncSaveWatchItem] = []
        failure: tuple[AsyncSaveWatchItem, BaseException] | None = None

        for item in items:
            if not item.future.done():
                continue

            finished.append(item)
            try:
                exc = item.future.exception()
            except BaseException as future_exc:
                failure = (item, future_exc)
                break
            if exc is not None:
                failure = (item, exc)
                break

        with self._lock:
            for item in finished:
                if item in self._items:
                    self._items.remove(item)

            if failure is not None and self._failure is None:
                item, exc = failure
                self._record_failure_locked(item, exc)

    def _record_failure_locked(self, item: AsyncSaveWatchItem, exc: BaseException) -> None:
        if self._failure is not None:
            return
        logger.error(f"{item.name} failed at step={item.step}, epoch={item.epoch}, path={item.path}: {exc}")
        self._failure = (item, exc)
