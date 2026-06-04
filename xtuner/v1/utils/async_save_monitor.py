import threading
from concurrent.futures import Future
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
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="AsyncSaveMonitor", daemon=True)
        self._thread.start()

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
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join()
        self._thread = None

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
