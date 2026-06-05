import os
import signal
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
        self._terminated = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._interval = interval
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AsyncSaveMonitor")
        self._monitor_future: Future

    def start(self) -> None:
        self._monitor_future = self._executor.submit(self._run)

    def register(self, item: AsyncSaveWatchItem) -> None:
        with self._lock:
            self._items.append(item)

    def stop(self) -> None:
        self._stop_event.set()
        self._monitor_future.result()
        self._executor.shutdown(wait=True)

    def _run(self) -> None:
        while True:
            stopped = self._stop_event.wait(self._interval)
            if stopped:
                self._check_watched_futures()
                break
            self._check_watched_futures()

    def _check_watched_futures(self) -> None:
        failure: tuple[AsyncSaveWatchItem, BaseException] | None = None

        with self._lock:
            for item in list(self._items):
                if not item.future.done():
                    continue

                self._items.remove(item)
                try:
                    exc = item.future.exception()
                except BaseException as future_exc:
                    failure = (item, future_exc)
                    break
                if exc is not None:
                    failure = (item, exc)
                    break

        if failure is not None:
            self._terminate_failure(failure)

    def _terminate_failure(self, failure: tuple[AsyncSaveWatchItem, BaseException]) -> None:
        with self._lock:
            if self._terminated:
                return
            self._terminated = True

        item, exc = failure
        logger.error(
            f"{item.name} failed at step={item.step}, epoch={item.epoch}, path={item.path}: {exc}",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        os.killpg(os.getpgrp(), signal.SIGTERM)
