import signal
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from xtuner.v1.utils.async_save_monitor import AsyncSaveMonitor, AsyncSaveWatchItem


def _watch_item(future: Future) -> AsyncSaveWatchItem:
    return AsyncSaveWatchItem(
        name="async_hf",
        future=future,
        path=Path("/tmp/hf-1"),
        step=1,
        epoch=1,
    )


class TestAsyncSaveMonitor(unittest.TestCase):
    def test_removes_successful_future(self):
        future: Future = Future()
        future.set_result(Path("/tmp/hf-1"))
        item = _watch_item(future)
        monitor = AsyncSaveMonitor()

        monitor.register(item)
        with patch("xtuner.v1.utils.async_save_monitor.os.killpg") as killpg:
            monitor._check_watched_futures()

        self.assertEqual(monitor._items, [])
        killpg.assert_not_called()

    def test_terminates_process_group_on_future_failure(self):
        future: Future = Future()
        future.set_exception(RuntimeError("mock async save failure"))
        item = _watch_item(future)
        monitor = AsyncSaveMonitor()

        monitor.register(item)
        with (
            patch("xtuner.v1.utils.async_save_monitor.os.getpgrp", return_value=1234) as getpgrp,
            patch("xtuner.v1.utils.async_save_monitor.os.killpg") as killpg,
        ):
            monitor._check_watched_futures()

        getpgrp.assert_called_once_with()
        killpg.assert_called_once_with(1234, signal.SIGTERM)
