# Copyright (c) OpenMMLab. All rights reserved.
import json
import multiprocessing
import os
import signal
import sys
import time
from multiprocessing import Queue

import pytest

from xtuner.v1.datasets.jsonl import parallel_execute
from xtuner.v1.utils import SharedMemory


def _tokenize(data: bytes) -> dict:
    if data == b"crash":
        os._exit(17)
    if data == b"exit0":
        os._exit(0)
    if data == b"slow":
        time.sleep(30)
    return {"value": json.loads(data)}


def _run_worker(shm_name: str, offsets: list, result_queue: Queue) -> None:
    # Give this test its own process group so a timeout can stop all tokenizer children.
    os.setsid()
    # A forked worker can inherit a SIGTERM handler from its training process.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        result = parallel_execute(_tokenize, offsets, shm_name, nproc=2, chunksize=1, rank=0)
    except RuntimeError as exc:
        result_queue.put(("error", str(exc), len(multiprocessing.active_children())))
    else:
        result_queue.put(("ok", result, len(multiprocessing.active_children())))


@pytest.mark.skipif(sys.platform != "linux", reason="parallel tokenization uses fork and CPU affinity")
class TestParallelTokenization:
    def _run(self, records: list[bytes]) -> tuple:
        offsets = []
        start = 0
        for index, record in enumerate(records):
            offsets.append((index, (start, start + len(record))))
            start += len(record)
        segment = SharedMemory(create=True, size=start)
        segment.buf[:] = b"".join(records)
        ctx = multiprocessing.get_context("fork")
        result_queue = ctx.Queue()
        process = ctx.Process(target=_run_worker, args=(segment.name, offsets, result_queue))
        process.start()
        try:
            process.join(timeout=10)
            if process.is_alive():
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.terminate()
                process.join()
                pytest.fail("parallel tokenization did not finish after a worker failed")
            assert process.exitcode == 0
            return result_queue.get(timeout=2)
        finally:
            segment.close()
            segment.unlink()
            result_queue.close()

    def test_success_returns_results_in_input_order_and_joins_workers(self) -> None:
        status, result, children = self._run([b"0", b"1", b"2", b"3"])
        assert status == "ok"
        assert result == [{"value": 0}, {"value": 1}, {"value": 2}, {"value": 3}]
        assert children == 0

    def test_tokenizer_exception_propagates_and_stops_other_workers(self) -> None:
        # Leave enough queued input to fill the feeder pipe after workers stop.
        status, error, children = self._run([b"slow", b"bad"] + [b"0"] * 10_000)
        assert status == "error"
        assert "sample 1" in error
        assert "JSONDecodeError" in error
        assert children == 0

    def test_unexpected_worker_exit_does_not_hang(self) -> None:
        status, error, children = self._run([b"slow", b"crash"])
        assert status == "error"
        assert "exited with code 17" in error
        assert children == 0

    def test_clean_exit_without_completion_does_not_hang(self) -> None:
        status, error, children = self._run([b"slow", b"exit0"])
        assert status == "error"
        assert "exited without reporting completion" in error
        assert children == 0
