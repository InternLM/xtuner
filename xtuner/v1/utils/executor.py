from __future__ import annotations

import multiprocessing
import multiprocessing.context
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Iterable, Iterator

import numpy as np


@dataclass
class _ShmRef:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


class _StopSignal:
    pass


def _worker_loop(
    fn: Callable,
    task_queue: Any,
    result_queue: Any,
    shm_refs: dict[str, _ShmRef],
    scalar_kwargs: dict[str, Any],
) -> None:
    kwargs: dict[str, Any] = {**scalar_kwargs}
    shm_handles: list[SharedMemory] = []
    for key, ref in shm_refs.items():
        shm = SharedMemory(name=ref.name)
        shm_handles.append(shm)
        kwargs[key] = np.ndarray(ref.shape, dtype=ref.dtype, buffer=shm.buf)

    while True:
        msg = task_queue.get()
        if isinstance(msg, _StopSignal):
            break
        idx, task = msg
        try:
            result = fn(task, **kwargs)
            result_queue.put((idx, result, None))
        except Exception as exc:  # noqa: BLE001
            result_queue.put((idx, None, exc))

    for shm in shm_handles:
        shm.close()


class SharedPoolExecutor:
    """A process pool where the target function and shared kwargs are bound
    once per worker, eliminating per-task serialization overhead.

    numpy ndarrays in ``partial_kwargs`` are placed in POSIX shared memory
    so that all worker processes on the same host attach to the **same
    physical pages** without any per-worker copy.  Without this, each worker
    would receive its own serialized (pickled) copy of every ndarray on
    every task — for large arrays such as cumulative-length tables this
    creates significant memory overhead proportional to the worker count.
    All other values are pickled normally once at worker startup.  Each call
    to ``submit`` or ``map`` only serializes the per-task varying argument.

    Args:
        fn (Callable): The function executed by every worker. Its signature
            must accept a single positional argument (the per-task item)
            followed by the keyword arguments in ``partial_kwargs``.
        partial_kwargs (dict[str, Any]): Fixed keyword arguments forwarded
            to ``fn`` on every call. ``np.ndarray`` values are transparently
            moved to shared memory; other values are pickled once at worker
            startup.
        max_workers (int): Number of worker processes to spawn.
        mp_context (BaseContext | str | None): Multiprocessing start method
            or context. Defaults to the system default when ``None``.

    Example:
        >>> with SharedPoolExecutor(fn=my_fn, partial_kwargs={"arr": big_array}, max_workers=4) as pool:
        ...     futures = [pool.submit(item) for item in chunks]
        ...     results = [f.result() for f in futures]
    """

    def __init__(
        self,
        fn: Callable,
        partial_kwargs: dict[str, Any],
        max_workers: int,
        mp_context: multiprocessing.context.BaseContext | str | None = None,
    ) -> None:
        self._max_workers = max_workers
        self._shm_handles: list[SharedMemory] = []
        self._pending: dict[int, Future] = {}
        self._next_idx = 0
        self._lock = threading.Lock()

        shm_refs: dict[str, _ShmRef] = {}
        scalar_kwargs: dict[str, Any] = {}

        for key, val in partial_kwargs.items():
            if isinstance(val, np.ndarray):
                # Place each ndarray in a named POSIX shared-memory segment.
                # All worker processes will attach to this segment by name and
                # wrap it with np.ndarray — no copy is made per worker, so
                # memory usage stays constant regardless of the worker count.
                shm = SharedMemory(create=True, size=val.nbytes)
                shm_arr = np.ndarray(val.shape, dtype=val.dtype, buffer=shm.buf)
                np.copyto(shm_arr, val)
                self._shm_handles.append(shm)
                shm_refs[key] = _ShmRef(name=shm.name, shape=val.shape, dtype=val.dtype)
            else:
                scalar_kwargs[key] = val

        # Typed as Any because multiprocessing.context.BaseContext stubs do
        # not expose .Process / .Queue; the concrete context objects do.
        ctx: Any
        if isinstance(mp_context, str):
            ctx = multiprocessing.get_context(mp_context)
        elif mp_context is None:
            ctx = multiprocessing.get_context()
        else:
            ctx = mp_context

        # Guard the rest of __init__ so that if any worker process fails to
        # start (e.g. the OS process-count limit is hit and fork() returns
        # EAGAIN), we release every SharedMemory segment that was already
        # allocated above.  Without this, those segments would linger in
        # /dev/shm for the lifetime of the OS because nothing else holds a
        # reference to unlink them.
        try:
            self._task_queue = ctx.Queue()
            self._result_queue = ctx.Queue()

            self._workers: list[multiprocessing.Process] = []
            for _ in range(max_workers):
                p = ctx.Process(
                    target=_worker_loop,
                    args=(fn, self._task_queue, self._result_queue, shm_refs, scalar_kwargs),
                    daemon=True,
                )
                p.start()
                self._workers.append(p)

            self._drain_thread = threading.Thread(target=self._drain, daemon=True)
            self._drain_thread.start()
        except Exception:
            for shm in self._shm_handles:
                shm.close()
                shm.unlink()
            raise

    def _drain(self) -> None:
        while True:
            msg = self._result_queue.get()
            if isinstance(msg, _StopSignal):
                break
            idx, result, exc = msg
            with self._lock:
                fut = self._pending.pop(idx)
            if exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(result)

    def submit(self, item: Any) -> Future:
        """Submit a single task and return a Future for its result.

        Args:
            item (Any): The per-task argument passed as the first positional
                argument to ``fn``.

        Returns:
            Future: A future whose result will be set when the worker completes.
        """
        fut: Future = Future()
        with self._lock:
            idx = self._next_idx
            self._next_idx += 1
            self._pending[idx] = fut
        self._task_queue.put((idx, item))
        return fut

    def map(self, iterable: Iterable) -> Iterator:
        """Submit all items and yield results in submission order.

        Args:
            iterable (Iterable): Per-task varying arguments passed as the
                first positional argument to ``fn``.

        Returns:
            Iterator: Results in the same order as ``iterable``.
        """
        futures = [self.submit(item) for item in iterable]
        for fut in futures:
            yield fut.result()

    def shutdown(self) -> None:
        """Stop all worker processes and release shared memory."""
        for _ in self._workers:
            self._task_queue.put(_StopSignal())
        for p in self._workers:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
                p.join()
        self._result_queue.put(_StopSignal())
        self._drain_thread.join()
        for shm in self._shm_handles:
            shm.close()
            shm.unlink()

    def __enter__(self) -> SharedPoolExecutor:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
