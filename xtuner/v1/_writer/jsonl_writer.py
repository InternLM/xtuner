import json
import threading
import time
import weakref
from pathlib import Path
from queue import Empty, Queue

from xtuner.v1.utils import get_logger


logger = get_logger()

LOG_FILE_NAME = "tracker.jsonl"


class JsonlWriter:
    def __init__(
        self,
        log_dir: str | Path | None = None,
    ):
        if log_dir is None:
            log_dir = Path()

        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self._lock = threading.Lock()

        self.log_file = log_dir / LOG_FILE_NAME
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self._queue = Queue[str | None](maxsize=10000)
        self._file_writer = open(self.log_file, "a", encoding="utf-8")
        self._async_writer = _AsyncWriter(weakref.ref(self))
        self._closed = False
        self._async_writer.run()

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ):
        with self._lock:
            if self._closed:
                logger.warning(f"{tag}: {scalar_value} will be descarded because the writer is closed.")
                return
            write_item = {
                tag: scalar_value,
                "step": global_step,
            }
            self._queue.put(json.dumps(write_item, ensure_ascii=False))

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ):
        with self._lock:
            if self._closed:
                logger.warning(f"{tag_scalar_dict} will be descarded because the writer is closed.")
                return
            write_item = tag_scalar_dict.copy()
            write_item["step"] = global_step
            self._queue.put(json.dumps(write_item, ensure_ascii=False))

    def _write(self, item: str):
        self._file_writer.write(item + "\n")

    def close(self):
        with self._lock:
            if self._closed:
                return

            self._closed = True
            self._queue.put(None)

            self._async_writer.join()
            self._flush()
            self._file_writer.close()

    def _flush(self):
        self._queue.join()
        self._file_writer.flush()

    def __del__(self):
        try:
            self.close()
        except KeyboardInterrupt:
            pass
        except Exception:
            logger.warning(f"Exception occurred during closing writer for {self.log_file}")


class _AsyncWriter:
    def __init__(self, writer: weakref.ReferenceType[JsonlWriter]):
        self._writer = writer
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _run(self):
        writer = self._writer()
        if writer is None:
            return

        last_flush = time.monotonic()
        flush_interval = 30.0  # seconds
        while True:
            try:
                item = writer._queue.get(timeout=1.0)
                got_item = True
            except Empty:
                got_item = False

            if got_item:
                if item is None:
                    writer._queue.task_done()
                    break
                if writer is None:
                    break

                try:
                    writer._write(item)
                except KeyboardInterrupt:
                    break
                except Exception:
                    # Swallow write errors to keep the background thread alive
                    # or exit gracefully on I/O issues.
                    logger.warning(f"Exception occurred during writing data to {writer.log_file}")
                finally:
                    writer._queue.task_done()

            now = time.monotonic()
            if now - last_flush >= flush_interval:
                try:
                    writer._file_writer.flush()
                except Exception:
                    logger.warning(f"Exception occurred during flushing {writer.log_file}")
                last_flush = now

    def run(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="JsonlWriterAsync", daemon=True)
                self._thread.start()

    def join(self):
        with self._lock:
            t = self._thread
        if t is not None and t.is_alive():
            t.join()
