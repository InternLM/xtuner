from queue import Queue
from typing import Optional

from transformers.generation.streamers import BaseStreamer


class LMDeployTextStreamer(BaseStreamer):

    def put(self, text):
        self.on_finalized_text(text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        self.on_finalized_text('', stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout.

        If the stream is ending, also prints a newline.
        """
        print(text, flush=True, end='' if not stream_end else None)


class LMDeployTextIteratorStreamer(LMDeployTextStreamer):

    def __init__(self, timeout: Optional[float] = None):
        super().__init__()
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue.

        If the stream is ending, also put a stop signal in the queue.
        """
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
