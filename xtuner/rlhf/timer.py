import time

from loguru import logger


class Timer:
    """Timer."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.duration = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def start(self):
        logger.info(f'Start {self.task_name}')
        self.start = time.time()

    def end(self):
        self.duration = time.time() - self.start
        logger.info(
            f'  End {self.task_name}, duration = {self.duration:.2f} seconds')
