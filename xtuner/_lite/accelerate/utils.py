import time
from contextlib import contextmanager

import torch

from xtuner._lite import get_logger

logger = get_logger()


def lmdeploy_is_available():

    available = False
    try:
        import lmdeploy  # noqa: F401
        available = True
    except ImportError:
        available = False

    return available


@contextmanager
def profile_time_and_memory(desc):

    start_t = time.time()
    torch.cuda.reset_peak_memory_stats()

    yield

    max_memory = torch.cuda.max_memory_allocated()
    cost_time = time.time() - start_t

    logger.info(f'{desc} Elapsed time {cost_time:.2f} seconds, '
                f'peak gpu memory {max_memory/1024**3:.1f}G')
