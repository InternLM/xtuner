import time
from contextlib import contextmanager

from xtuner.v1.utils import get_logger, get_torch_device_module


logger = get_logger()

@contextmanager
def profile_time(desc):
    start_t = time.time()

    yield

    cost_time = time.time() - start_t
    logger.success(f"{desc} Elapsed time {cost_time:.2f} seconds")


@contextmanager
def profile_time_and_memory(desc):
    torch_device = get_torch_device_module()
    start_t = time.time()
    torch_device.reset_peak_memory_stats()

    yield

    max_memory = torch_device.max_memory_allocated()
    cost_time = time.time() - start_t

    logger.success(f"{desc} Elapsed time {cost_time:.2f} seconds, peak gpu memory {max_memory / 1024**3:.1f}G")


# Adapted from https://github.com/volcengine/verl/blob/main/verl/utils/profiler/performance.py
@contextmanager
def timer(name: str, timer_dict: dict[str, float]):
    # TODO: install codetiming in xtuner latest images
    from codetiming import Timer

    with Timer(name=name, logger=None) as t:
        yield
    if name not in timer_dict:
        timer_dict[name] = 0.0
    timer_dict[name] += t.last


def timer_logger(time_dict: dict[str, float]):
    report_lines = [f"  - {name:<25}: {duration:.2f}s" for name, duration in time_dict.items()]
    total_duration = sum(time_dict.values())
    report_lines.append(f"  - {'Total':<25}: {total_duration:.2f}s")
    return "\n".join(report_lines)
