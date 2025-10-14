import time
from collections import OrderedDict
from contextlib import contextmanager

from xtuner.v1.utils import get_logger, get_torch_device_module


logger = get_logger()


@contextmanager
def profile_time_and_memory(desc):
    torch_device = get_torch_device_module()
    start_t = time.time()
    torch_device.reset_peak_memory_stats()

    yield

    max_memory = torch_device.max_memory_allocated()
    cost_time = time.time() - start_t

    logger.success(f"{desc} Elapsed time {cost_time:.2f} seconds, peak gpu memory {max_memory / 1024**3:.1f}G")


class StepTimer:
    """A simple timer for measuring durations of sequential steps."""

    def __init__(self):
        self._start_time = time.time()
        self.laps = OrderedDict()

    def lap(self, name: str):
        """Record the time elapsed since the last lap or start."""
        end_time = time.time()
        duration = end_time - self._start_time
        self.laps[name] = duration
        self._start_time = end_time

    def format_results(self) -> str:
        """Format the recorded laps into a human-readable string."""
        if not self.laps:
            return "No timing data recorded."

        report_lines = [f"  - {name:<25}: {duration:.2f}s" for name, duration in self.laps.items()]
        total_duration = sum(self.laps.values())
        # report_lines.append("-" * 30)
        report_lines.append(f"  - {'Total':<25}: {total_duration:.2f}s")

        return "Step Timing Report:\n" + "\n".join(report_lines)
