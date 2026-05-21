"""tb2-eval bench adapter for the rl_task framework."""

from .local_run.dataset import TB2EvalBench
from .pipeline import runner

__all__ = ["TB2EvalBench", "runner"]
