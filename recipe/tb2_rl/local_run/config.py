"""Run tb2-rl tasks with the default agent pipeline.

Invoke via::

    python -m recipe.tb2_rl.local_run --limit 5
"""

from recipe.tb2_rl.local_run.dataset import TB2RLBench
from recipe.tb2_rl.pipeline import runner


dataset = TB2RLBench(
    jsonl_path="/mnt/shared-storage-user/llmit1/user/liukuikun/delivery/data/tb2_rl_tasks.jsonl",
    pipeline=runner,
)
