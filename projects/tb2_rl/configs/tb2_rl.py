"""Run tb2-rl tasks with the default agent pipeline.

Invoke via::

    python xtuner/v1/ray/environment/rl_task/runner.py \\
        --config projects/tb2_rl/configs/tb2_rl.py \\
        --limit 5
"""

from tb2_rl.dataset import TB2RLBench
from tb2_rl.pipeline import tb2_rl_pipeline


dataset = TB2RLBench(
    tasks_root="/mnt/shared-storage-user/llmit1/user/liukuikun/delivery/data/data_process",
    pipeline=tb2_rl_pipeline(),
)
