"""Run tb2-eval tasks with the default agent pipeline.

Invoke via::

    python xtuner/v1/ray/environment/rl_task/runner.py \\
        --config projects/tb2_eval/configs/tb2_eval.py \\
        --limit 5
"""

from tb2_eval.dataset import TB2EvalBench
from tb2_eval.pipeline import tb2_eval_pipeline

dataset = TB2EvalBench(
    tasks_root="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/bench/terminal-bench-2",
    pipeline=tb2_eval_pipeline(),
)
