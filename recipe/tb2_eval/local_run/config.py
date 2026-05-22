"""Run tb2-eval tasks with the default agent pipeline.

Invoke via::

    python -m recipe.tb2_eval.local_run --limit 5
"""

from recipe.tb2_eval.local_run.dataset import TB2EvalBench
from recipe.tb2_eval.pipeline import runner


dataset = TB2EvalBench(
    jsonl_path="/mnt/shared-storage-user/llmit/user/wangziyi/projs/terminalbench2-harbor-p-cluster/tb2_eval_tasks.jsonl",
    pipeline=runner,
)
