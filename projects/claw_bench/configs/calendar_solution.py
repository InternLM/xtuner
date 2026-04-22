"""Debug: run claw-bench calendar tasks via solution/solve.sh (skip LLM).

If tasks fail here, the bug is in the validate side (upload, env, verifier,
judger).  If they pass here but fail with ``calendar.py`` config, the bug
is in agent inference.
"""

from claw_bench.dataset import ClawBench
from claw_bench.pipeline import claw_solution_pipeline


dataset = ClawBench(
    tasks_root="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/bench/claw-bench/tasks",
    pipeline=claw_solution_pipeline(),
)
