"""Run claw-bench calendar tasks with the default agent pipeline."""

from claw_bench.dataset import ClawBench
from claw_bench.pipeline import claw_pipeline


dataset = ClawBench(
    tasks_root="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/bench/claw-bench/tasks",
    pipeline=claw_pipeline(),
)
