"""Run claw-bench calendar tasks against the upstream task dirs."""

from claw_bench import ClawBench


dataset = ClawBench(
    tasks_root="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/bench/claw-bench/tasks",
)
