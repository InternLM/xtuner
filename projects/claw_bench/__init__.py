"""claw-bench: bench adapter project for the rl_task framework.

Layout::

    pipeline.py     # claw_pipeline + claw_solution_pipeline (Runner factory)
    dataset.py      # ClawBench (task.toml → TaskData; iterates tasks_root)
    wrappers/       # bench + agent-framework shell scripts shipped to sandbox
    agents/         # internclaw and any other agent templates

Use::

    from claw_bench.dataset import ClawBench
    ds = ClawBench(tasks_root="/path/to/upstream/claw-bench/tasks")
    for task_dir, data in ds.iter_tasks():
        await ds.pipeline.run_single(task_dir, data, uid, provider=...)
"""

from claw_bench.dataset import ClawBench
from claw_bench.pipeline import claw_pipeline, claw_solution_pipeline


__all__ = ["ClawBench", "claw_pipeline", "claw_solution_pipeline"]
