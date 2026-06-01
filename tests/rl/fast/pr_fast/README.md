# PR Fast

PR-fast tests cover low-cost RL logic: state machines, replay buffer behavior, train-data preparation,
single-turn agent-loop control flow, fake-controller producer / rollout-controller flows, worker control-flow
branches, and lightweight CPU/Ray utilities.

Do not add tests here if they require real model paths, real rollout servers, GPUs, or long-running backend
startup.

Reference runtime from the current suite:

```bash
conda activate pt28_all_env
python -m pytest tests/rl/fast/pr_fast -q
```

Latest measured result: `113 passed, 9 warnings in 47.61s`.
