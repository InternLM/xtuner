# XTuner — Codebase Guide

## Project Overview

This repository currently has two layers:

- `xtuner/` — the legacy XTuner package, old configs/tools, and compatibility surface.
- `xtuner/v1/` — the new V1 engine. On this RL refactor branch, most new training and RL work happens here.

For RL work on this branch, the most reliable sources are usually:

1. `xtuner/v1/rl/`
2. `xtuner/v1/train/rl_colocate_trainer.py`
3. `examples/v1/config/*.py`
4. `tests/rl/*.py`

If docs and code disagree, prefer the examples/tests in this branch.

---

## Repository Layout

### Repo Root

```text
.
├── xtuner/                  # Legacy package + V1 engine under xtuner/v1/
├── examples/v1/             # Runnable V1 example configs and launch scripts
├── tests/rl/                # RL unit/integration coverage for the refactored stack
├── design/                  # Draft design docs / experiments (not always source of truth)
├── docs/                    # User-facing docs; some pages may lag this branch
├── recipe/verl_agent/       # Older / side-path agentic RL recipes
├── autotest/                # CI-style automation scripts/configs
└── zdev/                    # Local dev notes / agent docs
```

### Current V1 Engine

```text
xtuner/v1/
├── config/       # FSDP / optimizer / LR / generation configs
├── data_proto/   # Typed runtime data objects: RolloutState, SampleParams, SequenceContext
├── datasets/     # JSONL datasets, tokenize_fns, dataloaders, collators, packing
├── engine/       # TrainEngine and vision-compose engines
├── float8/       # FP8 helpers and kernels
├── loss/         # SFT / pretrain losses
├── model/        # V1 model configs + implementations (dense / moe / compose)
├── module/       # Transformer building blocks: attention, router, rope, lm_head, etc.
├── ops/          # Kernels and distributed comm helpers
├── patch/        # Runtime patches / monkey patches
├── profiler/     # Time / memory profiling
├── rl/           # RL runtime (rollout, trainer workers, judger, replay buffer, agent loop)
├── train/        # SFT Trainer, RLColocateTrainer, CLI entry points
├── _writer/      # TensorBoard / JSONL logging backends
└── utils/        # Config loader, device helpers, Ray helpers, logging, metrics
```

### RL-Specific Layout

```text
xtuner/v1/rl/
├── agent_loop/       # sampler -> generate -> judge -> replay buffer orchestration
├── judger/           # GSM8K / GEO3K / DAPO Math reward logic + router/ray wrappers
├── loss/             # GRPO / OREAL losses and shared RL loss base
├── rollout/          # Inference workers/controllers for lmdeploy / vllm / sglang
├── trainer/          # Training workers/controllers for policy update + weight sync
├── utils/            # Ray worker helpers, async helpers, query DSL, logprob helpers
├── evaluator.py      # Eval batch sizing + metric hook
└── replay_buffer.py  # Sync FIFO buffer + async staleness-aware buffer
```

### Train Entry Files

```text
xtuner/v1/train/
├── trainer.py              # Main SFT / pretrain trainer
├── rl_colocate_trainer.py  # Main checked-in RL trainer on this branch
├── cli/sft.py              # SFT / pretrain CLI
└── cli/rl.py               # RL CLI
```

Important practical note:

- `examples/v1/config/rl_disagg_single.py` and `rl_disagg_multi.py` exist in this checkout as draft examples for a disaggregated RL design.
- The corresponding `xtuner/v1/train/rl_disaggregated_trainer.py` is not present in the checked-in code under `xtuner/v1/train/` right now.
- Treat those disagg example configs as branch-local WIP, not the main implemented trainer path.

---

## Entry Points

### SFT / Pretrain

Config-file mode:

```bash
python xtuner/v1/train/cli/sft.py --config <config.py>
```

Argument mode:

```bash
python xtuner/v1/train/cli/sft.py <training arguments>
```

The CLI converts either a config file or `TrainingArguments` into `TrainerConfig`, then runs `Trainer.fit()`.

### RL

Recommended launcher:

```bash
bash examples/v1/scripts/run_rl.sh <config.py> <backend> <model_path> <data_path> [eval_data_path]
```

What `run_rl.sh` does:

1. starts / joins a Ray cluster
2. exports backend env vars (`XTUNER_USE_LMDEPLOY`, `XTUNER_USE_SGLANG`, `XTUNER_USE_VLLM`)
3. sets `WORK_DIR`, `MODEL_PATH`, `DATA_PATH`, `EVAL_DATA_PATH`
4. runs:

```bash
python xtuner/v1/train/cli/rl.py --config <config.py>
```

The RL CLI loads `Config.fromfile(config)["trainer"]`, builds the trainer object, and calls `fit()`.

---

## Current RL Architecture

The old mental model in this repo was “one RL trainer config plus rollout/replay pieces”.
On this branch, the real center of gravity is:

- `RLColocateTrainerConfig`
- `WorkerConfig`
- `RolloutConfig`
- `AgentLoopManagerConfig`
- `SyncReplayBufferConfig` / `AsyncReplayBufferConfig`
- `EvaluatorConfig`

### Core Runtime Object: `RolloutState`

The central RL sample object is `xtuner/v1/data_proto/rl_data.py:RolloutState`, not a plain dict.

It carries most of the RL runtime state:

- original `message`
- `prompt_ids`
- generation inputs (`tokens`, `sample_params`, tools)
- generation outputs (`response`, `response_ids`, `logprobs`, `routed_experts`)
- reward info (`reward_model`, `reward`)
- rollout status (`INIT`, `COMPLETED`, `ABORTED`, `FAILED`, `FILTERED`, `EXPIRED`)
- multimodal data (`mm_info`)
- staleness / partial-rollout bookkeeping (`response_rollout_steps`, `seq_staleness`)

`SampleParams` also lives in `data_proto/rl_data.py` and is the canonical generation config used by agent loops and rollout workers.

### Actual Training Loop

The checked-in RL trainer is `xtuner/v1/train/rl_colocate_trainer.py`.

Its high-level loop is:

1. `AgentLoopManager.produce_batch(...)` generates rollout groups
2. trajectories are saved to `work_dir/.../train_rollout/`
3. rollout workers are offloaded
4. training data is prepared from grouped `RolloutState`
5. `TrainingController.fit(...)` runs policy updates on training workers
6. updated weights are synced back to rollout workers
7. optional checkpoint / HF export / evaluation runs

This is a controller-worker architecture:

- rollout side: `xtuner/v1/rl/rollout/controller.py` + `worker.py`
- train side: `xtuner/v1/rl/trainer/controller.py` + `worker.py`

### Agent Loop Layer

`xtuner/v1/rl/agent_loop/` is now the main orchestration layer above rollout workers.

Key pieces:

- `SamplerConfig` builds an RL dataloader from `DataloaderConfig`
- `Sampler` repeats each prompt `prompt_repeat_k` times and yields grouped `RolloutState`
- `AgentLoopConfig` is the base abstraction
- `SingleTurnAgentLoopConfig` is the default path for one-shot generate-then-judge
- `GSM8KToolAgentLoopConfig` supports a tool-call style multi-turn loop
- `AgentLoopManagerConfig` can manage one task or multiple weighted tasks
- `TaskSpecConfig.weight` is used for multi-task batch allocation

This means multi-task RL is no longer a special outer trainer concept; it is handled inside the agent-loop manager by multiple task specs.

### Produce Strategy: Sync vs Async

The old “RL = one rollout batch, then train” story is no longer the whole picture.

`xtuner/v1/rl/agent_loop/producer.py` now contains two collection strategies:

- `SyncProduceStrategy`
  - simple on-policy batch fill
  - launches enough groups to collect `batch_size` completed groups
  - uses FIFO replay-buffer behavior naturally

- `AsyncProduceStrategy`
  - supports oversampling via `over_sample_threshold`
  - supports partial rollout via `enable_partial_rollout`
  - supports stale-tail handling via `tail_batch_trigger_size` and `tail_batch_stale_threshold`
  - explicitly pauses rollout generation and recycles leftover groups

This async path is one of the biggest branch-level changes.

### Replay Buffer

`xtuner/v1/rl/replay_buffer.py` is no longer just a thin queue.

It now has:

- storage backends (`NaiveStorage`, `PandasStorage`)
- query DSL support for filtering stored groups
- replay policies:
  - `FIFOReplayPolicy`
  - `StalenessReplayPolicy`
- config wrappers:
  - `SyncReplayBufferConfig`
  - `AsyncReplayBufferConfig`

Current checked-in configs build `NaiveStorage`, but the abstraction already expects richer async / stale-sample policies.

### Judger Layer

Rewarding is handled through `xtuner/v1/rl/judger/`.

Common configs:

- `GSM8KJudgerConfig`
- `GEO3KJudgerConfig`
- `DapoMathJudgerConfig`

Execution modes come from `JudgerConfig.judger_type`:

- `native`
- `ray.actor`
- `router`

So “judger” is now a scalable service abstraction, not just a local reward function.

### Rollout Layer

`xtuner/v1/rl/rollout/worker.py:RolloutConfig` is the main rollout backend config.

Important responsibilities:

- backend process launch
- TP / EP topology
- context length / batching / chunked prefill
- retry / timeout / health checking
- optional routed-expert return
- worker offload / onload / recovery

Backend-specific adapters live in:

- `xtuner/v1/rl/rollout/lmdeploy.py`
- `xtuner/v1/rl/rollout/vllm.py`
- `xtuner/v1/rl/rollout/sglang.py`

`RolloutController` is also responsible for active-worker tracking and failed-worker recovery.

### Training Worker Layer

Policy updates live under `xtuner/v1/rl/trainer/worker.py`.

`WorkerConfig` now owns:

- model config / load path
- optimizer / lr / FSDP config
- RL loss config
- optional reference model for KL loss
- sequence parallel size
- packing length
- optional interleaved SFT data + CE loss

That last point matters: the RL worker can optionally mix in SFT-style training data during the RL loop.

### RL Losses

The current checked-in RL losses are:

- `GRPOLossConfig` in `xtuner/v1/rl/loss/grpo_loss.py`
- `OrealLossConfig` in `xtuner/v1/rl/loss/oreal_loss.py`

So the branch has already moved beyond “GRPO only”.

---

## Example Map

These are the most useful example configs in `examples/v1/config/` for understanding the current RL stack:

- `rl_grpo_gsm8k_judge.py`
  - simplest colocated GRPO + GSM8K path
- `rl_grpo_gsm8k_async.py`
  - async produce strategy
- `rl_grpo_gsm8k_with_tool.py`
  - tool-using agent loop
- `rl_grpo_geo3k_judge.py`
  - GEO3K judger path
- `rl_dapo_math.py`
  - DAPO Math reward path
- `rl_dapo_math_async.py`
  - async DAPO Math
- `rl_dapo_math_async_filter.py`
  - async path with extra filtering behavior
- `rl_multi_task_gsm8k_dapo_math.py`
  - multi-task RL via weighted `AgentLoopManagerConfig`
- `rl_disagg_single.py`
  - draft disaggregated single-task config
- `rl_disagg_multi.py`
  - draft disaggregated multi-task config

For the current implemented trainer path, start with `rl_grpo_gsm8k_judge.py`.

---

## Tests To Read First

If you need ground truth for current behavior, these are especially useful:

- `tests/rl/test_agent_loop.py`
- `tests/rl/test_producer.py`
- `tests/rl/test_replay_buffer.py`
- `tests/rl/test_async_rollout.py`
- `tests/rl/test_rl_colocate_trainer_integration.py`
- `tests/rl/test_judger.py`
- `tests/rl/test_rollout.py`

---

## Working Notes For Agents

- Do not confuse legacy `xtuner/configs/...` with the V1 engine config system in `xtuner/v1/...`.
- For RL on this branch, `examples/v1/config` and `tests/rl` are usually more up to date than `docs/`.
- The main implemented RL trainer is colocated (`rl_colocate_trainer.py`), while disaggregated configs are still branch-local WIP.
- The key “new” abstraction compared with older XTuner RL code is the `agent_loop` layer sitting between dataloader, rollout, judger, and replay buffer.
- Multi-task RL is handled in `AgentLoopManager`, not by a separate multi-task trainer.
