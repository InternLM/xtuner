---
name: rl-accuracy-verify
description: Verify that a new change to XTuner does not regress RL training accuracy, by running the GRPO (Qwen3-8B gsm8k) and OPD (Qwen3-VL-2B multi-teacher MOPD) end-to-end trainings and comparing their tracker metrics against fixed baselines. Use when the user asks to validate RL precision/accuracy of a modification, run GRPO/OPD accuracy checks, or do an RL e2e regression before merging.
---

# RL Accuracy Verification (GRPO + OPD)

Verify a code change keeps GRPO and OPD training accuracy aligned with baselines.
Both runs need a single 8-GPU node and take ~40 min (GRPO) / up to a few hours (OPD, 50 steps).

Before launching anything, present an experiment card (目标 / 判读标准 / 最小规模 / 前置条件)
per the collaboration workflow and wait for confirmation.

## 0. Prerequisites

1. **Record the code version under test**: `git rev-parse HEAD` (+ `git diff` stat if the tree is dirty).
   The report must name the exact commit the numbers came from.
2. **Check inputs exist** (fail fast if any is missing). Paths below use the share roots
   `CI_SHARE_MODEL` / `CI_SHARE_DATA` — export them first, pointing at your cluster's
   model and data share directories:
   - GRPO: `${CI_SHARE_MODEL}/Qwen3-8B`,
     `${CI_SHARE_DATA}/gsm8k/train-mini.jsonl`,
     `${CI_SHARE_DATA}/gsm8k/test.jsonl`
   - OPD: MODEL_PATH `${CI_SHARE_MODEL}/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`,
     teachers `${CI_SHARE_MODEL}/Qwen3-4B-Instruct-2507` and
     `${CI_SHARE_MODEL}/Qwen3-VL-4B-Instruct`,
     data `${CI_SHARE_DATA}/on_policy_distillation/gsm8k_geo3k_{train,eval}.json`

## 1. GRPO accuracy run

Launch (from the repo root):

```bash
bash -x examples/v1/scripts/run_rl.sh autotest/config/rl_qwen3_gsm8k_grpo.py "lmdeploy" \
    ${CI_SHARE_MODEL}/Qwen3-8B \
    ${CI_SHARE_DATA}/gsm8k/train-mini.jsonl \
    ${CI_SHARE_DATA}/gsm8k/test.jsonl
```

Shape: 3 epochs over 1000 gsm8k prompts, global_batch_size 64 → exactly 45 RL steps (the baseline
has 45 step-summary lines; `check_rl_result` asserts equal step counts, so do not change
`total_epochs`/`global_batch_size`/dataset size). Expected wall time: under 40 min.

**Baseline**: `${CI_SHARE_BASELINE}/qwen3-rl-lmdeploy/tracker.jsonl`
(`CI_SHARE_BASELINE` = the e2e baseline store root, i.e. the directory configured as
`base_path.base_baseline_path` in `autotest/config.yaml`).
Average of 5 historical runs.

**Thresholds** (from the GRPO e2e spec):

| metric | method | threshold |
|---|---|---|
| `eval/accuracy` | absolute | < 0.05 (final step only — see below) |
| `response/rewards/mean` | absolute | < 0.1 |
| `mismatch/mismatch_k3_kl` | absolute | <= 0.0001 |
| `response/response_len/mean` | relative | < 12% |
| `time/step` | absolute | < 10 s |
| e2e wall time | — | < 40 min |

**Compare** by calling the existing checker — do not reimplement comparison logic. Run it from the
results directory (it imports matplotlib and `publish_comparison_report` writes plots into
`$CWD/$GITHUB_RUN_ID`, default `./0/`):

```python
import sys
sys.path.insert(0, "autotest")  # path to the repo's autotest dir
from utils.check_metric import check_rl_result

assert_info = {"check_metrics": [
    {"metric": "response/rewards/mean", "threshold": 0.1, "method": "absolute", "operator": "<"},
    {"metric": "mismatch/mismatch_k3_kl", "threshold": 0.0001, "method": "absolute", "operator": "<="},
    {"metric": "response/response_len/mean", "threshold": 0.12, "method": "relative", "operator": "<"},
    {"metric": "time/step", "threshold": 10, "method": "absolute", "operator": "<"},
]}
ok, msg = check_rl_result("grpo-gsm8k", BASE_TRACKER, CUR_TRACKER, assert_info)
```

Note: `check_rl_result` aggregates `response/rewards/mean` and `response/response_len/mean` with p80
across steps by default (`RL_PERCENTILE_METRICS`); other metrics are compared stepwise.

**`eval/accuracy` must be compared separately**: the baseline tracker records it only once, at the
final step (step 45, value 0.9007), while the config evaluates at steps 0/15/30/45. Putting it in
`check_metrics` fails with a series-length mismatch. Instead compare the final-step value directly
(final-point-only check is deliberate — the stored baseline has no earlier eval points; OPD below
checks every eval point because its golden values were recorded per point):

```python
import json

def final_eval_acc(path):
    vals = [r["eval/accuracy"] for line in open(path)
            if "eval/accuracy" in (r := json.loads(line))]
    return vals[-1]

assert abs(final_eval_acc(CUR_TRACKER) - final_eval_acc(BASE_TRACKER)) < 0.05
```

## 2. OPD accuracy run (multi-teacher MOPD, rollout teachers)

Single 8-GPU node: 6 student workers + 2 teachers (1 replica x 1 worker each).
The launch script starts the two teacher HTTP servers, injects `XTUNER_OPD_TEACHER_ENDPOINTS_JSON`,
then runs the RL training. Verified scale for accuracy checks: `TOTAL_TRAIN_STEPS=25` (constant LR,
no warmup, so steps 0-25 are identical to the 50-step variant; the gsm8k gain shows by step 10).
One 25-step run takes ~15 min on 8xH200.

```bash
export MODEL_PATH=${CI_SHARE_MODEL}/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
export DATA_PATH=${CI_SHARE_DATA}/on_policy_distillation/gsm8k_geo3k_train.json
export EVAL_DATA_PATH=${CI_SHARE_DATA}/on_policy_distillation/gsm8k_geo3k_eval.json
export GSM8K_TEACHER_MODEL_PATH=${CI_SHARE_MODEL}/Qwen3-4B-Instruct-2507
export GEO3K_TEACHER_MODEL_PATH=${CI_SHARE_MODEL}/Qwen3-VL-4B-Instruct
export OPD_CONFIG_PATH=recipe/on_policy_distillation/config/rl_qwen3_vl_2b_sampled_token_mopd_rollout_teacher.py
export XTUNER_USE_LMDEPLOY=1 XTUNER_USE_FA3=1
export TOTAL_TRAIN_STEPS=25 TRAIN_BATCH_SIZE=128 EVALUATE_STEP=5
export XTUNER_RL_NUM_WORKERS=6 MOPD_TEACHER_NUM_REPLICAS=1 MOPD_TEACHER_REPLICA_NUM_WORKERS=1
export WORK_DIR=<work dir for this run>

# Positional args are required: STUDENT_MODEL_PATH DATA_PATH
bash recipe/on_policy_distillation/scripts/run_sampled_token_mopd.sh "$MODEL_PATH" "$DATA_PATH"
```

The script partitions the node's GPUs itself (teachers first, students get the rest via
`STUDENT_CUDA_VISIBLE_DEVICES`) and starts/stops the teacher servers and Ray via an EXIT trap — do
not set `CUDA_VISIBLE_DEVICES` yourself.

**Baseline** (golden values): 3-run mean measured 2026-10-26 on the config above at
`TOTAL_TRAIN_STEPS=25` (commit `4e7db028`, lmdeploy main, ray 2.56, transformers 5.20.1).
Metric mapping: `distill_kl` = `distillation/reduced_distillation_kl`,
`abs_loss` = `distillation/reduced_distillation_abs_loss`, `rewards` = `response/rewards/mean`,
`gsm8k_acc` = `eval/gsm8k/acc`, `geo3k_acc` = `eval/geo3k/acc`.

| step | distill_kl | abs_loss | rewards | gsm8k_acc | geo3k_acc |
|------|------------|----------|---------|-----------|-----------|
|    0 |          - |        - |       - |    0.4339 |    0.2601 |
|    5 |     0.3260 |   0.4640 |  0.5286 |    0.5972 |    0.2686 |
|   10 |     0.3305 |   0.4617 |  0.6156 |    0.6871 |    0.2830 |
|   15 |     0.3162 |   0.4576 |  0.5914 |    0.6990 |    0.2745 |
|   20 |     0.3129 |   0.4422 |  0.6794 |    0.6904 |    0.3040 |
|   25 |     0.3118 |   0.4438 |  0.5771 |    0.6811 |    0.2945 |

**Compare at every eval point (steps 0/5/10/15/20/25), not just the final step.** The current run's
tracker is at `$WORK_DIR/<exp>/logs/exp_tracking/tracker.jsonl`; eval metrics are on the eval rows
(step 0 is the eval-only row before training), training metrics on the RL step-summary rows
(`response/rewards/mean` present; step N = row N).

**Thresholds** (measured max run-to-run deviation x 1.5 margin, N=3, 25 steps), applied per eval
point against the golden table:

| metric | threshold | measured max dev |
|---|---|---|
| `distillation/reduced_distillation_kl` | absolute < 0.03 | 0.0174 |
| `distillation/reduced_distillation_abs_loss` | absolute < 0.03 | 0.0179 |
| `response/rewards/mean` | absolute < 0.08 | 0.0523 |
| `eval/gsm8k/acc` | absolute < 0.03 | 0.0169 |
| `eval/geo3k/acc` | absolute < 0.04 | 0.0225 |
| `response/response_len/mean` | relative < 18% | 12.1% |
| `time/step` | relative < 10% (reference only; machine-dependent) | 4.3% |

In k1 loss mode `opd_reverse_kl` == `reduced_distillation_kl` and `opd_abs_logprob_loss` ==
`reduced_distillation_abs_loss` (verified numerically identical) — do not check both.

The config's docstring holds a 50-step reference table (recorded 2026-07-31); steps 0-25 of a
25-step run match it (e.g. step-25 distill_kl 0.3165 vs 0.3167). Steps 30-50 have no calibrated
thresholds yet — record a 50-step CI baseline before tightening that range.

Reference comparison snippet:

```python
import json

def series(path, metric, marker="response/rewards/mean"):
    vals = []
    for line in open(path):
        r = json.loads(line)
        if isinstance(r, dict) and marker in r and metric in r:
            vals.append(r[metric])
    return vals

# eval metrics live on eval-point rows (steps 0/5/.../25), keyed separately:
# series(path, "eval/gsm8k/acc", marker="eval/gsm8k/acc")
```

## 3. Interpreting results

- A metric drifting past threshold at **step 1 / first eval** points to initialization or config
  differences (weight mapping, seed, data order), not the training loop.
- A metric that **diverges gradually** points to the training loop (loss, advantage, KL, weight sync).
- `mismatch/mismatch_k3_kl` breach means train/inference inconsistency (logprob path, weight update,
  rollout backend) — check that first for GRPO failures.
- OPD `distill_kl` breach with healthy rewards points at the teacher path (teacher endpoints, sampled
  tokens, distillation loss); rewards-only drift points at rollout/judger, not distillation.

## 4. Report

Report per run: PASS/FAIL -> metric table (baseline vs current vs error vs threshold) -> tracker and log
paths -> commit under test -> suggested next step. Every number must point to a concrete tracker file;
mark inference vs measurement explicitly. If anything failed or looks abnormal, report immediately
rather than waiting to be asked.

## 5. Cleanup

`ray stop --force`
if a run died mid-flight, and keep the copied trackers/logs under the project directory for the record.
