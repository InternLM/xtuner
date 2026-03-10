# XTuner — Codebase Guide

## Project Overview

XTuner is a distributed training framework for Large Language Models (LLMs) and Multimodal LLMs (MLLMs). It supports Supervised Fine-Tuning (SFT), pretraining, and Reinforcement Learning (RL/GRPO). All active development lives under `xtuner/v1/`.

---

## Repository Layout

```
xtuner/v1/
├── config/         # FSDP, optimizer, LR scheduler, generation configs
├── model/          # Model definitions (Dense, MoE, VLM); base.py is the core
├── module/         # Custom layers: attention, RMSNorm, RoPE, MoE router/dispatcher
├── datasets/       # Data pipeline: tokenization, packing, caching, collation
├── train/          # Trainer, RL trainer, CLI entry points
│   ├── trainer.py       # Main Trainer + TrainerConfig (SFT / pretrain)
│   ├── rl_trainer.py    # RLTrainer + RLTrainerConfig (GRPO)
│   └── cli/sft.py       # torchrun entry point for SFT
├── engine/         # TrainEngine: forward/backward/optimizer step abstraction
├── loss/           # CE loss, chunked loss, MoE balancing loss, Z-loss
├── ops/            # CUDA/Triton kernels: attention, comm, MoE ops, RoPE
├── rl/             # GRPO algorithm, rollout workers, Ray coordination
├── float8/         # FP8 quantization + FSDP integration
├── optim/          # Muon optimizer
├── profiler/       # TPS / memory profilers
├── patch/          # HuggingFace monkey-patches
├── ray/            # Ray actor abstractions for distributed RL
├── data_proto/     # Serialization for distributed data handling
└── utils/          # Config loader, logging, grad norm, device helpers, etc.
```

---

## Key Abstractions

### Configuration Pattern
Every major component is represented by a Pydantic `BaseModel` config with a `.build()` method. Configs compose hierarchically into `TrainerConfig` / `RLTrainerConfig`.

```python
# All components follow this pattern
class FooConfig(BaseModel):
    ...
    def build(self) -> Foo: ...
```

### Training Entry Points

**SFT / Pretrain** (command-line):
```bash
torchrun xtuner/v1/train/cli/sft.py \
    --load-from <hf_model_path> \
    --chat_template qwen3 \
    --dataset <jsonl_path>
```

**SFT / Pretrain** (config file):
```bash
torchrun xtuner/v1/train/cli/sft.py <config.py>
```

**RL / GRPO**:
```bash
bash examples/v1/run_rl.sh <config.py> lmdeploy <model_path> <data_path>
```

> Command-line mode and config-file mode are **mutually exclusive** for the SFT CLI.

---

## Data Pipeline

### Data Protocol
All datasets must produce dicts with this schema:
```python
{
    "input_ids": list[int],   # token IDs
    "labels":    list[int],   # -100 for masked positions
    "num_tokens": int,        # for length-balanced sampling
}
```

### Tokenization Functions
Register a tokenization function by subclassing `BaseTokenizeFnConfig` and implementing its `build()` method. Built-in variants:
- `OpenaiTokenizeFnConfig` — OpenAI-format chat JSONL
- `PretrainTokenizeFunction` — long-sequence pretraining
- `RLTokenizeFnConfig` — RL rollout data with rewards
- `InternS1VLTokenizeFunction`, `Qwen3VLTokenizeFunction` — multimodal

### Dataset Caching
Tokenized data is cached automatically. Cache is invalidated when any of these change:
- Source file hash
- Tokenizer hash
- Tokenize function source code

Control with `cache_dir` and `cache_tag` fields on `DatasetConfig`.

### Packing Strategies
- `ExpandSoftPackDataset` — dynamic packing (variable-length)
- `HardPackDataset` — static packing
- `MLLMPretrainHybridPackDataset` — hybrid for multimodal pretraining

---

## Model Architecture

### Supported Model Families
| Type | Models |
|------|--------|
| Dense LLM | Qwen3 (0.6B / 4B / 8B), Qwen2 (7B) |
| MoE LLM | Qwen3-MoE (30B/A3), DeepSeek V3, GptOss (20B / 120B) |
| VLM | Qwen3-VL (4B / 8B), InternVL 3.5 (1B / 8B / 30B/A3), Intern-S1, Intern-S1-mini |

### Adding a New Model
1. Subclass `XTunerBaseModelConfig` (in `xtuner/v1/model/base.py`)
2. Implement `from_hf()` and `save_hf()` for HuggingFace interop
3. Register via `get_model_config()`

---

## Distributed Training

- **FSDP** is the primary sharding strategy (`FSDPConfig`)
- **Sequence parallelism**: Ulysses all-to-all via `ops/comm/`
- **MoE dispatch**: `NaiveDispatcher` or `TorchAll2AllDispatcher`
- **RL orchestration**: Ray actors (`xtuner/v1/ray/`)

FSDP wrapping and checkpoint saving are handled inside `Trainer`; avoid calling FSDP APIs directly in model code.

---

## RL / GRPO

The RL system has four main components inside `RLTrainerConfig`:
- **DataFlowConfig**: controls prompt repetition (`prompt_repeat_k`) and global batch size
- **ReplayBufferConfig**: stores rollouts; configures model/data paths and max lengths
- **RolloutConfig**: selects and configures the inference engine (lmdeploy / vllm / sglang / xtuner)
- **TrainerConfig**: the standard SFT trainer config re-used for policy gradient updates

The GRPO loss lives in `xtuner/v1/rl/grpo/loss.py`.

---

## Loss Functions

| Class | Purpose |
|-------|---------|
| `CELossContext` | Standard cross-entropy |
| `ChunkLoss` | Chunked CE for memory-efficient long sequences |
| `BalancingLoss` | MoE load-balance auxiliary loss |
| `ZLoss` | Router Z-loss to prevent logit collapse |
| `LigerFusedLinearCrossEntropyLossWithWeights` | Fused CUDA (optional, requires Liger) |

Custom losses: subclass `BaseLossConfig` / `BaseLossContext` and pass via `TrainerConfig.loss`.

---

## Float8 Quantization

Enabled via `Float8Handler` in `xtuner/v1/float8/`. Supports per-block and per-tensor FP8. Integrates with FSDP and DCP checkpointing. See `docs/zh_cn/pretrain_sft/advanced_tutorial/float8.md`.

---

## Important Files

| File | Role |
|------|------|
| `xtuner/v1/train/trainer.py` | Main training loop, checkpointing, profiling |
| `xtuner/v1/train/rl_trainer.py` | GRPO training loop |
| `xtuner/v1/model/base.py` | All model configs and base classes |
| `xtuner/v1/datasets/__init__.py` | Data pipeline public API |
| `xtuner/v1/train/cli/sft.py` | `torchrun` entry point for SFT |
| `xtuner/v1/config/__init__.py` | FSDPConfig, OptimConfig, LRConfig, etc. |
| `xtuner/v1/utils/config.py` | `Config.fromfile()` — Python config loader |

---

## Documentation Index

| Path | Topic |
|------|-------|
| `docs/zh_cn/get_started/installation.md` | GPU driver reqs, optional deps (flash-attn, GroupedGEMM) |
| `docs/zh_cn/get_started/sft.md` | LLM SFT quickstart |
| `docs/zh_cn/get_started/mllm_sft.md` | Multimodal SFT quickstart |
| `docs/zh_cn/get_started/grpo.md` | GRPO RL quickstart |
| `docs/zh_cn/pretrain_sft/tutorial/config.md` | Config system deep-dive |
| `docs/zh_cn/pretrain_sft/tutorial/dataset.md` | Dataset caching & custom tokenize |
| `docs/zh_cn/pretrain_sft/tutorial/parallel.md` | Distributed strategies |
| `docs/zh_cn/pretrain_sft/tutorial/resume.md` | Checkpoint resumption |
| `docs/zh_cn/pretrain_sft/advanced_tutorial/float8.md` | FP8 quantization |
| `docs/zh_cn/pretrain_sft/advanced_tutorial/loss.md` | Custom loss functions |
| `docs/zh_cn/rl/tutorial/rl_grpo_trainer.md` | RL trainer config reference |
| `docs/zh_cn/rl/advanced_tutorial/dataset.md` | RL dataset format |
