import os

# Replace the env's broken Triton-3.4-targeted MoE expert GEMM kernel
# (`m_grouped_gemm_TMA_triton3_4.py` PassManager::run failure under Triton 3.5.1)
# with the `grouped_gemm` PyPI CUTLASS backend, which is compile-friendly via the
# `moe::gmm` torch.library.custom_op shim in `group_gemm_cutlass.py`.
os.environ.setdefault("XTUNER_USE_CUTLASS_GROUP_GEMM", "1")
# Stage each decoder layer's HC-expanded activation on CPU during the forward,
# fetch back on the backward. Halves peak HBM at the cost of D2H/H2D bandwidth;
# with 4× HC expansion this is the main lever that lets the full 256-expert /
# pack_max_length=32768 setup fit on 8× H200.
os.environ.setdefault("XTUNER_ACTIVATION_OFFLOAD", "1")

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.deepseek_v4 import DeepSeekV4Config
from xtuner.v1.train import TrainerConfig


# DEEPSEEK_V4_PATH should point at a directory that holds the BF16 DeepSeek-V4-Flash
# release plus its tokenizer files. The BF16 dequant of the 46-shard FP4/FP8 release
# lives at /mnt/shared-storage-user/llmrazor-share/yehaochen/model/DeepSeek-V4-Flash
# on the shared storage (109 safetensor shards, 542 GB). `from_hf` reads the local
# `config.json` to recover the 44-entry `compress_ratios` list and all V4 hyper-params.
DEEPSEEK_V4_PATH = os.environ["DEEPSEEK_V4_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]


# Use `from_hf` rather than the default-arg constructor so the per-layer
# `compress_ratios` (length = num_hidden_layers + 1) and other release-specific
# fields (num_hash_layers, swiglu_limit, attn_sink dims) are picked up from the
# checkpoint instead of relying on the Config defaults.
moe_cfg = DeepSeekV4Config.from_hf(DEEPSEEK_V4_PATH)
moe_cfg.num_hidden_layers = 4
# # 256 experts at 2048 inter × 4096 hidden × 3 (w1/w2/w3) × 2 bytes ≈ 12.6 GB per
# # layer, and we keep a fp32 master copy → 25 GB per layer. With 2 layers and HC's
# # 5D residual mix tensor (~8 GB per packed sequence) plus master weights, even
# # 140 GB H200 is tight. Cut experts to 16 to leave headroom.
# moe_cfg.n_routed_experts = 16
# Honor the release config's `num_hash_layers` (3 in the BF16 dequant). With
# `num_hidden_layers=2` both layers fall under `layer_idx < num_hash_layers`
# → both are hash-routed (matching the ckpt's `tid2eid` and absent
# `e_score_correction_bias` for these layers). The all-hash side effect — no
# layer accumulating routing stats — is handled inside `DeepSeekV4._forward`
# (skips `aux_loss.finalize` when `num_hash_layers >= num_hidden_layers`).

# EP works now that MoE.fully_shard patches nn.Linear.forward + nn.Embedding.forward
# to .to_local() the weight when it's a Replicate-on-ep DTensor (created by
# _replicate_other_params), plus DSA's raw .weight accesses (attn_sink, wo_a.weight)
# do the unwrap inline. Bumping ep_size from 1 → 2 cuts per-rank expert memory in
# half (256 experts → 128 owned per ep group), trading off for an all2all dispatcher
# on the MoE forward.
# EP sweep on this 8× H200 / 2-layer toy:
#   ep=1  pack=8192  →  62.50 GB max_mem, 11.0 s/step  ← works for multi-step
#   ep=2  pack=4096  →  29.55 GB max_mem, but only HALF the ranks complete
#                       step 1's clip_grad_norm; the other half hangs in
#                       cal_grad_norm's `dist.all_reduce` on a 2D-composed
#                       (Replicate-on-ep × Shard-on-fsdp) DTensor placement
#                       that splits the all_reduce group asymmetrically. The
#                       "Training finished" we previously saw at ep=2 was
#                       torchrun killing the stuck half after the other half
#                       exited debug_skip_save. EP needs cal_grad_norm + the
#                       patched_linear_forward grad path to be EP-aware
#                       (tracked as known issue).
# Until EP desync is fixed, run with ep=1 so multi-step (step 2+) actually
# completes on all ranks.
moe_cfg.ep_size = 4
moe_cfg.dispatcher = "deepep"
# DSA backend selector for the per-layer ``sparse_attn`` call (see
# ``DSAConfig.backend``):
#   * "native"    — pure-PyTorch reference (default, always works)
#   * "flash_mla" — Phase-1: FlashMLA forward + native-recompute backward
#   * "cudnn"     — Phase-2: FlashMLA forward + cudnn SparseAttentionBackward
moe_cfg.attention.backend = "cudnn"
# Compile is now safe — cutlass group_gemm is annotated with @torch.library.custom_op
# (compile-friendly), and HC + DSA helpers are pure-Tensor.
# Temporarily disabled: under pack=8192 + intra_layer_micro_batch=1 +
# recompute_ratio=1.0 some backward path allocates a 130 GiB fp32 tensor.
# The 06:00 run with compile_cfg=False reached step 50 at max_mem 114 GB so
# the baseline fits — debug what compile_cfg=True is changing in the eager
# code path that adds 130 GB on top.
moe_cfg.compile_cfg = True

optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    # `FSDPConfig.torch_compile` is deprecated (1.1.0) and now acts as a master
    # OFF switch — setting it to False overrides `moe_cfg.compile_cfg` and
    # disables compile entirely. Omit it (defaults to True so the model-level
    # `compile_cfg` controls the actual targets).
    cpu_offload=False,
    ep_size=moe_cfg.ep_size,
    # Activation checkpointing on. Together with XTUNER_ACTIVATION_OFFLOAD this
    # gives recompute-on-backward + CPU-resident inter-layer hidden states.
    recompute_ratio=1.0,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        # `chat_template="internlm2"` is the default — it wraps each turn with
        # `<|im_start|>role\n...<|im_end|>` markers. DeepSeek-V4's tokenizer has
        # no such tokens, so applying the InternLM2 template causes the markers
        # to fall back to per-byte tokens that V4's embed/lm_head never saw
        # → per-token CE > ln(vocab) (worse than random init), ~22 instead of
        # the ~12 expected from-scratch initial loss. Use the matching DeepSeek
        # family template instead (V4 reuses V3's chat schema).
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=4096, chat_template="deepseekv3"),
    },
]

# With recompute_ratio=1.0, every backward recomputes the full forward of a
# layer — that re-allocates the HC 5D mix tensor (~4 GB at pack=32768 × hc_mult=4
# × hidden=4096 bf16), plus DSA grads, plus per-expert grads, simultaneously on
# top of the 137 GB of params/fp32-master/Adam states already resident. 32768
# OOMs on 140 GB. 8192 leaves ~30 GB headroom for the recompute peak.
# XTUNER_ACTIVATION_OFFLOAD additionally stages the inter-layer hidden state on
# CPU during the original forward, so the next layer's recompute sees a clean
# slate when it starts.
dataloader_config = DataloaderConfig(pack_max_length=4096)

loss_cfg = CELossConfig()


trainer = TrainerConfig(
    load_from=DEEPSEEK_V4_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=DEEPSEEK_V4_PATH,
    global_batch_size=16,
    work_dir="/tmp/deepseek_v4_flash",
    seed=0,
    strict_load=False,
    # Force at least 2 training steps. `total_epoch=1` over alpaca_tiny (20 tiny
    # samples, ~1864 tokens) only yields 1 packed batch per rank → default
    # `total_step = len(dataloader) * total_epoch = 1`. Set explicitly to 2 so
    # `_data_iter`'s `while cur_step < total_step` keeps yielding; on StopIteration
    # it bumps epoch and re-iterates the same data (so step 2 trains on the same
    # sequence).
    total_step=10000,
    total_epoch=None,
    # Smoke-test V4 `_micro_batch_forward`: list-input path is taken when > 1.
    # Sequential per-layer per-MB loop (see DeepSeekV4._micro_batch_forward).
    intra_layer_micro_batch=1,
    # debug_skip_save left off — ep=1 save path is collective-safe (it was only
    # broken under ep_size>1 where the fsdp_mesh got split into ep groups).
    # Memory profiling: dumps `rank{r}_memory_snapshot.pickle` under work_dir/profile/
    # for the listed step. Load in PyTorch's https://pytorch.org/memory_viz to see
    # the timeline of allocations. profile_time also writes a trace per step.
    # trainer._cur_step is 0-indexed at check time (incremented AFTER the step),
    # so profile_step=0 fires on the first step.
    # Profile step 3 (steady-state — step 1 is compile warmup, step 2 may still
    # see specialization recompile). Drops a per-rank ``rank{r}_trace.json``
    # under ``work_dir/profiling_time/`` viewable in chrome://tracing or
    # https://ui.perfetto.dev/. profile_time defaults to True.
    # Profile step 0 (== first training step) so we can read peak memory even
    # when the rest of the run OOMs before reaching steady state. ``profile_memory``
    # dumps ``rank{r}_memory_snapshot.pickle`` under ``work_dir/profiling_memory/step-0/``,
    # viewable at https://pytorch.org/memory_viz.
    # profile_step=4,
    # profile_memory=True,
)
