"""Shared helpers for the decoupled EP/FSDP ("dp2ep") numerics and checkpoint checks.

The GPU pytest gates in ``tests/engine/test_decoupled_ep_fsdp_train_engine.py`` and the manual
experiment scripts under ``tests/model/`` (``run_decoupled_ep_fsdp_numerics.py`` and
``run_decoupled_ep_fsdp_ckpt.py``) build the same tiny random Qwen3-MoE, feed it the same token
stream and compare layouts with the same functions, so a number in a report and a gate threshold
mean the same thing.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor

from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.utils.device import get_device
from xtuner.v1.utils.dtensor import cal_total_norm


DEVICE = get_device()

MODEL_SIZES: dict[str, dict[str, int]] = {
    # ~108M params: numerics (fast, 50 steps in < 10 s)
    "tiny": {
        "vocab_size": 4096,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "moe_intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "num_experts": 64,
        "num_experts_per_tok": 4,
    },
    # ~3.4B params (3.2B in routed experts): step-time / memory comparison
    "medium": {
        "vocab_size": 32768,
        "hidden_size": 2048,
        "intermediate_size": 4096,
        "moe_intermediate_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 4,
    },
}


class LayoutMode(TypedDict):
    """One EP/FSDP layout under test.

    ``name`` labels the layout in results, ``ep`` is the expert-parallel size, ``decouple`` selects
    ``FSDPConfig.decouple_ep_fsdp`` and ``hsdp`` is ``FSDPConfig.hsdp_sharding_size`` (``None`` for
    plain FSDP over the world).
    """

    name: str
    ep: int
    decouple: bool
    hsdp: int | None


def parse_mode(spec: str) -> LayoutMode:
    """Parse a ``<name>:key=value[,key=value...]`` layout spec.

    Args:
        spec (str): Mode spec with keys ``ep`` (int), ``decouple`` (0/1) and ``hsdp`` (int).

    Returns:
        LayoutMode: The parsed layout; unspecified keys default to ``ep=1, decouple=0, hsdp=None``.
    """
    name, _, kv = spec.partition(":")
    mode: LayoutMode = {"name": name, "ep": 1, "decouple": False, "hsdp": None}
    for item in filter(None, kv.split(",")):
        key, value = item.split("=")
        if key == "ep":
            mode["ep"] = int(value)
        elif key == "decouple":
            mode["decouple"] = bool(int(value))
        elif key == "hsdp":
            mode["hsdp"] = int(value)
        else:
            raise ValueError(f"unknown mode key {key}")
    return mode


def build_hf_checkpoint(path: Path, seed: int, model_size: str) -> None:
    """Write a randomly initialised Qwen3-MoE HF checkpoint to ``path``.

    Args:
        path (Path): Output directory (created by ``save_pretrained``).
        seed (int): Seed for the random initialisation.
        model_size (str): A key of :data:`MODEL_SIZES`.
    """
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        **MODEL_SIZES[model_size],
        max_position_embeddings=8192,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=1000000.0,
        use_sliding_window=False,
        max_window_layers=MODEL_SIZES[model_size]["num_hidden_layers"],
    )
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(config)
    model.save_pretrained(path, safe_serialization=True)


def make_batch(step: int, rank: int, vocab_size: int, seq_len: int) -> ModelItem:
    """Build the deterministic random token batch of ``(step, rank)``.

    Args:
        step (int): Training step; together with ``rank`` it seeds the batch.
        rank (int): Data-parallel rank.
        vocab_size (int): Vocabulary size to sample tokens from.
        seq_len (int): Number of input tokens.

    Returns:
        ModelItem: Sequence context and CE loss context on :data:`DEVICE`.
    """
    generator = torch.Generator().manual_seed(100_000 + step * 1024 + rank)
    tokens = torch.randint(0, vocab_size, (1, seq_len + 1), generator=generator)
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:].to(DEVICE)
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    loss_cfg = CELossConfig()
    loss_ctx = loss_cfg.build(data={"shifted_labels": labels}, sp_mesh=None)
    loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
    return ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})


def build_engine(
    mode: LayoutMode,
    hf_path: Path,
    tag: str,
    dispatcher: str = "all2all",
    fp8: bool = False,
    lr: float = 1e-4,
) -> TrainEngine:
    """Build a ``TrainEngine`` for ``mode`` without loading weights.

    Args:
        mode (LayoutMode): Layout under test.
        hf_path (Path): HF checkpoint the model config is read from.
        tag (str): Unique tag for the mesh prefix; every engine alive in a process needs its own.
        dispatcher (str): MoE dispatcher (``"all2all"`` or ``"deepep"``).
        fp8 (bool): Enable tile-wise float8 for linear and grouped linear layers.
        lr (float): AdamW learning rate.

    Returns:
        TrainEngine: Sharded engine with freshly built (uninitialised) parameters.
    """
    model_cfg = Qwen3MoEConfig.from_hf(hf_path)
    model_cfg.ep_size = mode["ep"]
    model_cfg.dispatcher = dispatcher
    model_cfg.compile_cfg = False
    model_cfg.mesh_prefix = f"{tag}_{mode['name']}"
    if fp8:
        model_cfg.float8_cfg = Float8Config(
            scaling_granularity_gemm=ScalingGranularity.TILEWISE,
            scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
        )
    fsdp_cfg = FSDPConfig(
        ep_size=mode["ep"],
        decouple_ep_fsdp=mode["decouple"],
        hsdp_sharding_size=mode["hsdp"],
        torch_compile=False,
    )
    return TrainEngine(model_cfg=model_cfg, optim_cfg=AdamWConfig(lr=lr), fsdp_cfg=fsdp_cfg)


def train(engine: TrainEngine, steps: Iterable[int], vocab_size: int, seq_len: int) -> list[float]:
    """Run ``steps`` optimizer steps on the shared token stream.

    Args:
        engine (TrainEngine): Engine to train.
        steps (Iterable[int]): Step indices; they select the batches via :func:`make_batch`.
        vocab_size (int): Vocabulary size of the model.
        seq_len (int): Tokens per batch.

    Returns:
        list[float]: ``reduced_llm_loss`` of every step.
    """
    losses = []
    for step in steps:
        info = engine.train_step([make_batch(step, dist.get_rank(), vocab_size, seq_len)])
        grad_norm = engine.clip_grad_norm()
        engine.step_optimizer(grad_norm)
        losses.append(float(info["logs_info"]["reduced_llm_loss"]))
    return losses


def run_mode(
    mode: LayoutMode,
    hf_path: Path,
    steps: int,
    seq_len: int,
    lr: float,
    dispatcher: str,
    fp8: bool,
    grad_norm_steps: Sequence[int] = (0, 1, 25),
    tag: str = "numerics",
) -> dict[str, Any]:
    """Train ``mode`` from the HF checkpoint and record losses, grad norms, memory and step time.

    Args:
        mode (LayoutMode): Layout under test.
        hf_path (Path): HF checkpoint to load.
        steps (int): Number of optimizer steps.
        seq_len (int): Tokens per batch.
        lr (float): AdamW learning rate.
        dispatcher (str): MoE dispatcher.
        fp8 (bool): Enable tile-wise float8.
        grad_norm_steps (Sequence[int]): Steps whose per-parameter grad norms are recorded.
        tag (str): Mesh-prefix tag passed to :func:`build_engine`.

    Returns:
        dict[str, Any]: ``losses``, ``grad_norms`` (per step), ``param_grad_norms`` (per recorded
        step, keyed by parameter name), ``memory`` (per-rank MiB) and ``step_time_s``.
    """
    rank = dist.get_rank()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    base_allocated = torch.cuda.memory_allocated()

    engine = build_engine(mode, hf_path, tag, dispatcher=dispatcher, fp8=fp8, lr=lr)
    engine.from_hf(hf_path=hf_path, strict=True)
    vocab_size = engine.model.config.vocab_size
    torch.cuda.synchronize()
    result: dict[str, Any] = {
        "mode": mode,
        "memory": param_memory(engine.model),
        "losses": [],
        "grad_norms": [],
        "param_grad_norms": {},
        "step_time_s": None,
    }
    result["memory"]["allocated_after_load_mib"] = (torch.cuda.memory_allocated() - base_allocated) / 2**20

    step_times = []
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = make_batch(step, rank, vocab_size, seq_len)
        info = engine.train_step([batch])
        grad_norm = engine.clip_grad_norm()
        if step in grad_norm_steps:
            result["param_grad_norms"][str(step)] = per_param_grad_norms(engine.model)
        engine.step_optimizer(grad_norm)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        result["losses"].append(float(info["logs_info"]["reduced_llm_loss"]))
        result["grad_norms"].append(float(grad_norm))
        if step == 0:
            result["memory"]["allocated_after_step0_mib"] = (torch.cuda.memory_allocated() - base_allocated) / 2**20

    warm = step_times[5:] if len(step_times) > 10 else step_times
    result["step_time_s"] = sum(warm) / len(warm)
    result["step_times_s"] = step_times
    result["memory"]["peak_allocated_mib"] = (torch.cuda.max_memory_allocated() - base_allocated) / 2**20
    result["memory"]["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20

    release(engine)
    return result


def param_memory(model: torch.nn.Module) -> dict[str, float]:
    """Per-rank parameter bytes of the local shards, split into routed experts and everything else.

    Args:
        model (torch.nn.Module): Sharded model.

    Returns:
        dict[str, float]: ``expert_param_mib`` and ``dense_param_mib``.
    """
    expert_bytes = 0
    dense_bytes = 0
    for name, param in model.named_parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        nbytes = local.numel() * local.element_size()
        if ".experts" in name:
            expert_bytes += nbytes
        else:
            dense_bytes += nbytes
    return {"expert_param_mib": expert_bytes / 2**20, "dense_param_mib": dense_bytes / 2**20}


def per_param_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """Global L2 norm of every parameter's gradient, keyed by the unwrapped parameter name.

    Args:
        model (torch.nn.Module): Sharded model after ``backward``.

    Returns:
        dict[str, float]: Parameter name (without ``_checkpoint_wrapped_module.``) to grad norm.
    """
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if isinstance(param.grad, DTensor):
            norm = cal_total_norm([param.grad], norm_type=2.0, foreach=True, dtype=torch.float32)
        else:
            norm = torch.linalg.vector_norm(param.grad, 2.0, dtype=torch.float32)
        norms[clean_name(name)] = float(norm)
    return norms


def clean_name(name: str) -> str:
    """Strip the activation-checkpoint wrapper prefix from a parameter name.

    Args:
        name (str): Name from ``named_parameters``.

    Returns:
        str: The name without ``_checkpoint_wrapped_module.``.
    """
    return name.replace("_checkpoint_wrapped_module.", "")


def rel_diff(a: float, b: float) -> float:
    """Relative difference ``|a - b| / |b|`` (``b`` is the reference).

    Args:
        a (float): Value under test.
        b (float): Reference value.

    Returns:
        float: The relative difference; the denominator is clamped to ``1e-12``.
    """
    return abs(a - b) / max(abs(b), 1e-12)


def max_rel_diff(values: Sequence[float], reference: Sequence[float]) -> float:
    """Largest element-wise :func:`rel_diff` of two equally long sequences.

    Args:
        values (Sequence[float]): Values under test.
        reference (Sequence[float]): Reference values.

    Returns:
        float: ``max(rel_diff(v, r))`` over the zipped sequences.
    """
    if len(values) != len(reference):
        raise ValueError(f"length mismatch: {len(values)} vs {len(reference)}")
    return max(rel_diff(a, b) for a, b in zip(values, reference))


def load_hf_dir(path: Path) -> dict[str, torch.Tensor]:
    """Load every ``*.safetensors`` file of an HF checkpoint directory.

    Args:
        path (Path): Checkpoint directory.

    Returns:
        dict[str, torch.Tensor]: All tensors keyed by their checkpoint name.
    """
    tensors: dict[str, torch.Tensor] = {}
    for file in sorted(path.glob("*.safetensors")):
        tensors.update(load_file(str(file)))
    return tensors


def compare_hf(lhs: Path, rhs: Path) -> dict[str, Any]:
    """Compare two HF checkpoint directories tensor by tensor.

    Args:
        lhs (Path): Reference checkpoint.
        rhs (Path): Checkpoint under test.

    Returns:
        dict[str, Any]: Key counts, missing / extra keys, max absolute and relative differences,
        the number of mismatched elements and the worst key; ``{"error": ...}`` on a shape mismatch.
    """
    a = load_hf_dir(lhs)
    b = load_hf_dir(rhs)
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    max_abs = 0.0
    max_rel = 0.0
    mismatched = 0
    total = 0
    worst = ""
    for key in sorted(set(a) & set(b)):
        x = a[key].to(torch.float32)
        y = b[key].to(torch.float32)
        if x.shape != y.shape:
            return {"error": f"shape mismatch for {key}: {tuple(x.shape)} vs {tuple(y.shape)}"}
        diff = (x - y).abs()
        cur = float(diff.max())
        if cur > max_abs:
            max_abs, worst = cur, key
        max_rel = max(max_rel, float((diff / x.abs().clamp_min(1e-6)).max()))
        mismatched += int((diff > 0).sum())
        total += diff.numel()
    return {
        "keys": len(set(a) & set(b)),
        "missing_in_rhs": missing[:5],
        "extra_in_rhs": extra[:5],
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mismatched_elements": mismatched,
        "total_elements": total,
        "worst_key": worst,
    }


def assert_hf_dirs_close(lhs: Path, rhs: Path, rtol: float, atol: float, label: str) -> None:
    """Fail unless two HF checkpoint directories hold the same keys, shapes and close values.

    Args:
        lhs (Path): Reference checkpoint.
        rhs (Path): Checkpoint under test.
        rtol (float): Relative tolerance passed to ``torch.testing.assert_close``.
        atol (float): Absolute tolerance passed to ``torch.testing.assert_close``.
        label (str): Prefix for the failure message.
    """
    a = load_hf_dir(lhs)
    b = load_hf_dir(rhs)
    if set(a) != set(b):
        raise AssertionError(
            f"{label}: key sets differ; missing in rhs {sorted(set(a) - set(b))[:5]}, "
            f"extra in rhs {sorted(set(b) - set(a))[:5]}"
        )
    for key in sorted(a):
        x, y = a[key], b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise AssertionError(f"{label}: {key} has {tuple(y.shape)}/{y.dtype}, expected {tuple(x.shape)}/{x.dtype}")
        torch.testing.assert_close(y.float(), x.float(), rtol=rtol, atol=atol, msg=lambda m: f"{label}: {key}: {m}")


def release(engine: TrainEngine) -> None:
    """Drop an engine and return its device memory to the allocator.

    Args:
        engine (TrainEngine): Engine to release; the caller must not use it afterwards.
    """
    del engine
    gc.collect()
    torch.cuda.empty_cache()
