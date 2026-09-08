"""Qwen3.5 MoE FSDP benchmark for XTuner grouped GEMM vs SonicMoE.

This is intentionally a standalone profiler rather than a pytest test: the
default invocation builds the complete 40-layer, 35B-parameter text model and
requires one 8-GPU node.

The SonicMoE run uses general (dropless) routing and preserves every native
top-k assignment. Token rounding is deliberately outside the scope of this
benchmark and will be covered by a separate profiler.

The two runs either load the same HuggingFace checkpoint or deterministically
initialize the requested model shape, and reuse the exact same per-rank token
batches. Each measured step performs forward and backward but does not update
parameters. Keeping weights fixed makes every one of the 10 steps a direct
numerical comparison of the two expert implementations. Every measured step is
atomically persisted; accuracy covers all 10 steps while the performance
comparison uses the 10th step only.

Example:
    XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 8 \
      tests/profiler/qwen35_sonicmoe_fsdp_benchmark.py \
      --model-path /path/to/Qwen3.5-35B-A3B \
      --output /path/to/qwen35_sonicmoe_fsdp_benchmark.json \
      --steps 10 --warmup-steps 5 --deterministic
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.attention import GatedDeltaNet, GatedDeltaNetConfig, MHAConfig
from xtuner.v1.module.moe_backend import SonicMoEBackendConfig
from xtuner.v1.utils import default_init_weights, init_params, set_deterministic
from xtuner.v1.utils.misc import clean_param_name, monkey_patch_hf_modules_cache


Backend = Literal["grouped_gemm", "sonicmoe"]

# Values from Qwen/Qwen3.5-35B-A3B config.json.  Failing early here prevents a
# future config drift from silently turning this into a different benchmark.
OFFICIAL_SHAPE = {
    "vocab_size": 248320,
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "attention_head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
}


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _print_rank0(message: str) -> None:
    if _rank() == 0:
        print(message, flush=True)


def _atomic_write_json(output: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so readers never observe a partially written file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary_output, output)


def _progress_path(output: str, backend: Backend, suffix: str) -> Path:
    path = Path(output)
    return path.with_name(f"{path.stem}.{backend}.{suffix}.json")


def _init_distributed(deterministic: bool) -> tuple[int, int, int]:
    if deterministic:
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NCCL_PROTO", "Simple")
        os.environ.setdefault("NCCL_NUM_CHANNELS", "1")
        set_deterministic()

    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    torch.accelerator.set_device_index(local_rank)

    communication_warmup = torch.ones(4, device="cuda")
    dist.all_reduce(communication_warmup)
    return rank, world_size, local_rank


def _requested_shape(args: argparse.Namespace) -> dict[str, int]:
    return {
        "vocab_size": args.vocab_size,
        "num_hidden_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "n_routed_experts": args.num_routed_experts,
        "n_shared_experts": args.num_shared_experts,
        "num_experts_per_tok": args.top_k,
        "moe_intermediate_size": args.expert_intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "attention_head_dim": args.attention_head_dim,
        "linear_num_key_heads": args.linear_num_key_heads,
        "linear_num_value_heads": args.linear_num_value_heads,
        "linear_key_head_dim": args.linear_key_head_dim,
        "linear_value_head_dim": args.linear_value_head_dim,
    }


def _validate_requested_shape(config: Qwen3_5_VLTextMoE35BA3BConfig, args: argparse.Namespace) -> None:
    actual = {
        "vocab_size": config.vocab_size,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "moe_intermediate_size": config.moe_intermediate_size,
        "num_attention_heads": config.attention.num_attention_heads,
        "num_key_value_heads": config.attention.num_key_value_heads,
        "attention_head_dim": config.attention.head_dim,
        "linear_num_key_heads": config.linear_attention.num_key_heads,
        "linear_num_value_heads": config.linear_attention.num_value_heads,
        "linear_key_head_dim": config.linear_attention.key_head_dim,
        "linear_value_head_dim": config.linear_attention.value_head_dim,
    }
    expected = _requested_shape(args)
    if actual != expected:
        raise AssertionError(f"Qwen3.5 benchmark shape mismatch: expected={expected}, actual={actual}")

    if config.layers_type != [
        "linear_attention" if (idx + 1) % 4 else "full_attention" for idx in range(args.num_layers)
    ]:
        raise AssertionError(
            "Qwen3.5 layer schedule must be 3 linear-attention layers followed by 1 full-attention layer"
        )


def _make_token_batches(
    *,
    rank: int,
    count: int,
    seq_len: int,
    micro_batch_size: int,
    vocab_size: int,
    data_seed: int,
) -> tuple[list[torch.Tensor], str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(data_seed + rank * 1_000_003)
    batches: list[torch.Tensor] = []
    digest = hashlib.sha256()
    for _ in range(count):
        tokens = torch.randint(
            0,
            vocab_size,
            (micro_batch_size, seq_len + 1),
            generator=generator,
            dtype=torch.long,
        )
        batches.append(tokens)
        digest.update(tokens.numpy().tobytes())
    return batches, digest.hexdigest()


def _build_config(args: argparse.Namespace, backend: Backend) -> Qwen3_5_VLTextMoE35BA3BConfig:
    config = Qwen3_5_VLTextMoE35BA3BConfig(
        vocab_size=args.vocab_size,
        num_hidden_layers=args.num_layers,
        max_window_layers=args.num_layers,
        hidden_size=args.hidden_size,
        attention=MHAConfig(
            with_gate=True,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            head_dim=args.attention_head_dim,
            qk_norm=True,
            rms_norm_eps=1e-6,
            rms_norm_type="zero_centered",
            sliding_window=1024,
        ),
        linear_attention=GatedDeltaNetConfig(
            num_value_heads=args.linear_num_value_heads,
            num_key_heads=args.linear_num_key_heads,
            key_head_dim=args.linear_key_head_dim,
            value_head_dim=args.linear_value_head_dim,
            conv_kernel_dim=4,
            hidden_act="silu",
            rms_norm_eps=1e-6,
        ),
        n_routed_experts=args.num_routed_experts,
        n_shared_experts=args.num_shared_experts,
        num_experts_per_tok=args.top_k,
        moe_intermediate_size=args.expert_intermediate_size,
        expert_backend=backend,
        sonicmoe_cfg=SonicMoEBackendConfig(routing_mode=args.sonic_routing_mode),
        lm_loss_cfg=CELossConfig(mode="chunk", chunk_size=args.loss_chunk_size),
        compile_cfg=None if args.compile else False,
    )
    config.ep_size = 1
    config.dispatcher = None
    _validate_requested_shape(config, args)
    return config


@torch.no_grad()
def _init_synthetic_weights(model: torch.nn.Module) -> None:
    """Initialize all parameters, including GatedDeltaNet's custom tensors."""
    initialized = default_init_weights(model)
    for module_name, module in model.named_modules():
        if not isinstance(module, GatedDeltaNet):
            continue
        init_params(module.dt_bias, torch.nn.init.ones_)
        init_params(module.A_log, lambda tensor: tensor.uniform_(0.0, 16.0).log_())
        initialized.add(clean_param_name(f"{module_name}.dt_bias"))
        initialized.add(clean_param_name(f"{module_name}.A_log"))

    parameter_names = {clean_param_name(name) for name, _ in model.named_parameters()}
    if missing := parameter_names - initialized:
        raise RuntimeError(f"Synthetic initialization did not cover parameters: {sorted(missing)}")


def _build_model(args: argparse.Namespace, backend: Backend) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = _build_config(args, backend)

    fsdp_config = FSDPConfig(
        ep_size=1,
        cpu_offload=False,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        recompute_ratio=args.recompute_ratio,
        torch_compile=args.compile,
    )

    set_random_seed(args.model_seed, deterministic=args.deterministic)
    with torch.device("meta"):
        model = config.build()
    global_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    estimated_active_parameter_count = round(
        sum(
            (
                parameter.numel() * args.top_k / args.num_routed_experts
                if ".experts." in clean_param_name(name)
                else parameter.numel()
            )
            for name, parameter in model.named_parameters()
        )
    )

    model = model.fully_shard(fsdp_config=fsdp_config)
    set_random_seed(args.model_seed, deterministic=args.deterministic)
    if args.synthetic_weights:
        _init_synthetic_weights(model)
        load_info = {
            "weight_source": "deterministic_synthetic_initialization",
            "model_seed": args.model_seed,
            "global_parameter_count": global_parameter_count,
            "estimated_active_parameter_count": estimated_active_parameter_count,
            "loaded_key_count": 0,
            "unloaded_key_count": 0,
            "ignored_checkpoint_key_count": 0,
        }
    else:
        loaded_keys, unloaded_keys, missing_keys = model.from_hf(args.model_path, strict=False)
        if unloaded_keys:
            examples = sorted(unloaded_keys)[:20]
            raise RuntimeError(f"Checkpoint did not initialize {len(unloaded_keys)} XTuner parameters: {examples}")
        load_info = {
            "weight_source": "huggingface_checkpoint",
            "global_parameter_count": global_parameter_count,
            "estimated_active_parameter_count": estimated_active_parameter_count,
            "loaded_key_count": len(loaded_keys),
            "unloaded_key_count": len(unloaded_keys),
            # Missing keys are checkpoint entries not consumed by the text-only
            # model, chiefly the vision tower and projector.
            "ignored_checkpoint_key_count": len(missing_keys),
        }
    model.train()
    return model, load_info


def _build_step_inputs(
    model: torch.nn.Module,
    token_batches: list[torch.Tensor],
) -> list[tuple[SequenceContext, dict[str, Any]]]:
    """Build and globally calibrate every micro-batch in one training step."""
    raw_batches: list[dict[str, Any]] = []
    for tokens in token_batches:
        raw_batches.append(
            {
                "seq_ctx": SequenceContext.from_input_ids(input_ids=(tokens[:, :-1],), device="cpu"),
                "shifted_labels": tokens[:, 1:],
            }
        )

    loss_contexts = model.build_loss_ctx_batch(raw_batches, sp_mesh=None)
    return [(raw["seq_ctx"].to("cuda"), loss_ctx) for raw, loss_ctx in zip(raw_batches, loss_contexts)]


def _total_loss(outputs: Any) -> torch.Tensor:
    loss = torch.zeros((), dtype=torch.float32, device="cuda")
    for key in type(outputs).model_fields:
        value = getattr(outputs, key)
        if "loss" in key and isinstance(value, torch.Tensor):
            loss = loss + value.float()
    return loss


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    return tensor


def _find_probes(model: torch.nn.Module, num_layers: int) -> dict[str, torch.nn.Parameter]:
    suffixes = (
        "layers.0.gate.weight",
        "layers.0.experts.fused_w1w3.weight",
        "layers.0.experts.fused_w2.weight",
        f"layers.{num_layers - 1}.experts.fused_w2.weight",
    )
    found: dict[str, torch.nn.Parameter] = {}
    for name, parameter in model.named_parameters():
        name = clean_param_name(name)
        for suffix in suffixes:
            if name.endswith(suffix):
                found[suffix] = parameter
    missing = sorted(set(suffixes) - set(found))
    if missing:
        raise RuntimeError(f"Could not resolve numerical probe parameters: {missing}")
    return found


@torch.no_grad()
def _probe_tensors(probes: dict[str, torch.nn.Parameter], *, gradients: bool) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, parameter in probes.items():
        tensor = parameter.grad if gradients else parameter
        if tensor is None:
            raise RuntimeError(f"Probe {name} has no gradient")
        local = _local_tensor(tensor).detach().float()
        values = torch.stack(
            (
                local.sum(dtype=torch.float64),
                torch.square(local.to(torch.float64)).sum(),
                local.abs().max().to(torch.float64),
            )
        ).to("cuda")
        dist.all_reduce(values[:2], op=dist.ReduceOp.SUM)
        dist.all_reduce(values[2:], op=dist.ReduceOp.MAX)
        result[name] = {
            "sum": float(values[0].item()),
            "l2": math.sqrt(max(float(values[1].item()), 0.0)),
            "absmax": float(values[2].item()),
        }
    return result


def _global_loss(value: torch.Tensor) -> float:
    """Sum per-rank CE contributions already normalized by the global token count."""
    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return float(reduced.item())


def _global_max_milliseconds(milliseconds: float) -> float:
    value = torch.tensor(milliseconds, dtype=torch.float64, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def _one_step(
    model: torch.nn.Module,
    token_batches: list[torch.Tensor],
    *,
    timed: bool,
) -> tuple[float, float]:
    step_inputs = _build_step_inputs(model, token_batches)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    total_loss = torch.zeros((), dtype=torch.float32, device="cuda")
    for seq_ctx, loss_ctx in step_inputs:
        outputs = model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        if hasattr(outputs, "free_nongrad_feature"):
            outputs.free_nongrad_feature()
        loss = _total_loss(outputs)
        loss.backward()
        total_loss += loss.detach()
    end.record()
    end.synchronize()
    elapsed_ms = _global_max_milliseconds(start.elapsed_time(end)) if timed else 0.0
    global_loss = _global_loss(total_loss)
    return global_loss, elapsed_ms


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(int(math.ceil(quantile * len(ordered))) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _resume_signature(args: argparse.Namespace, backend: Backend) -> dict[str, Any]:
    """Fields that must match before persisted benchmark steps can be reused."""
    return {
        "backend": backend,
        "model_shape": _requested_shape(args),
        "synthetic_weights": args.synthetic_weights,
        "model_path": args.model_path,
        "model_seed": args.model_seed,
        "data_seed": args.data_seed,
        "sequence_length": args.seq_len,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "recompute_ratio": args.recompute_ratio,
        "compile": args.compile,
        "sonic_routing_mode": args.sonic_routing_mode,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _run_backend(
    args: argparse.Namespace,
    backend: Backend,
    token_batches: list[torch.Tensor],
    input_hashes: list[str],
) -> dict[str, Any]:
    result_path = _progress_path(args.output, backend, "result")
    progress_path = _progress_path(args.output, backend, "progress")
    signature = _resume_signature(args, backend)
    if args.resume and result_path.exists():
        completed = _load_json(result_path)
        legacy_compatible = (
            completed.get("backend") == backend
            and completed.get("input_sha256_by_rank") == input_hashes
            and len(completed.get("steps", ())) == args.steps
            and len(completed.get("gradient_probes", ())) == args.steps
        )
        if (
            completed.get("resume_signature") == signature and len(completed.get("steps", ())) == args.steps
        ) or legacy_compatible:
            _print_rank0(f"[{backend}] reusing completed result from {result_path}")
            dist.barrier()
            return completed

    weight_source = "deterministic synthetic weights" if args.synthetic_weights else args.model_path
    _print_rank0(f"[{backend}] building full FSDP model with {weight_source} ...")
    load_started = time.perf_counter()
    model, load_info = _build_model(args, backend)
    probes = _find_probes(model, args.num_layers)
    parameter_probes = _probe_tensors(probes, gradients=False)
    dist.barrier()
    load_seconds = time.perf_counter() - load_started

    micro_batches_per_step = args.gradient_accumulation_steps
    for warmup_step in range(args.warmup_steps):
        begin = warmup_step * micro_batches_per_step
        end = begin + micro_batches_per_step
        _one_step(model, token_batches[begin:end], timed=False)
        _print_rank0(f"[{backend}] warmup {warmup_step + 1}/{args.warmup_steps}")

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    step_milliseconds: list[float] = []
    gradient_probes: list[dict[str, dict[str, float]]] = []
    step_records: list[dict[str, float | int]] = []
    resumed_max_allocated_gib = 0.0
    resumed_max_reserved_gib = 0.0

    if args.resume and progress_path.exists():
        progress = _load_json(progress_path)
        compatible = (
            progress.get("schema_version") == 2
            and progress.get("resume_signature") == signature
            and progress.get("input_sha256_by_rank") == input_hashes
            and progress.get("parameter_probes") == parameter_probes
            and len(progress.get("steps", ())) == len(progress.get("gradient_probes", ()))
            and len(progress.get("steps", ())) <= args.steps
        )
        if compatible:
            step_records = progress["steps"]
            losses = [float(record["loss"]) for record in step_records]
            step_milliseconds = [float(record["step_milliseconds"]) for record in step_records]
            gradient_probes = progress["gradient_probes"]
            resumed_max_allocated_gib = float(progress.get("max_memory_allocated_gib_so_far", 0.0))
            resumed_max_reserved_gib = float(progress.get("max_memory_reserved_gib_so_far", 0.0))
            _print_rank0(f"[{backend}] resumed {len(step_records)}/{args.steps} persisted steps")
        else:
            _print_rank0(f"[{backend}] ignoring incompatible legacy progress at {progress_path}")

    for step in range(len(step_records), args.steps):
        global_step = args.warmup_steps + step
        begin = global_step * micro_batches_per_step
        end = begin + micro_batches_per_step
        loss, elapsed_ms = _one_step(model, token_batches[begin:end], timed=True)
        losses.append(loss)
        step_milliseconds.append(elapsed_ms)
        gradient_probes.append(_probe_tensors(probes, gradients=True))
        samples_per_second = args.global_batch_size * 1000.0 / elapsed_ms
        tokens_per_second = args.global_batch_size * args.seq_len * 1000.0 / elapsed_ms
        tokens_per_second_per_card = tokens_per_second / dist.get_world_size()
        step_records.append(
            {
                "step": step + 1,
                "loss": loss,
                "step_milliseconds": elapsed_ms,
                "global_samples_per_second": samples_per_second,
                "global_tokens_per_second": tokens_per_second,
                "tokens_per_second_per_card": tokens_per_second_per_card,
            }
        )
        if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == args.steps:
            _print_rank0(
                f"[{backend}] step={step + 1}/{args.steps} loss={loss:.8f} "
                f"time={elapsed_ms:.3f}ms "
                f"samples/s={samples_per_second:.4f} "
                f"global_tokens/s={tokens_per_second:.2f} "
                f"tokens/s/card={tokens_per_second_per_card:.2f}"
            )
        current_peaks = torch.tensor(
            (
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            ),
            dtype=torch.float64,
            device="cuda",
        )
        dist.all_reduce(current_peaks, op=dist.ReduceOp.MAX)
        if _rank() == 0:
            current_mean_ms = statistics.fmean(step_milliseconds)
            current_max_allocated_gib = max(resumed_max_allocated_gib, float(current_peaks[0].item()))
            current_max_reserved_gib = max(resumed_max_reserved_gib, float(current_peaks[1].item()))
            _atomic_write_json(
                progress_path,
                {
                    "schema_version": 2,
                    "resume_signature": signature,
                    "backend": backend,
                    "routing_mode": args.sonic_routing_mode if backend == "sonicmoe" else "native_topk",
                    "steps_completed": step + 1,
                    "steps_total": args.steps,
                    "latest_loss": loss,
                    "mean_step_ms_so_far": current_mean_ms,
                    "global_samples_per_second_so_far": args.global_batch_size * 1000.0 / current_mean_ms,
                    "global_tokens_per_second_so_far": (
                        args.global_batch_size * args.seq_len * 1000.0 / current_mean_ms
                    ),
                    "tokens_per_second_per_card_so_far": (
                        args.global_batch_size * args.seq_len * 1000.0 / current_mean_ms / dist.get_world_size()
                    ),
                    "steps": step_records,
                    "gradient_probes": gradient_probes,
                    "input_sha256_by_rank": input_hashes,
                    "parameter_probes": parameter_probes,
                    "max_memory_allocated_gib_so_far": current_max_allocated_gib,
                    "max_memory_reserved_gib_so_far": current_max_reserved_gib,
                },
            )

    allocated = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, dtype=torch.float64, device="cuda")
    reserved = torch.tensor(torch.cuda.max_memory_reserved() / 2**30, dtype=torch.float64, device="cuda")
    dist.all_reduce(allocated, op=dist.ReduceOp.MAX)
    dist.all_reduce(reserved, op=dist.ReduceOp.MAX)

    max_allocated_gib = max(resumed_max_allocated_gib, float(allocated.item()))
    max_reserved_gib = max(resumed_max_reserved_gib, float(reserved.item()))
    mean_ms = statistics.fmean(step_milliseconds)
    result = {
        "schema_version": 2,
        "resume_signature": signature,
        "backend": backend,
        "routing_mode": args.sonic_routing_mode if backend == "sonicmoe" else "native_topk",
        "load_seconds": load_seconds,
        "load_info": load_info,
        "input_sha256_by_rank": input_hashes,
        "parameter_probes": parameter_probes,
        "losses": losses,
        "gradient_probes": gradient_probes,
        "step_milliseconds": step_milliseconds,
        "steps": step_records,
        "last_step": step_records[-1],
        "performance": {
            "mean_step_ms": mean_ms,
            "median_step_ms": statistics.median(step_milliseconds),
            "p90_step_ms": _percentile(step_milliseconds, 0.90),
            "global_samples_per_second": args.global_batch_size * 1000.0 / mean_ms,
            "global_tokens_per_second": args.global_batch_size * args.seq_len * 1000.0 / mean_ms,
            "tokens_per_second_per_card": (
                args.global_batch_size * args.seq_len * 1000.0 / mean_ms / dist.get_world_size()
            ),
            "max_memory_allocated_gib": max_allocated_gib,
            "max_memory_reserved_gib": max_reserved_gib,
        },
    }

    if _rank() == 0:
        _atomic_write_json(result_path, result)

    del probes, model
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    dist.barrier()
    return result


def _relative_difference(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(lhs), 1e-12)


def _compare(native: dict[str, Any], sonic: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if native["input_sha256_by_rank"] != sonic["input_sha256_by_rank"]:
        raise AssertionError("Native and SonicMoE runs did not use identical input batches")
    if native["parameter_probes"] != sonic["parameter_probes"]:
        raise AssertionError("Native and SonicMoE runs did not start from identical checkpoint weights")

    if len(native["losses"]) != args.steps or len(sonic["losses"]) != args.steps:
        raise AssertionError(
            f"Expected {args.steps} measured losses per backend, got "
            f"native={len(native['losses'])}, sonicmoe={len(sonic['losses'])}"
        )
    loss_abs = [abs(a - b) for a, b in zip(native["losses"], sonic["losses"])]
    loss_rel = [_relative_difference(a, b) for a, b in zip(native["losses"], sonic["losses"])]
    loss_within_tolerance = [
        diff <= args.accuracy_atol + args.accuracy_rtol * abs(reference)
        for diff, reference in zip(loss_abs, native["losses"])
    ]
    per_step_loss = [
        {
            "step": step,
            "native_loss": native_loss,
            "sonicmoe_loss": sonic_loss,
            "absolute_difference": absolute_difference,
            "relative_difference": relative_difference,
            "within_tolerance": within_tolerance,
        }
        for step, (native_loss, sonic_loss, absolute_difference, relative_difference, within_tolerance) in enumerate(
            zip(native["losses"], sonic["losses"], loss_abs, loss_rel, loss_within_tolerance), start=1
        )
    ]
    gradient_abs: list[float] = []
    gradient_rel: list[float] = []
    gradient_l2_abs: list[float] = []
    gradient_l2_rel: list[float] = []
    for native_step, sonic_step in zip(native["gradient_probes"], sonic["gradient_probes"]):
        for name in native_step:
            for statistic_name in ("sum", "l2", "absmax"):
                a = native_step[name][statistic_name]
                b = sonic_step[name][statistic_name]
                gradient_abs.append(abs(a - b))
                gradient_rel.append(_relative_difference(a, b))
                if statistic_name == "l2":
                    gradient_l2_abs.append(abs(a - b))
                    gradient_l2_rel.append(_relative_difference(a, b))

    native_ms = native["last_step"]["step_milliseconds"]
    sonic_ms = sonic["last_step"]["step_milliseconds"]
    speedup_ratio = native_ms / sonic_ms
    world_size = dist.get_world_size()
    accuracy_comparable = args.sonic_routing_mode == "general"
    return {
        "strictly_identical_inputs": True,
        "strictly_identical_checkpoint_probes": True,
        "strictly_identical_initial_parameter_probes": True,
        "accuracy_comparable": accuracy_comparable,
        "accuracy_note": (
            "general routing preserves native top-k assignments"
            if accuracy_comparable
            else "token rounding intentionally changes expert assignments; numerical equivalence is not expected"
        ),
        "loss": {
            "measurement_scope": f"steps_1_to_{args.steps}_inclusive",
            "num_steps": args.steps,
            "mean_absolute_difference": statistics.fmean(loss_abs),
            "max_absolute_difference": max(loss_abs),
            "mean_relative_difference": statistics.fmean(loss_rel),
            "max_relative_difference": max(loss_rel),
            "within_tolerance_all_steps": all(loss_within_tolerance),
            "per_step": per_step_loss,
        },
        "gradient_probe": {
            "mean_absolute_difference": statistics.fmean(gradient_abs),
            "max_absolute_difference": max(gradient_abs),
            "mean_relative_difference": statistics.fmean(gradient_rel),
            "max_relative_difference": max(gradient_rel),
            "l2_norm_mean_absolute_difference": statistics.fmean(gradient_l2_abs),
            "l2_norm_max_absolute_difference": max(gradient_l2_abs),
            "l2_norm_mean_relative_difference": statistics.fmean(gradient_l2_rel),
            "l2_norm_max_relative_difference": max(gradient_l2_rel),
            "note": (
                "L2-norm differences are the stable gradient comparison; relative differences of signed sums "
                "can become arbitrarily large when cancellation makes the reference sum nearly zero."
            ),
        },
        "performance": {
            "measurement_scope": f"step_{args.steps}_only",
            "measurement_step": args.steps,
            "native_step_ms": native_ms,
            "sonicmoe_step_ms": sonic_ms,
            "native_global_samples_per_second": args.global_batch_size * 1000.0 / native_ms,
            "sonicmoe_global_samples_per_second": args.global_batch_size * 1000.0 / sonic_ms,
            "native_global_tokens_per_second": args.global_batch_size * args.seq_len * 1000.0 / native_ms,
            "sonicmoe_global_tokens_per_second": args.global_batch_size * args.seq_len * 1000.0 / sonic_ms,
            "native_tokens_per_second_per_card": (
                args.global_batch_size * args.seq_len * 1000.0 / native_ms / world_size
            ),
            "sonicmoe_tokens_per_second_per_card": (
                args.global_batch_size * args.seq_len * 1000.0 / sonic_ms / world_size
            ),
            "speedup_ratio": speedup_ratio,
            "latency_reduction_percent": (1.0 - sonic_ms / native_ms) * 100.0,
            "throughput_improvement_percent": (speedup_ratio - 1.0) * 100.0,
        },
    }


def _render_comparison_report(
    native: dict[str, Any],
    sonic: dict[str, Any],
    comparison: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """Render a concise report suitable for both logs and a Markdown file."""
    native_last = native["last_step"]
    sonic_last = sonic["last_step"]
    perf = comparison["performance"]
    accuracy = comparison["loss"]
    accuracy_rows = [
        (
            f"| {row['step']} | {row['native_loss']:.8f} | {row['sonicmoe_loss']:.8f} | "
            f"{row['absolute_difference']:.8e} | {row['relative_difference']:.8e} | "
            f"{row['within_tolerance']} |"
        )
        for row in accuracy["per_step"]
    ]
    lines = [
        f"# {args.model_label}：原生 MoE 与 SonicMoE Dropless 对比",
        "",
        f"精度统计：第 1–{args.steps} 步；性能统计：仅第 {args.steps} 步。",
        f"模型结构：layers={args.num_layers}，hidden_size={args.hidden_size}，"
        f"routed_experts={args.num_routed_experts}，shared_experts={args.num_shared_experts}，"
        f"expert_size={args.expert_intermediate_size}，top_k={args.top_k}。",
        f"实际参数量：总参数={native['load_info']['global_parameter_count']:,}，"
        f"估算激活参数={native['load_info']['estimated_active_parameter_count']:,}。",
        f"配置：seq_len={args.seq_len}，global_batch_size={args.global_batch_size}，"
        f"micro_batch_size={args.micro_batch_size}，gradient_accumulation_steps={args.gradient_accumulation_steps}，"
        "FSDP=8，EP/SP/TP=1，BF16，Token Rounding=关闭。",
        f"权重来源：{'确定性合成初始化' if args.synthetic_weights else args.model_path}。",
        "",
        f"| 后端 | 第 {args.steps} 步耗时 (ms) | 第 {args.steps} 步全局样本吞吐 (samples/s) | "
        f"第 {args.steps} 步单卡 TPS (tokens/s/card) | 第 {args.steps} 步全局 Token 吞吐 (tokens/s) | "
        f"{args.steps} 步平均 Loss | 峰值显存 (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Qwen3.5 原生 grouped_gemm | {native_last['step_milliseconds']:.3f} | "
            f"{native_last['global_samples_per_second']:.4f} | "
            f"{native_last['tokens_per_second_per_card']:.2f} | "
            f"{native_last['global_tokens_per_second']:.2f} | {statistics.fmean(native['losses']):.8f} | "
            f"{native['performance']['max_memory_allocated_gib']:.3f} |"
        ),
        (
            f"| SonicMoE general/dropless | {sonic_last['step_milliseconds']:.3f} | "
            f"{sonic_last['global_samples_per_second']:.4f} | "
            f"{sonic_last['tokens_per_second_per_card']:.2f} | "
            f"{sonic_last['global_tokens_per_second']:.2f} | {statistics.fmean(sonic['losses']):.8f} | "
            f"{sonic['performance']['max_memory_allocated_gib']:.3f} |"
        ),
        "",
        f"- SonicMoE 加速比：**{perf['speedup_ratio']:.4f}x**",
        f"- 吞吐提升：**{perf['throughput_improvement_percent']:.2f}%**",
        f"- 步耗时下降：**{perf['latency_reduction_percent']:.2f}%**",
        f"- {args.steps} 步最大 Loss 绝对差：**{accuracy['max_absolute_difference']:.8e}**",
        f"- {args.steps} 步 Loss 全部通过容差：**{accuracy['within_tolerance_all_steps']}**",
        f"- 梯度探针 L2 范数最大相对差：" f"**{comparison['gradient_probe']['l2_norm_max_relative_difference']:.8e}**",
        "- 输入数据 SHA256 和初始参数探针均要求完全一致。",
        "- 本测试执行前向和反向但不更新参数，以避免优化器累计差异干扰 kernel 数值对比。",
        "",
        f"## 第 1–{args.steps} 步 Loss 精度明细",
        "",
        "| Step | 原生 Loss | SonicMoE Loss | 绝对差 | 相对差 | 通过容差 |",
        "|---:|---:|---:|---:|---:|:---:|",
        *accuracy_rows,
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN3_5_MOE_PATH"),
        help="HuggingFace checkpoint path; defaults to QWEN3_5_MOE_PATH.",
    )
    parser.add_argument("--model-label", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument(
        "--synthetic-weights",
        action="store_true",
        help="Initialize weights deterministically instead of loading --model-path; intended for shape benchmarks.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible atomically persisted formal steps after an external job interruption.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=OFFICIAL_SHAPE["hidden_size"])
    parser.add_argument("--vocab-size", type=int, default=OFFICIAL_SHAPE["vocab_size"])
    parser.add_argument("--num-attention-heads", type=int, default=OFFICIAL_SHAPE["num_attention_heads"])
    parser.add_argument("--num-key-value-heads", type=int, default=OFFICIAL_SHAPE["num_key_value_heads"])
    parser.add_argument("--attention-head-dim", type=int, default=OFFICIAL_SHAPE["attention_head_dim"])
    parser.add_argument("--linear-num-key-heads", type=int, default=OFFICIAL_SHAPE["linear_num_key_heads"])
    parser.add_argument("--linear-num-value-heads", type=int, default=OFFICIAL_SHAPE["linear_num_value_heads"])
    parser.add_argument("--linear-key-head-dim", type=int, default=OFFICIAL_SHAPE["linear_key_head_dim"])
    parser.add_argument("--linear-value-head-dim", type=int, default=OFFICIAL_SHAPE["linear_value_head_dim"])
    parser.add_argument("--num-routed-experts", type=int, default=OFFICIAL_SHAPE["n_routed_experts"])
    parser.add_argument("--num-shared-experts", type=int, default=OFFICIAL_SHAPE["n_shared_experts"])
    parser.add_argument("--expert-intermediate-size", type=int, default=OFFICIAL_SHAPE["moe_intermediate_size"])
    parser.add_argument("--top-k", type=int, default=OFFICIAL_SHAPE["num_experts_per_tok"])
    parser.add_argument("--declared-total-parameters", type=int)
    parser.add_argument("--declared-active-parameters", type=int)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument("--recompute-ratio", type=float, default=1.0)
    parser.add_argument("--model-seed", type=int, default=20260820)
    parser.add_argument("--data-seed", type=int, default=20260821)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--accuracy-atol", type=float, default=2e-2)
    parser.add_argument("--accuracy-rtol", type=float, default=2e-2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--sonic-routing-mode",
        choices=("general",),
        default="general",
        help="SonicMoE routing mode. This profiler intentionally supports dropless general routing only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        args.steps <= 0
        or args.warmup_steps < 0
        or args.seq_len <= 0
        or args.global_batch_size <= 0
        or args.micro_batch_size <= 0
    ):
        raise ValueError(
            "steps, seq-len, global-batch-size and micro-batch-size must be positive; "
            "warmup-steps must be non-negative"
        )
    positive_shape_values = {
        "num-layers": args.num_layers,
        "hidden-size": args.hidden_size,
        "vocab-size": args.vocab_size,
        "num-attention-heads": args.num_attention_heads,
        "num-key-value-heads": args.num_key_value_heads,
        "attention-head-dim": args.attention_head_dim,
        "linear-num-key-heads": args.linear_num_key_heads,
        "linear-num-value-heads": args.linear_num_value_heads,
        "linear-key-head-dim": args.linear_key_head_dim,
        "linear-value-head-dim": args.linear_value_head_dim,
        "num-routed-experts": args.num_routed_experts,
        "expert-intermediate-size": args.expert_intermediate_size,
        "top-k": args.top_k,
    }
    invalid_shape_values = {name: value for name, value in positive_shape_values.items() if value <= 0}
    if args.num_shared_experts < 0:
        invalid_shape_values["num-shared-experts"] = args.num_shared_experts
    if invalid_shape_values:
        raise ValueError(f"Model shape values must be positive (shared experts may be zero): {invalid_shape_values}")
    if args.num_attention_heads % args.num_key_value_heads:
        raise ValueError("num-attention-heads must be divisible by num-key-value-heads")
    if args.linear_num_value_heads % args.linear_num_key_heads:
        raise ValueError("linear-num-value-heads must be divisible by linear-num-key-heads")
    if args.top_k > args.num_routed_experts:
        raise ValueError("top-k cannot exceed num-routed-experts")
    if not args.synthetic_weights:
        if args.model_path is None:
            raise FileNotFoundError(
                "Set --model-path or QWEN3_5_MOE_PATH to a complete HuggingFace checkpoint, "
                "or use --synthetic-weights."
            )
        if not Path(args.model_path).is_dir():
            raise FileNotFoundError(f"Qwen3.5 checkpoint directory does not exist: {args.model_path}.")

    monkey_patch_hf_modules_cache()
    rank, world_size, local_rank = _init_distributed(args.deterministic)
    if world_size != 8:
        raise RuntimeError(f"This benchmark requires exactly 8 processes/GPUs, got world_size={world_size}")
    sequences_per_micro_step = world_size * args.micro_batch_size
    if args.global_batch_size % sequences_per_micro_step != 0:
        raise ValueError(
            "global-batch-size must be divisible by world-size * micro-batch-size; "
            f"got {args.global_batch_size} % ({world_size} * {args.micro_batch_size})"
        )
    args.gradient_accumulation_steps = args.global_batch_size // sequences_per_micro_step
    set_random_seed(args.model_seed, deterministic=args.deterministic)

    if rank == 0:
        _atomic_write_json(
            Path(args.output).with_name(f"{Path(args.output).stem}.status.json"),
            {
                "status": "started",
                "model": args.model_label,
                "model_shape": _requested_shape(args),
                "weight_source": (
                    "deterministic_synthetic_initialization" if args.synthetic_weights else args.model_path
                ),
                "sequence_length": args.seq_len,
                "global_batch_size": args.global_batch_size,
                "micro_batch_size": args.micro_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "steps": args.steps,
                "warmup_steps": args.warmup_steps,
                "accuracy_scope": f"steps_1_to_{args.steps}_inclusive",
                "performance_scope": f"step_{args.steps}_only",
                "routing": "general_dropless",
                "token_rounding": False,
            },
        )

    batches, local_input_hash = _make_token_batches(
        rank=rank,
        count=(args.warmup_steps + args.steps) * args.gradient_accumulation_steps,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        vocab_size=args.vocab_size,
        data_seed=args.data_seed,
    )
    input_hashes: list[str | None] = [None] * world_size
    dist.all_gather_object(input_hashes, local_input_hash)
    resolved_hashes = [value for value in input_hashes if value is not None]

    _print_rank0(
        "[benchmark] "
        f"world_size={world_size} local_rank0={local_rank} model={args.model_label} "
        f"layers={args.num_layers} hidden_size={args.hidden_size} vocab_size={args.vocab_size} "
        f"routed_experts={args.num_routed_experts} shared_experts={args.num_shared_experts} "
        f"expert_size={args.expert_intermediate_size} top_k={args.top_k} seq_len={args.seq_len} "
        f"global_batch_size={args.global_batch_size} micro_batch_size={args.micro_batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"steps={args.steps} warmup={args.warmup_steps} FSDP=on EP=off SP=off TP=off "
        f"compile={args.compile} recompute_ratio={args.recompute_ratio} routing={args.sonic_routing_mode}"
    )
    native = _run_backend(args, "grouped_gemm", batches, resolved_hashes)
    sonic = _run_backend(args, "sonicmoe", batches, resolved_hashes)
    comparison = _compare(native, sonic, args)
    report = _render_comparison_report(native, sonic, comparison, args)

    payload = {
        "schema_version": 2,
        "model": args.model_label,
        "model_shape": _requested_shape(args),
        "declared_parameter_scale": {
            "total_parameters": args.declared_total_parameters,
            "active_parameters": args.declared_active_parameters,
        },
        "scope": (f"complete {args.num_layers}-layer text model; forward+backward; fixed weights; no optimizer step"),
        "parallelism": {"fsdp": 8, "ep": 1, "sp": 1, "tp": 1},
        "batching": {
            "sequence_length": args.seq_len,
            "global_batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_tokens_per_step": args.global_batch_size * args.seq_len,
        },
        "dtype": "bfloat16",
        "arguments": vars(args),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "world_size": world_size,
        },
        "native": native,
        "sonicmoe": sonic,
        "comparison": comparison,
    }
    if rank == 0:
        output = Path(args.output)
        _atomic_write_json(output, payload)

        report_output = output.with_suffix(".md")
        temporary_report = report_output.with_suffix(report_output.suffix + ".tmp")
        temporary_report.write_text(report + "\n", encoding="utf-8")
        os.replace(temporary_report, report_output)
        _atomic_write_json(
            output.with_name(f"{output.stem}.status.json"),
            {
                "status": "completed",
                "benchmark_output": str(output),
                "benchmark_report": str(report_output),
            },
        )
        print("\n" + report + "\n", flush=True)
        print("BENCHMARK_RESULT=" + json.dumps(comparison, ensure_ascii=False), flush=True)
        print(f"BENCHMARK_OUTPUT={output}", flush=True)
        print(f"BENCHMARK_REPORT={report_output}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"BENCHMARK_ERROR rank={os.environ.get('RANK', 'unknown')}: {error}", file=sys.stderr, flush=True)
        raise
