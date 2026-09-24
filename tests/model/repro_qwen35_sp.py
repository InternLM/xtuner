"""Reproduce Qwen3.5 Dense + Muon SP parity with two GPUs and no checkpoint.

Run once with --sp 1 and once with --sp 2; see docs/en/sp_qwen35_parity.md.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path


os.environ.setdefault("XTUNER_DETERMINISTIC", "true")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from xtuner.v1.config import FSDPConfig, MuonConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense4BConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.utils import set_deterministic


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@torch.no_grad()
def initialize(model: torch.nn.Module) -> str:
    """Initialize by parameter name, independently of sharding order and SP."""
    checksum = hashlib.sha256()
    for name, param in model.named_parameters():
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
        full = torch.empty(tuple(param.shape), dtype=torch.float32, device="cuda")
        if name.endswith("A_log"):
            full.uniform_(1, 16, generator=generator).log_()
        elif name.endswith("dt_bias"):
            full.fill_(-4.0)
        elif "norm" in name:
            full.fill_(1.0 if ".self_attn.norm." in name else 0.0)
        elif name.endswith("bias"):
            full.zero_()
        else:
            full.normal_(0, 0.02, generator=generator)
        checksum.update(name.encode())
        checksum.update(full.cpu().numpy().tobytes())
        if isinstance(param, DTensor):
            param.copy_(distribute_tensor(full, param.device_mesh, param.placements))
        else:
            param.copy_(full)
    return checksum.hexdigest()


def replicated_parameters(model: torch.nn.Module) -> list:
    """Select this Dense model's fully replicated, FSDP-ignored FP32 tensors."""
    result = []
    for name, param in model.named_parameters():
        if isinstance(param, DTensor) and all(isinstance(p, Replicate) for p in param.placements):
            assert param.dtype == torch.float32, (name, param.dtype)
            assert name.endswith("A_log") or ".self_attn.norm.weight" in name or name.endswith("conv1d.weight"), name
            assert param.device_mesh.mesh.numel() == dist.get_world_size()
            result.append((name, param))
    assert len(result) in (6, 9), [name for name, _ in result]
    return result


@torch.no_grad()
def snapshot(model: torch.nn.Module, grads: bool = False) -> tuple[dict, dict]:
    values, spreads = {}, {}
    for name, param in replicated_parameters(model):
        value = local(param.grad if grads else param).detach().float().contiguous()
        gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, value)
        stacked = torch.stack(gathered)
        spreads[name] = (stacked.max(0).values - stacked.min(0).values).abs().max().item()
        values[name] = stacked.cpu()
    return values, spreads


@torch.no_grad()
def full_snapshot(model: torch.nn.Module, path: Path, grads: bool = False) -> None:
    values = {}
    for name, param in model.named_parameters():
        value = param.grad if grads else param
        assert value is not None, name
        value = value.full_tensor() if isinstance(value, DTensor) else value
        if dist.get_rank() == 0:
            values[name] = value.detach().float().cpu()
    if dist.get_rank() == 0:
        torch.save(values, path)


def deterministic_state() -> dict:
    import triton
    from torch._inductor import config
    from torch._inductor.runtime import triton_heuristics

    return {
        "entrypoint_called": True,
        "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "debug_mode": torch.get_deterministic_debug_mode(),
        "dynamic_scale_rblock": config.dynamic_scale_rblock,
        "compile_threads": config.compile_threads,
        "triton_autotune_patched": getattr(triton.autotune, "_xtuner_deterministic_patched", False),
        "inductor_reduction_patched": getattr(
            triton_heuristics._reduction_configs, "_xtuner_deterministic_patched", False
        ),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": os.environ.get("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK"),
        "TORCHINDUCTOR_COMPILE_THREADS": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        "XTUNER_DETERMINISTIC": os.environ.get("XTUNER_DETERMINISTIC"),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "entrypoint_source_sha256": hashlib.sha256(inspect.getsource(set_deterministic).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Call the same entry point as the standard Trainer, before CUDA/process-group setup.
    set_deterministic()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    det_state = deterministic_state()
    assert det_state["algorithms_enabled"] and det_state["warn_only"]
    assert det_state["triton_autotune_patched"] and det_state["inductor_reduction_patched"]
    assert det_state["dynamic_scale_rblock"] is False and det_state["compile_threads"] == 1
    assert det_state["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    rank_states = [None] * dist.get_world_size()
    dist.all_gather_object(rank_states, det_state)
    assert all(s == det_state for s in rank_states)
    rank = dist.get_rank()
    print("DETERMINISM_ACTIVE", rank, json.dumps(det_state), flush=True)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    mesh = init_device_mesh("cuda", (2 // args.sp, args.sp), mesh_dim_names=("data_dp", "data_sp"))
    sp_mesh = mesh["data_sp"]
    cfg = Qwen3_5_VLTextDense4BConfig(
        vocab_size=256,
        eos_token_id=0,
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=768,
        tie_word_embeddings=False,
        compile_cfg=False,
        lm_loss_cfg=CELossConfig(mode="eager"),
        attention=MHAConfig(
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=64,
            qk_norm=True,
            rms_norm_type="zero_centered",
            with_gate=True,
            attn_impl="eager_attention",
        ),
        linear_attention=GatedDeltaNetConfig(
            num_value_heads=8,
            num_key_heads=8,
            key_head_dim=64,
            value_head_dim=64,
            conv_kernel_dim=4,
            hidden_act="silu",
            rms_norm_eps=1e-6,
        ),
        rope_parameters_cfg=RopeParametersConfig(
            rope_type="qwen3_vl", rope_theta=10000000.0, partial_rotary_factor=0.25, mrope_section=[3, 3, 2]
        ),
    )
    engine = TrainEngine(
        cfg,
        MuonConfig(lr=1e-3, weight_decay=0.0, enable_all2all=True),
        FSDPConfig(recompute_ratio=0.0, torch_compile=False, reduce_dtype=torch.float32),
    )
    initial_hash = initialize(engine.model)
    corpus = (
        "Stay hydrated by drinking plenty of water throughout the day.\n\nIncorporate regular physical activity into your routine, such as walking or exercising.\n\nEat a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.\n\n"
        * 8
    ).encode()
    assert len(corpus) > 512
    # Byte tokenization retains real text and makes the test independent of model downloads.
    stream = torch.tensor(list(corpus), dtype=torch.long)
    metadata = {
        "determinism": det_state,
        "determinism_all_ranks": rank_states,
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "world_size": 2,
        "optimizer": "Muon",
        "muon_enable_all2all": True,
        "sp": args.sp,
        "steps": args.steps,
        "model_source": inspect.getfile(engine.model.__class__),
        "gradient_reduction_source": inspect.getfile(engine.model.scale_and_reduce_grad),
        "initial_weights_sha256": initial_hash,
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "global_batch_size": 2,
        "sequence_length": 512,
        "tokenization": "UTF-8 bytes",
        "attention": "eager",
        "gdn": "FLA",
        "dtype": "bf16; fp32 gradient reduction",
        "num_parameters": sum(p.numel() for p in engine.model.parameters()),
        "versions": {
            k: importlib.metadata.version(k)
            for k in ["torch", "transformers", "flash-linear-attention", "causal-conv1d"]
        },
        "optimizer_config": engine.optim_cfg.model_dump(mode="json"),
        "model": cfg.model_dump(mode="json"),
    }
    if rank == 0:
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    initial_values, initial_spread = snapshot(engine.model)
    if rank == 0:
        torch.save(initial_values, args.out / "initial_replicas.pt")
        print("INITIAL", initial_hash, initial_spread, flush=True)
    assert deterministic_state() == det_state
    full_snapshot(engine.model, args.out / "initial.pt")
    traces = []
    for step in range(args.steps):
        assert deterministic_state() == det_state
        engine.optimizer.zero_grad(set_to_none=True)
        # The same two documents occur in the same global order for either SP size.
        indices = (torch.arange(2 * 512).reshape(2, 512) + step * 1024) % len(stream)
        global_ids = stream[indices]
        first = rank // args.sp * args.sp
        docs = [global_ids[i : i + 1].cuda() for i in range(first, first + args.sp)]
        seq_ctx = SequenceContext.from_input_ids(tuple(docs))
        labels = torch.cat([torch.cat((doc[:, 1:], torch.full_like(doc[:, :1], -100)), dim=1) for doc in docs], dim=1)
        data = [{"seq_ctx": seq_ctx, "shifted_labels": labels}]
        if args.sp > 1:
            seq_ctx = seq_ctx.split(sequence_parallel_mesh=sp_mesh)
        loss_ctx = engine.model.build_loss_ctx_batch(data, sp_mesh=sp_mesh)[0]
        output = engine.model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        loss = output.loss
        assert loss is not None
        assert torch.isfinite(loss).item(), f"Non-finite loss at step {step}"
        loss.backward()
        raw_grads, raw_spread = snapshot(engine.model, grads=True)
        grad_norm = engine.clip_grad_norm()
        assert torch.isfinite(grad_norm).item(), f"Non-finite gradient norm at step {step}"
        assert engine.optim_cfg.skip_grad_norm_threshold is None
        synced_grads, synced_spread = snapshot(engine.model, grads=True)
        if step == 0:
            full_snapshot(engine.model, args.out / "step0_gradients.pt", grads=True)
        engine.step_optimizer(grad_norm)
        if step == 0:
            full_snapshot(engine.model, args.out / "step0_updated.pt")
        weights, spread = snapshot(engine.model)
        if rank == 0:
            row = {
                "step": step,
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "batch_sha256": digest(global_ids),
                "raw_grad_spread": raw_spread,
                "post_clip_grad_spread": synced_spread,
                "post_update_param_spread": spread,
            }
            with (args.out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            traces.append({"step": step, "raw_grads": raw_grads, "post_clip_grads": synced_grads, "weights": weights})
            print("STEP", json.dumps(row), flush=True)
        del output, loss, raw_grads, synced_grads, weights
    assert deterministic_state() == det_state
    full_snapshot(engine.model, args.out / "final.pt")
    if rank == 0:
        torch.save(traces, args.out / "replica_traces.pt")
        (args.out / "COMPLETE").write_text(f"{args.steps} optimizer updates\n")
        print("PARITY_RUN_COMPLETE", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
