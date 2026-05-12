"""Compare real MoEBlock grouped-GEMM outputs with and without TP+EP.

The TP+EP path uses the same token layout as ``validate_xtuner_tpep_md.py``:

    rank 0 -> (ep=0, tp=0): A0, A1
    rank 1 -> (ep=0, tp=1): A2, A3
    rank 2 -> (ep=1, tp=0): B0, B1
    rank 3 -> (ep=1, tp=1): B2, B3

Rank 0 additionally runs a non-parallel reference over all 8 tokens with a full
MoEBlock.  Each distributed rank runs the TP+EP dispatcher plus a sharded
MoEBlock.  The local TP+EP outputs are gathered back to rank 0 and compared
against the non-parallel reference in global-rank token order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

# The Triton TMA grouped-GEMM kernel can fail to compile on some local Triton/LLVM
# combinations.  Use XTuner's Cutlass backend by default while still exercising
# the real grouped-GEMM operator path.
os.environ.setdefault("XTUNER_USE_CUTLASS_GROUP_GEMM", "1")

from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock
from xtuner.v1.module.dispatcher.base import NaiveDispatcher
from xtuner.v1.module.dispatcher.torch_all2all_tpep import TorchAll2AllTPEPDispatcher


EP_SIZE = 2
TP_SIZE = 2
DEFAULT_DP_SIZE = 1
N_ROUTED_EXPERTS = 6
HIDDEN_SIZE = 128
MOE_INTERMEDIATE_SIZE = 256
DTYPE = torch.bfloat16
ATOL = 3e-2
RTOL = 3e-2


@dataclass(frozen=True)
class RankCase:
    token_values: tuple[float, ...]
    topk_ids: tuple[tuple[int, int], ...]
    topk_weights: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ParallelInfo:
    global_rank: int
    ep_rank: int
    tp_rank: int
    device: torch.device
    ep_mesh: DeviceMesh
    tp_mesh: DeviceMesh
    ep_group: dist.ProcessGroup
    tp_group: dist.ProcessGroup


CASES: dict[tuple[int, int], RankCase] = {
    # (ep, tp) -> RankCase(token_values, topk_ids, topk_weights)
    (0, 0): RankCase(
        token_values=(10.0, 11.0),
        topk_ids=((0, 4), (3, 1)),
        topk_weights=((0.25, 0.75), (0.4, 0.6)),
    ),
    (0, 1): RankCase(
        token_values=(12.0, 13.0),
        topk_ids=((2, 5), (4, 0)),
        topk_weights=((0.7, 0.3), (0.8, 0.2)),
    ),
    (1, 0): RankCase(
        token_values=(20.0, 21.0),
        topk_ids=((1, 3), (4, 2)),
        topk_weights=((0.2, 0.8), (0.5, 0.5)),
    ),
    (1, 1): RankCase(
        token_values=(22.0, 23.0),
        topk_ids=((5, 0), (3, 1)),
        topk_weights=((0.9, 0.1), (0.35, 0.65)),
    ),
}

CASE_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))


def main() -> None:
    try:
        parallel_info = _init_distributed()
        full_w1w3, full_w2 = _make_full_weights(parallel_info.device)
        local_hidden, local_topk_ids, local_topk_weights = _make_local_inputs(parallel_info)

        local_output = _run_tpep_moeblock(
            parallel_info=parallel_info,
            hidden_states=local_hidden,
            topk_ids=local_topk_ids,
            topk_weights=local_topk_weights,
            full_w1w3=full_w1w3,
            full_w2=full_w2,
        )

        gathered_outputs: list[torch.Tensor] | None = None
        if parallel_info.global_rank == 0:
            gathered_outputs = [torch.empty_like(local_output) for _ in range(dist.get_world_size())]
        dist.gather(local_output.contiguous(), gather_list=gathered_outputs, dst=0)

        if parallel_info.global_rank == 0:
            assert gathered_outputs is not None
            parallel_output = torch.cat(gathered_outputs, dim=0)
            reference_output = _run_single_moeblock_reference(
                device=parallel_info.device,
                full_w1w3=full_w1w3,
                full_w2=full_w2,
            )
            _assert_close(parallel_output, reference_output)
            max_abs_diff = (parallel_output.float() - reference_output.float()).abs().max().item()
            print(
                "真实 MoEBlock grouped-GEMM TP+EP 输出与无并行输出一致，"
                f"max_abs_diff={max_abs_diff:.6e}。"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _init_distributed() -> ParallelInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("真实 MoEBlock TP+EP 校验依赖 CUDA，请在 GPU 上用 torchrun 运行。")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    ep_size = _get_env_int("EP_SIZE", EP_SIZE)
    tp_size = _get_env_int("TP_SIZE", TP_SIZE)
    dp_size = _get_env_int("DP_SIZE", DEFAULT_DP_SIZE)
    if ep_size != EP_SIZE or tp_size != TP_SIZE:
        raise RuntimeError("本脚本固定验证 EP=2, TP=2。")

    world_size = dist.get_world_size()
    if world_size != dp_size * ep_size * tp_size:
        raise RuntimeError(f"需要 world_size = DP*EP*TP = {dp_size * ep_size * tp_size}，实际为 {world_size}。")

    mesh = init_device_mesh(
        "cuda",
        (dp_size, ep_size, tp_size),
        mesh_dim_names=("dp", "ep", "tp"),
    )
    ep_mesh = mesh["ep"]
    tp_mesh = mesh["tp"]
    return ParallelInfo(
        global_rank=dist.get_rank(),
        ep_rank=ep_mesh.get_local_rank(),
        tp_rank=tp_mesh.get_local_rank(),
        device=torch.device("cuda", local_rank),
        ep_mesh=ep_mesh,
        tp_mesh=tp_mesh,
        ep_group=ep_mesh.get_group(),
        tp_group=tp_mesh.get_group(),
    )


def _make_full_weights(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260428)
    w1w3 = torch.randn(
        N_ROUTED_EXPERTS * 2 * MOE_INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        generator=generator,
        device=device,
        dtype=DTYPE,
    )
    w2 = torch.randn(
        N_ROUTED_EXPERTS * HIDDEN_SIZE,
        MOE_INTERMEDIATE_SIZE,
        generator=generator,
        device=device,
        dtype=DTYPE,
    )
    return w1w3 * 0.02, w2 * 0.02


def _make_local_inputs(parallel_info: ParallelInfo) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    case = CASES[(parallel_info.ep_rank, parallel_info.tp_rank)]
    hidden = _make_full_hidden(parallel_info.device)[_local_slice(parallel_info)]
    hidden[:, 0] = torch.tensor(case.token_values, dtype=DTYPE, device=parallel_info.device)
    topk_ids = torch.tensor(case.topk_ids, dtype=torch.long, device=parallel_info.device)
    topk_weights = torch.tensor(case.topk_weights, dtype=torch.float32, device=parallel_info.device)
    return hidden, topk_ids, topk_weights


def _make_full_hidden(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260429)
    hidden = torch.randn(len(CASE_ORDER) * 2, HIDDEN_SIZE, generator=generator, device=device, dtype=DTYPE)
    token_values = [token for key in CASE_ORDER for token in CASES[key].token_values]
    hidden[:, 0] = torch.tensor(token_values, dtype=DTYPE, device=device)
    return hidden


def _local_slice(parallel_info: ParallelInfo) -> slice:
    rank_offset = CASE_ORDER.index((parallel_info.ep_rank, parallel_info.tp_rank))
    start = rank_offset * 2
    return slice(start, start + 2)


def _run_tpep_moeblock(
    *,
    parallel_info: ParallelInfo,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    full_w1w3: torch.Tensor,
    full_w2: torch.Tensor,
) -> torch.Tensor:
    dispatcher = TorchAll2AllTPEPDispatcher(
        n_routed_experts=N_ROUTED_EXPERTS,
        ep_group=parallel_info.ep_group,
        tp_group=parallel_info.tp_group,
        training_dtype="bf16",
    )
    experts = _build_moeblock(parallel_info.device, ep_mesh=parallel_info.ep_mesh, tp_mesh=parallel_info.tp_mesh)
    _load_weights(experts, full_w1w3, full_w2)

    pre_dispatched = dispatcher.dispatch_preprocess(hidden_states=hidden_states, topk_ids=topk_ids)
    dispatched = dispatcher.dispatch(
        pre_dispatched=pre_dispatched,
        topk_weights=topk_weights,
        decoding=False,
    )
    post_dispatched = dispatcher.dispatch_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
    )
    experts_out = experts(
        post_dispatched["hidden_states"],
        post_dispatched["tokens_per_expert"],
        decoding=False,
    )
    pre_combined = dispatcher.combine_preprocess(
        hidden_states=experts_out,
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        decoding=False,
    )
    combined = dispatcher.combine(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        decoding=False,
    )
    post_combined = dispatcher.combine_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        combined=combined,
    )
    return post_combined["hidden_states"]


def _run_single_moeblock_reference(
    *,
    device: torch.device,
    full_w1w3: torch.Tensor,
    full_w2: torch.Tensor,
) -> torch.Tensor:
    hidden_states = _make_full_hidden(device)
    topk_ids = torch.tensor(
        [topk_id for key in CASE_ORDER for topk_id in CASES[key].topk_ids],
        dtype=torch.long,
        device=device,
    )
    topk_weights = torch.tensor(
        [topk_weight for key in CASE_ORDER for topk_weight in CASES[key].topk_weights],
        dtype=torch.float32,
        device=device,
    )

    dispatcher = NaiveDispatcher(n_routed_experts=N_ROUTED_EXPERTS)
    experts = _build_moeblock(device, ep_mesh=None, tp_mesh=None)
    _load_weights(experts, full_w1w3, full_w2)

    pre_dispatched = dispatcher.dispatch_preprocess(hidden_states=hidden_states, topk_ids=topk_ids)
    dispatched = dispatcher.dispatch(
        pre_dispatched=pre_dispatched,
        topk_weights=topk_weights,
        decoding=False,
    )
    post_dispatched = dispatcher.dispatch_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
    )
    experts_out = experts(
        post_dispatched["hidden_states"],
        post_dispatched["tokens_per_expert"],
        decoding=False,
    )
    pre_combined = dispatcher.combine_preprocess(
        hidden_states=experts_out,
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        decoding=False,
    )
    combined = dispatcher.combine(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        decoding=False,
    )
    post_combined = dispatcher.combine_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        combined=combined,
    )
    return post_combined["hidden_states"]


def _build_moeblock(device: torch.device, ep_mesh: DeviceMesh | None, tp_mesh: DeviceMesh | None) -> MoEBlock:
    block = MoEBlock(
        hidden_size=HIDDEN_SIZE,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        n_routed_experts=N_ROUTED_EXPERTS,
        moe_bias=False,
        ep_mesh=ep_mesh,
        tp_mesh=tp_mesh,
        float8_cfg=None,
        moe_act_fn_cfg=MoEActFnConfig(),
    )
    return block.to(device=device, dtype=DTYPE).eval()


def _load_weights(experts: MoEBlock, full_w1w3: torch.Tensor, full_w2: torch.Tensor) -> None:
    with torch.no_grad():
        _copy_weight(experts.fused_w1w3, full_w1w3, fused_gate_up=True)
        _copy_weight(experts.fused_w2, full_w2, fused_gate_up=False)


def _copy_weight(grouped_linear: torch.nn.Module, full_weight: torch.Tensor, *, fused_gate_up: bool) -> None:
    param = grouped_linear.weight
    if isinstance(param, DTensor):
        param.copy_(distribute_tensor(full_weight, param.device_mesh, [Shard(0)]))
    elif getattr(grouped_linear, "tp_enabled", False):
        param.copy_(_slice_tpep_weight(grouped_linear, full_weight, fused_gate_up=fused_gate_up))
    else:
        param.copy_(full_weight)


def _slice_tpep_weight(grouped_linear: torch.nn.Module, full_weight: torch.Tensor, *, fused_gate_up: bool) -> torch.Tensor:
    num_experts = grouped_linear.num_routed_experts
    out_features = grouped_linear.out_features
    in_features = grouped_linear.in_features
    expert_weight = full_weight.view(num_experts, out_features, in_features)
    expert_weight = expert_weight[grouped_linear.local_expert_start : grouped_linear.local_expert_end]

    tp_rank = grouped_linear.tp_rank
    tp_size = grouped_linear.tp_size
    if grouped_linear.parallel_style == "column":
        if fused_gate_up:
            intermediate_size = out_features // 2
            local_intermediate_size = intermediate_size // tp_size
            gate_start = tp_rank * local_intermediate_size
            gate_end = gate_start + local_intermediate_size
            up_start = intermediate_size + gate_start
            up_end = intermediate_size + gate_end
            expert_weight = torch.cat(
                [
                    expert_weight[:, gate_start:gate_end, :],
                    expert_weight[:, up_start:up_end, :],
                ],
                dim=1,
            )
        else:
            local_out_features = out_features // tp_size
            out_start = tp_rank * local_out_features
            out_end = out_start + local_out_features
            expert_weight = expert_weight[:, out_start:out_end, :]
    elif grouped_linear.parallel_style == "row":
        local_in_features = in_features // tp_size
        in_start = tp_rank * local_in_features
        in_end = in_start + local_in_features
        expert_weight = expert_weight[:, :, in_start:in_end]
    else:
        raise RuntimeError(f"Unexpected grouped linear parallel style: {grouped_linear.parallel_style}.")

    return expert_weight.reshape(grouped_linear.weight.shape)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    try:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        max_abs_diff = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(
            "真实 MoEBlock grouped-GEMM TP+EP 输出与无并行输出不一致："
            f"max_abs_diff={max_abs_diff:.6f}, actual_first_col={actual[:, 0].float().tolist()}, "
            f"expected_first_col={expected[:, 0].float().tolist()}"
        ) from exc


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


if __name__ == "__main__":
    main()
