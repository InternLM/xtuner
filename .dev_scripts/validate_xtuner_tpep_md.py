"""验证 XTuner TP+EP all2all 示例的中间顺序。

参数设置（固定）:
    EP = 2, TP = 2 → world_size = EP * TP * DP = 4 * DP_SIZE

Device mesh 排列（mesh_shape=(dp, ep, tp)）:
    rank 0 → (dp=0, ep=0, tp=0)  tokens: A0=10, A1=11
    rank 1 → (dp=0, ep=0, tp=1)  tokens: A2=12, A3=13
    rank 2 → (dp=0, ep=1, tp=0)  tokens: B0=20, B1=21
    rank 3 → (dp=0, ep=1, tp=1)  tokens: B2=22, B3=23

每个 TP rank 持有 N_local=2 个 token，EP+TP 后的流程：

    dispatch_preprocess : 按 expert 排序（每 TP rank 独立）
    dispatch            : EP AlltoAll（每 TP rank 独立，仅路由本 TP 的 token 副本）
    dispatch_postprocess: TP AllGather → 将 TP slices 合并成 M_total token
                          + 按 local expert 再排序（供 grouped GEMM）
    [Expert GEMM]       : 冗余计算（同一 EP rank 内各 TP rank 计算结果相同）
    combine_preprocess  : unpermute → TP ReduceScatterSum → 恢复每 TP rank M_ep_recv
    combine             : EP AlltoAll 逆向
    combine_postprocess : unpermute + topk 加权求和 → [N_local, H]

运行方式：
    EP_SIZE=2 TP_SIZE=2 DP_SIZE=1 torchrun --nproc-per-node=4 \
        .dev_scripts/validate_xtuner_tpep_md.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.module.dispatcher.torch_all2all_tpep import TorchAll2AllTPEPDispatcher


EP_SIZE = 2
TP_SIZE = 2
DEFAULT_DP_SIZE = 1
N_ROUTED_EXPERTS = 6
EXPERTS_PER_RANK = 3
EXPERT_OUTPUT_SCALE = 100.0
HIDDEN_SIZE = 128


@dataclass(frozen=True)
class RankCase:
    token_values: tuple[float, ...]
    topk_ids: tuple[tuple[int, int], ...]
    topk_weights: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RankExpected:
    input_hidden: tuple[float, ...]
    topk_ids: tuple[tuple[int, int], ...]
    pre_hidden: tuple[float, ...]
    pre_row_id_map: tuple[int, ...]
    dispatch_hidden: tuple[float, ...]
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    tokens_per_expert_group: tuple[float, ...]
    output_splits_tp: tuple[int, ...]
    post_hidden: tuple[float, ...]
    post_row_ids_map: tuple[int, ...]
    tokens_per_expert: tuple[float, ...]
    experts_out: tuple[float, ...]
    pre_combine_hidden: tuple[float, ...]
    combine_hidden: tuple[float, ...]
    post_combine_hidden: tuple[float, ...]


@dataclass(frozen=True)
class ParallelInfo:
    global_rank: int
    dp_rank: int
    ep_rank: int
    tp_rank: int
    device: torch.device
    ep_group: dist.ProcessGroup
    tp_group: dist.ProcessGroup


# (ep_rank, tp_rank) → RankCase
# ep0_tp0: A0, A1  |  ep0_tp1: A2, A3
# ep1_tp0: B0, B1  |  ep1_tp1: B2, B3
CASES: dict[tuple[int, int], RankCase] = {
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


# All expected values derived by hand.  See xtuner_tpep.md for the full derivation.
#
# Notation (token value as token id):
#   A0=10, A1=11, A2=12, A3=13  (ep0 source tokens)
#   B0=20, B1=21, B2=22, B3=23  (ep1 source tokens)
#   expert mock: out = in + global_expert_id * 100
EXPECTED: dict[tuple[int, int], RankExpected] = {
    # rank 0: (ep=0, tp=0) — tokens A0, A1
    (0, 0): RankExpected(
        input_hidden=(10.0, 11.0),
        topk_ids=((0, 4), (3, 1)),
        # sorted (topk-slot-first then by expert): A0(e0), A1(e1), A1(e3), A0(e4)
        pre_hidden=(10.0, 11.0, 11.0, 10.0),
        pre_row_id_map=(0, 2, 3, 1),
        # after EP A2A: from self=[A0(e0),A1(e1)], from ep1_tp0=[B0(e1),B1(e2)]
        dispatch_hidden=(10.0, 11.0, 20.0, 21.0),
        input_splits=(2, 2),
        output_splits=(2, 2),
        tokens_per_expert_group=(1.0, 1.0, 0.0, 0.0, 1.0, 1.0),
        output_splits_tp=(4, 4),
        # after TP AllGather (tp0||tp1) + sort by local expert:
        # e0: A0,A3,B2  e1: A1,B0,B3  e2: B1,A2
        post_hidden=(10.0, 13.0, 22.0, 11.0, 20.0, 23.0, 21.0, 12.0),
        post_row_ids_map=(0, 3, 4, 6, 1, 7, 2, 5),
        tokens_per_expert=(3.0, 3.0, 2.0),
        # expert adds global_expert_id * 100
        experts_out=(10.0, 13.0, 22.0, 111.0, 120.0, 123.0, 221.0, 212.0),
        # after ReduceScatterSum — tp0 slice [0:4]
        pre_combine_hidden=(20.0, 222.0, 240.0, 442.0),
        # after EP A2A reverse: from self=[20,222], from ep1_tp0=[622,820]
        combine_hidden=(20.0, 222.0, 622.0, 820.0),
        post_combine_hidden=(620.0, 382.0),
    ),
    # rank 1: (ep=0, tp=1) — tokens A2, A3
    (0, 1): RankExpected(
        input_hidden=(12.0, 13.0),
        topk_ids=((2, 5), (4, 0)),
        # sorted: A3(e0), A2(e2), A3(e4), A2(e5)
        pre_hidden=(13.0, 12.0, 13.0, 12.0),
        pre_row_id_map=(1, 2, 3, 0),
        # after EP A2A: from self=[A3(e0),A2(e2)], from ep1_tp1=[B2(e0),B3(e1)]
        dispatch_hidden=(13.0, 12.0, 22.0, 23.0),
        input_splits=(2, 2),
        output_splits=(2, 2),
        tokens_per_expert_group=(1.0, 0.0, 1.0, 1.0, 1.0, 0.0),
        output_splits_tp=(4, 4),
        # both tp ranks see the same gathered tensor after AllGather
        post_hidden=(10.0, 13.0, 22.0, 11.0, 20.0, 23.0, 21.0, 12.0),
        post_row_ids_map=(0, 3, 4, 6, 1, 7, 2, 5),
        tokens_per_expert=(3.0, 3.0, 2.0),
        experts_out=(10.0, 13.0, 22.0, 111.0, 120.0, 123.0, 221.0, 212.0),
        # after ReduceScatterSum — tp1 slice [4:8]
        pre_combine_hidden=(26.0, 424.0, 44.0, 246.0),
        # after EP A2A reverse: from self=[26,424], from ep1_tp1=[826,1024]
        combine_hidden=(26.0, 424.0, 826.0, 1024.0),
        post_combine_hidden=(604.0, 666.0),
    ),
    # rank 2: (ep=1, tp=0) — tokens B0, B1
    (1, 0): RankExpected(
        input_hidden=(20.0, 21.0),
        topk_ids=((1, 3), (4, 2)),
        # sorted: B0(e1), B1(e2), B0(e3), B1(e4)
        pre_hidden=(20.0, 21.0, 20.0, 21.0),
        pre_row_id_map=(0, 3, 2, 1),
        # after EP A2A: from ep0_tp0=[A1(e3),A0(e4)], from self=[B0(e3),B1(e4)]
        dispatch_hidden=(11.0, 10.0, 20.0, 21.0),
        input_splits=(2, 2),
        output_splits=(2, 2),
        tokens_per_expert_group=(1.0, 1.0, 0.0, 1.0, 1.0, 0.0),
        output_splits_tp=(4, 4),
        # after TP AllGather (tp0||tp1) + sort: e3: A1,B0,B3  e4: A0,B1,A3  e5: A2,B2
        post_hidden=(11.0, 20.0, 23.0, 10.0, 21.0, 13.0, 12.0, 22.0),
        post_row_ids_map=(0, 3, 1, 4, 5, 6, 2, 7),
        tokens_per_expert=(3.0, 3.0, 2.0),
        experts_out=(311.0, 320.0, 323.0, 410.0, 421.0, 413.0, 512.0, 522.0),
        # after ReduceScatterSum — tp0 slice [0:4]
        pre_combine_hidden=(622.0, 820.0, 640.0, 842.0),
        # after EP A2A reverse: from ep0_tp0=[240,442], from self=[640,842]
        combine_hidden=(240.0, 442.0, 640.0, 842.0),
        post_combine_hidden=(560.0, 642.0),
    ),
    # rank 3: (ep=1, tp=1) — tokens B2, B3
    (1, 1): RankExpected(
        input_hidden=(22.0, 23.0),
        topk_ids=((5, 0), (3, 1)),
        # sorted: B2(e0), B3(e1), B3(e3), B2(e5)
        pre_hidden=(22.0, 23.0, 23.0, 22.0),
        pre_row_id_map=(3, 2, 0, 1),
        # after EP A2A: from ep0_tp1=[A3(e4),A2(e5)], from self=[B3(e3),B2(e5)]
        dispatch_hidden=(13.0, 12.0, 23.0, 22.0),
        input_splits=(2, 2),
        output_splits=(2, 2),
        tokens_per_expert_group=(0.0, 1.0, 1.0, 1.0, 0.0, 1.0),
        output_splits_tp=(4, 4),
        post_hidden=(11.0, 20.0, 23.0, 10.0, 21.0, 13.0, 12.0, 22.0),
        post_row_ids_map=(0, 3, 1, 4, 5, 6, 2, 7),
        tokens_per_expert=(3.0, 3.0, 2.0),
        experts_out=(311.0, 320.0, 323.0, 410.0, 421.0, 413.0, 512.0, 522.0),
        # after ReduceScatterSum — tp1 slice [4:8]
        pre_combine_hidden=(826.0, 1024.0, 646.0, 1044.0),
        # after EP A2A reverse: from ep0_tp1=[44,246], from self=[646,1044]
        combine_hidden=(44.0, 246.0, 646.0, 1044.0),
        post_combine_hidden=(944.0, 386.0),
    ),
}


def main() -> None:
    try:
        parallel_info = _init_distributed()
        snapshots = _run_tpep_case(parallel_info)
        _validate(parallel_info, snapshots)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _init_distributed() -> ParallelInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("TorchAll2AllTPEPDispatcher 当前依赖 CUDA，请在 GPU 上用 torchrun 运行。")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    ep_size = _get_env_int("EP_SIZE", EP_SIZE)
    tp_size = _get_env_int("TP_SIZE", TP_SIZE)
    dp_size = _get_env_int("DP_SIZE", DEFAULT_DP_SIZE)

    if ep_size != EP_SIZE or tp_size != TP_SIZE:
        raise RuntimeError("本脚本固定为 EP=2, TP=2。")

    world_size = dist.get_world_size()
    if world_size != dp_size * ep_size * tp_size:
        raise RuntimeError(f"需要 world_size = DP*EP*TP = {dp_size * ep_size * tp_size}，实际为 {world_size}。")

    # mesh_shape=(dp, ep, tp):
    #   rank 0 → (dp=0,ep=0,tp=0), rank 1 → (dp=0,ep=0,tp=1)
    #   rank 2 → (dp=0,ep=1,tp=0), rank 3 → (dp=0,ep=1,tp=1)
    mesh = init_device_mesh(
        "cuda",
        (dp_size, ep_size, tp_size),
        mesh_dim_names=("dp", "ep", "tp"),
    )

    global_rank = dist.get_rank()
    ep_rank = mesh["ep"].get_local_rank()
    tp_rank = mesh["tp"].get_local_rank()
    dp_rank = mesh["dp"].get_local_rank()

    return ParallelInfo(
        global_rank=global_rank,
        dp_rank=dp_rank,
        ep_rank=ep_rank,
        tp_rank=tp_rank,
        device=torch.device("cuda", local_rank),
        ep_group=mesh["ep"].get_group(),
        tp_group=mesh["tp"].get_group(),
    )


@torch.no_grad()
def _run_tpep_case(parallel_info: ParallelInfo) -> dict[str, Any]:
    case = CASES[(parallel_info.ep_rank, parallel_info.tp_rank)]
    hidden_states = torch.zeros(
        (len(case.token_values), HIDDEN_SIZE), dtype=torch.float32, device=parallel_info.device
    )
    hidden_states[:, 0] = torch.tensor(case.token_values, dtype=torch.float32, device=parallel_info.device)
    topk_ids = torch.tensor(case.topk_ids, dtype=torch.long, device=parallel_info.device)
    topk_weights = torch.tensor(case.topk_weights, dtype=torch.float32, device=parallel_info.device)

    dispatcher = TorchAll2AllTPEPDispatcher(
        n_routed_experts=N_ROUTED_EXPERTS,
        ep_group=parallel_info.ep_group,
        tp_group=parallel_info.tp_group,
        training_dtype="bf16",
    )

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

    experts_out = _mock_local_experts(
        hidden_states=post_dispatched["hidden_states"],
        tokens_per_expert=post_dispatched["tokens_per_expert"],
        ep_rank=parallel_info.ep_rank,
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

    return {
        "input_hidden": hidden_states,
        "topk_ids": topk_ids,
        "pre_hidden": pre_dispatched["hidden_states"],
        "pre_row_id_map": pre_dispatched["row_id_map"],
        "dispatch_hidden": dispatched["hidden_states"],
        "input_splits": dispatched["input_splits"],
        "output_splits": dispatched["output_splits"],
        "tokens_per_expert_group": dispatched["tokens_per_expert_group"],
        "output_splits_tp": post_dispatched["output_splits_tp"],
        "post_hidden": post_dispatched["hidden_states"],
        "post_row_ids_map": post_dispatched["row_ids_map"],
        "tokens_per_expert": post_dispatched["tokens_per_expert"],
        "experts_out": experts_out,
        "pre_combine_hidden": pre_combined["hidden_states"],
        "combine_hidden": combined["hidden_states"],
        "post_combine_hidden": post_combined["hidden_states"],
    }


def _mock_local_experts(
    *,
    hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    ep_rank: int,
) -> torch.Tensor:
    local_expert_ids = torch.arange(EXPERTS_PER_RANK, dtype=torch.float32, device=hidden_states.device)
    local_expert_ids = torch.repeat_interleave(local_expert_ids, tokens_per_expert.to(torch.long))
    global_expert_ids = ep_rank * EXPERTS_PER_RANK + local_expert_ids
    return hidden_states + global_expert_ids.view(-1, 1) * EXPERT_OUTPUT_SCALE


def _validate(parallel_info: ParallelInfo, snapshots: dict[str, Any]) -> None:
    key = (parallel_info.ep_rank, parallel_info.tp_rank)
    expected = EXPECTED[key]
    error: AssertionError | None = None

    try:
        if os.getenv("XTUNER_TPEP_DEBUG", "0") == "1":
            _print_snapshots(parallel_info, snapshots)

        _assert_tensor_close(parallel_info, "pre_hidden", snapshots["pre_hidden"], expected.pre_hidden, first_col=True)
        _assert_tensor_close(parallel_info, "pre_row_id_map", snapshots["pre_row_id_map"], expected.pre_row_id_map)
        _assert_tensor_close(
            parallel_info, "dispatch_hidden", snapshots["dispatch_hidden"], expected.dispatch_hidden, first_col=True
        )
        _assert_list_equal(parallel_info, "input_splits", snapshots["input_splits"], expected.input_splits)
        _assert_list_equal(parallel_info, "output_splits", snapshots["output_splits"], expected.output_splits)
        _assert_tensor_close(
            parallel_info,
            "tokens_per_expert_group",
            snapshots["tokens_per_expert_group"],
            expected.tokens_per_expert_group,
        )
        _assert_list_equal(parallel_info, "output_splits_tp", snapshots["output_splits_tp"], expected.output_splits_tp)
        _assert_tensor_close(
            parallel_info, "post_hidden", snapshots["post_hidden"], expected.post_hidden, first_col=True
        )
        _assert_tensor_close(
            parallel_info, "post_row_ids_map", snapshots["post_row_ids_map"], expected.post_row_ids_map
        )
        _assert_tensor_close(
            parallel_info, "tokens_per_expert", snapshots["tokens_per_expert"], expected.tokens_per_expert
        )
        _assert_tensor_close(
            parallel_info, "experts_out", snapshots["experts_out"], expected.experts_out, first_col=True
        )
        _assert_tensor_close(
            parallel_info,
            "pre_combine_hidden",
            snapshots["pre_combine_hidden"],
            expected.pre_combine_hidden,
            first_col=True,
        )
        _assert_tensor_close(
            parallel_info,
            "combine_hidden",
            snapshots["combine_hidden"],
            expected.combine_hidden,
            first_col=True,
        )
        _assert_tensor_close(
            parallel_info,
            "post_combine_hidden",
            snapshots["post_combine_hidden"],
            expected.post_combine_hidden,
            atol=1e-4,
            first_col=True,
        )
    except AssertionError as exc:
        error = exc

    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=parallel_info.device)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)

    if failed.item() != 0:
        if error is not None:
            raise error
        raise AssertionError("其他 rank 的 TP+EP 示例校验失败。")

    if parallel_info.global_rank == 0:
        print("xtuner TP+EP EP=2 TP=2 all2all 示例校验通过。")


def _assert_tensor_close(
    parallel_info: ParallelInfo,
    name: str,
    actual: torch.Tensor,
    expected: tuple[float, ...] | tuple[int, ...],
    *,
    atol: float = 0.0,
    first_col: bool = False,
) -> None:
    actual_1d = actual.detach()
    if first_col and actual_1d.dim() > 1:
        actual_1d = actual_1d[:, 0]
    actual_1d = actual_1d.reshape(-1).to(torch.float32)
    expected_tensor = torch.tensor(expected, dtype=torch.float32, device=actual.device)
    try:
        torch.testing.assert_close(actual_1d, expected_tensor, rtol=0.0, atol=atol)
    except AssertionError as exc:
        raise AssertionError(
            f"global_rank={parallel_info.global_rank} ep_rank={parallel_info.ep_rank} "
            f"tp_rank={parallel_info.tp_rank} 的 {name} 不符合预期："
            f"actual={actual_1d.cpu().tolist()}, expected={expected_tensor.cpu().tolist()}"
        ) from exc


def _assert_list_equal(
    parallel_info: ParallelInfo,
    name: str,
    actual: list[int],
    expected: tuple[int, ...],
) -> None:
    if actual != list(expected):
        raise AssertionError(
            f"global_rank={parallel_info.global_rank} ep_rank={parallel_info.ep_rank} "
            f"tp_rank={parallel_info.tp_rank} 的 {name} 不符合预期："
            f"actual={actual}, expected={list(expected)}"
        )


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _print_snapshots(parallel_info: ParallelInfo, snapshots: dict[str, Any]) -> None:
    hidden_names = {
        "input_hidden",
        "pre_hidden",
        "dispatch_hidden",
        "post_hidden",
        "experts_out",
        "pre_combine_hidden",
        "combine_hidden",
        "post_combine_hidden",
    }
    for name, value in snapshots.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if name in hidden_names and tensor.dim() > 1:
                tensor = tensor[:, 0]
            print(
                f"[global_rank={parallel_info.global_rank} ep_rank={parallel_info.ep_rank} "
                f"tp_rank={parallel_info.tp_rank}] {name}: {tensor.reshape(-1).cpu().tolist()}",
                flush=True,
            )
        else:
            print(
                f"[global_rank={parallel_info.global_rank} ep_rank={parallel_info.ep_rank} "
                f"tp_rank={parallel_info.tp_rank}] {name}: {value}",
                flush=True,
            )


if __name__ == "__main__":
    main()
