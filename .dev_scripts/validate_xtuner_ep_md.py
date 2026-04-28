"""验证 xtuner_ep.md 中 EP all2all 示例的中间顺序。

运行方式：
    EP_SIZE=2 DP_SIZE=4 torchrun --nproc-per-node=8 .dev_scripts/validate_xtuner_ep_md.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# 只从 xtuner 引入被验证的 dispatcher，避免依赖无关的模型/训练类。
from xtuner.v1.module.dispatcher.torch_all2all import TorchAll2AllDispatcher


EP_SIZE = 2
DEFAULT_DP_SIZE = 4
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
    post_hidden: tuple[float, ...]
    post_row_ids_map: tuple[int, ...]
    tokens_per_expert: tuple[float, ...]
    pre_combine_hidden: tuple[float, ...]
    combine_hidden: tuple[float, ...]
    post_combine_hidden: tuple[float, ...]


@dataclass(frozen=True)
class ParallelInfo:
    global_rank: int
    dp_rank: int
    ep_rank: int
    device: torch.device
    ep_group: dist.ProcessGroup


CASES: dict[int, RankCase] = {
    0: RankCase(
        token_values=(10.0, 11.0, 12.0, 13.0),
        topk_ids=((0, 4), (3, 1), (2, 5), (4, 0)),
        topk_weights=((0.25, 0.75), (0.4, 0.6), (0.7, 0.3), (0.8, 0.2)),
    ),
    1: RankCase(
        token_values=(20.0, 21.0, 22.0, 23.0),
        topk_ids=((1, 3), (4, 2), (5, 0), (3, 1)),
        topk_weights=((0.2, 0.8), (0.5, 0.5), (0.9, 0.1), (0.35, 0.65)),
    ),
}


EXPECTED: dict[int, RankExpected] = {
    0: RankExpected(
        input_hidden=(10.0, 11.0, 12.0, 13.0),
        topk_ids=((0, 4), (3, 1), (2, 5), (4, 0)),
        pre_hidden=(10.0, 13.0, 11.0, 12.0, 11.0, 10.0, 13.0, 12.0),
        pre_row_id_map=(0, 4, 3, 6, 5, 2, 7, 1),
        dispatch_hidden=(10.0, 13.0, 11.0, 12.0, 22.0, 20.0, 23.0, 21.0),
        input_splits=(4, 4),
        output_splits=(4, 4),
        tokens_per_expert_group=(2.0, 1.0, 1.0, 1.0, 2.0, 1.0),
        post_hidden=(10.0, 13.0, 22.0, 11.0, 20.0, 23.0, 12.0, 21.0),
        post_row_ids_map=(0, 1, 3, 6, 2, 4, 5, 7),
        tokens_per_expert=(3.0, 3.0, 2.0),
        pre_combine_hidden=(10.0, 13.0, 111.0, 212.0, 22.0, 120.0, 123.0, 221.0),
        combine_hidden=(10.0, 13.0, 111.0, 212.0, 311.0, 410.0, 413.0, 512.0),
        post_combine_hidden=(310.0, 191.0, 302.0, 333.0),
    ),
    1: RankExpected(
        input_hidden=(20.0, 21.0, 22.0, 23.0),
        topk_ids=((1, 3), (4, 2), (5, 0), (3, 1)),
        pre_hidden=(22.0, 20.0, 23.0, 21.0, 20.0, 23.0, 21.0, 22.0),
        pre_row_id_map=(1, 6, 7, 5, 4, 3, 0, 2),
        dispatch_hidden=(11.0, 10.0, 13.0, 12.0, 20.0, 23.0, 21.0, 22.0),
        input_splits=(4, 4),
        output_splits=(4, 4),
        tokens_per_expert_group=(1.0, 2.0, 1.0, 2.0, 1.0, 1.0),
        post_hidden=(11.0, 20.0, 23.0, 10.0, 13.0, 21.0, 12.0, 22.0),
        post_row_ids_map=(0, 3, 4, 6, 1, 2, 5, 7),
        tokens_per_expert=(3.0, 3.0, 2.0),
        pre_combine_hidden=(311.0, 410.0, 413.0, 512.0, 320.0, 323.0, 421.0, 522.0),
        combine_hidden=(22.0, 120.0, 123.0, 221.0, 320.0, 323.0, 421.0, 522.0),
        post_combine_hidden=(280.0, 321.0, 472.0, 193.0),
    ),
}


def main() -> None:
    try:
        parallel_info = _init_distributed()
        snapshots = _run_xtuner_ep_case(parallel_info)
        _validate(parallel_info, snapshots)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _init_distributed() -> ParallelInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("TorchAll2AllDispatcher 当前依赖 CUDA，请在 GPU 上用 torchrun 运行。")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    ep_size = _get_env_int("EP_SIZE", EP_SIZE)
    dp_size = _get_env_int("DP_SIZE", DEFAULT_DP_SIZE)
    world_size = dist.get_world_size()
    if ep_size != EP_SIZE:
        raise RuntimeError("xtuner_ep.md 的示例固定为 EP=2。")
    if world_size != ep_size * dp_size:
        raise RuntimeError(
            f"当前配置要求 world_size = EP_SIZE * DP_SIZE = {ep_size * dp_size}，实际为 {world_size}。"
        )

    # 与 MoE 初始化保持一致：mesh_shape=(dp, ep)，EP 组为连续 rank 对。
    ep_mesh = init_device_mesh(
        "cuda",
        (dp_size, ep_size),
        mesh_dim_names=("dp", "ep"),
    )["ep"]

    global_rank = dist.get_rank()
    return ParallelInfo(
        global_rank=global_rank,
        dp_rank=global_rank // ep_size,
        ep_rank=ep_mesh.get_local_rank(),
        device=torch.device("cuda", local_rank),
        ep_group=ep_mesh.get_group(),
    )


@torch.no_grad()
def _run_xtuner_ep_case(parallel_info: ParallelInfo) -> dict[str, Any]:
    case = CASES[parallel_info.ep_rank]
    hidden_states = torch.zeros((len(case.token_values), HIDDEN_SIZE), dtype=torch.float32, device=parallel_info.device)
    hidden_states[:, 0] = torch.tensor(case.token_values, dtype=torch.float32, device=parallel_info.device)
    topk_ids = torch.tensor(case.topk_ids, dtype=torch.long, device=parallel_info.device)
    topk_weights = torch.tensor(case.topk_weights, dtype=torch.float32, device=parallel_info.device)

    dispatcher = TorchAll2AllDispatcher(
        n_routed_experts=N_ROUTED_EXPERTS,
        training_dtype="bf16",
        process_group=parallel_info.ep_group,
    )

    # 对应文档 1：source rank 内按 global expert 排序。
    pre_dispatched = dispatcher.dispatch_preprocess(hidden_states=hidden_states, topk_ids=topk_ids)

    # 对应文档 2：第一次 all2all，发往目标 EP rank。
    dispatched = dispatcher.dispatch(
        pre_dispatched=pre_dispatched,
        topk_weights=topk_weights,
        decoding=False,
    )

    # 对应文档 3：destination rank 内按 local expert 重新分组。
    post_dispatched = dispatcher.dispatch_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
    )

    # 用 expert id 改写输出，确保最后的 topK 加权还原也被验证。
    experts_out = _mock_local_experts(
        hidden_states=post_dispatched["hidden_states"],
        tokens_per_expert=post_dispatched["tokens_per_expert"],
        ep_rank=parallel_info.ep_rank,
    )

    # 对应文档 5：恢复 all2all receive 顺序。
    pre_combined = dispatcher.combine_preprocess(
        hidden_states=experts_out,
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        decoding=False,
    )

    # 对应文档 6：第二次 all2all，把 expert 输出送回 source rank。
    combined = dispatcher.combine(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        decoding=False,
    )

    # 对应文档 7：用第一次 row_id_map 加权合并 topK。
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
        "post_hidden": post_dispatched["hidden_states"],
        "post_row_ids_map": post_dispatched["row_ids_map"],
        "tokens_per_expert": post_dispatched["tokens_per_expert"],
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
    expected = EXPECTED[parallel_info.ep_rank]
    error: AssertionError | None = None

    try:
        if os.getenv("XTUNER_EP_DEBUG", "0") == "1":
            _print_snapshots(parallel_info, snapshots)
        _assert_tensor_close(parallel_info, "pre_hidden", snapshots["pre_hidden"], expected.pre_hidden, first_col=True)
        _assert_tensor_close(parallel_info, "pre_row_id_map", snapshots["pre_row_id_map"], expected.pre_row_id_map)
        _assert_tensor_close(
            parallel_info,
            "dispatch_hidden",
            snapshots["dispatch_hidden"],
            expected.dispatch_hidden,
            first_col=True,
        )
        _assert_list_equal(parallel_info, "input_splits", snapshots["input_splits"], expected.input_splits)
        _assert_list_equal(parallel_info, "output_splits", snapshots["output_splits"], expected.output_splits)
        _assert_tensor_close(
            parallel_info,
            "tokens_per_expert_group",
            snapshots["tokens_per_expert_group"],
            expected.tokens_per_expert_group,
        )
        _assert_tensor_close(parallel_info, "post_hidden", snapshots["post_hidden"], expected.post_hidden, first_col=True)
        _assert_tensor_close(parallel_info, "post_row_ids_map", snapshots["post_row_ids_map"], expected.post_row_ids_map)
        _assert_tensor_close(parallel_info, "tokens_per_expert", snapshots["tokens_per_expert"], expected.tokens_per_expert)
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
        raise AssertionError("其他 rank 的 xtuner_ep.md 校验失败。")

    if parallel_info.global_rank == 0:
        print("xtuner_ep.md EP=2 DP=4 all2all 示例校验通过。")


def _assert_tensor_close(
    parallel_info: ParallelInfo,
    name: str,
    actual: torch.Tensor,
    expected: tuple[float, ...] | tuple[int, ...],
    *,
    atol: float = 0.0,
    first_col: bool = False,
) -> None:
    # 文档只跟踪 activation 行来源，不展开 D_h；脚本用第一列承载 token 标识。
    actual_1d = actual.detach()
    if first_col and actual_1d.dim() > 1:
        actual_1d = actual_1d[:, 0]
    actual_1d = actual_1d.reshape(-1).to(torch.float32)
    expected_tensor = torch.tensor(expected, dtype=torch.float32, device=actual.device)
    try:
        torch.testing.assert_close(actual_1d, expected_tensor, rtol=0.0, atol=atol)
    except AssertionError as exc:
        raise AssertionError(
            f"global_rank={parallel_info.global_rank}, dp_rank={parallel_info.dp_rank}, "
            f"ep_rank={parallel_info.ep_rank} 的 {name} 不符合 xtuner_ep.md 示例："
            f"actual={actual_1d.cpu().tolist()}, expected={expected_tensor.cpu().tolist()}"
        ) from exc


def _assert_list_equal(parallel_info: ParallelInfo, name: str, actual: list[int], expected: tuple[int, ...]) -> None:
    if actual != list(expected):
        raise AssertionError(
            f"global_rank={parallel_info.global_rank}, dp_rank={parallel_info.dp_rank}, "
            f"ep_rank={parallel_info.ep_rank} 的 {name} 不符合 xtuner_ep.md 示例："
            f"actual={actual}, expected={expected}"
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
                f"[global_rank={parallel_info.global_rank} dp_rank={parallel_info.dp_rank} "
                f"ep_rank={parallel_info.ep_rank}] {name}: {tensor.reshape(-1).cpu().tolist()}",
                flush=True,
            )
        else:
            print(
                f"[global_rank={parallel_info.global_rank} dp_rank={parallel_info.dp_rank} "
                f"ep_rank={parallel_info.ep_rank}] {name}: {value}",
                flush=True,
            )


if __name__ == "__main__":
    main()
