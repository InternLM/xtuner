"""验证 xtuner_ep_dispatcher.md 中 DeepEP 前向示例的中间顺序。

运行方式：
    EP_SIZE=2 DP_SIZE=4 torchrun --nproc-per-node=8 .dev_scripts/validate_xtuner_deepep_md.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


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
    dispatch_hidden: tuple[float, ...]
    dispatch_topk_ids: tuple[int, ...]
    dispatch_topk_weights: tuple[float, ...]
    num_recv_tokens_per_expert_list: tuple[int, ...]
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
        pre_hidden=(10.0, 11.0, 12.0, 13.0),
        dispatch_hidden=(10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0),
        dispatch_topk_ids=(0, -1, -1, 1, 2, -1, -1, 0, 1, -1, -1, 2, -1, 0, -1, 1),
        dispatch_topk_weights=(0.25, 0.0, 0.0, 0.6, 0.7, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.5, 0.0, 0.1, 0.0, 0.65),
        num_recv_tokens_per_expert_list=(3, 3, 2),
        post_hidden=(10.0, 13.0, 22.0, 11.0, 20.0, 23.0, 12.0, 21.0),
        post_row_ids_map=(0, -1, 6, -1, 4, -1, -1, -1, -1, 3, -1, 1, -1, 7, 2, 5),
        tokens_per_expert=(3.0, 3.0, 2.0),
        experts_out=(10.0, 13.0, 22.0, 111.0, 120.0, 123.0, 212.0, 221.0),
        pre_combine_hidden=(2.5, 66.6, 148.4, 2.6, 24.0, 110.5, 2.2, 79.95),
        combine_hidden=(310.0, 191.0, 302.0, 333.0),
        post_combine_hidden=(310.0, 191.0, 302.0, 333.0),
    ),
    1: RankExpected(
        input_hidden=(20.0, 21.0, 22.0, 23.0),
        topk_ids=((1, 3), (4, 2), (5, 0), (3, 1)),
        pre_hidden=(20.0, 21.0, 22.0, 23.0),
        dispatch_hidden=(10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0),
        dispatch_topk_ids=(-1, 1, 0, -1, -1, 2, 1, -1, -1, 0, 1, -1, 2, -1, 0, -1),
        dispatch_topk_weights=(0.0, 0.75, 0.4, 0.0, 0.0, 0.3, 0.8, 0.0, 0.0, 0.8, 0.5, 0.0, 0.9, 0.0, 0.35, 0.0),
        num_recv_tokens_per_expert_list=(3, 3, 2),
        post_hidden=(11.0, 20.0, 23.0, 10.0, 13.0, 21.0, 12.0, 22.0),
        post_row_ids_map=(-1, 0, -1, 4, -1, 5, 7, 2, 3, -1, 6, -1, 1, -1, -1, -1),
        tokens_per_expert=(3.0, 3.0, 2.0),
        experts_out=(311.0, 320.0, 323.0, 410.0, 413.0, 421.0, 512.0, 522.0),
        pre_combine_hidden=(307.5, 124.4, 153.6, 330.4, 256.0, 210.5, 469.8, 113.05),
        combine_hidden=(280.0, 321.0, 472.0, 193.0),
        post_combine_hidden=(280.0, 321.0, 472.0, 193.0),
    ),
}


def main() -> None:
    try:
        parallel_info = _init_distributed()
        snapshots = _run_xtuner_deepep_case(parallel_info)
        _validate(parallel_info, snapshots)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _init_distributed() -> ParallelInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("DeepEPDispatcher 当前依赖 CUDA，请在 GPU 上用 torchrun 运行。")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    ep_size = _get_env_int("EP_SIZE", EP_SIZE)
    dp_size = _get_env_int("DP_SIZE", DEFAULT_DP_SIZE)
    world_size = dist.get_world_size()
    if ep_size != EP_SIZE:
        raise RuntimeError("xtuner_ep_dispatcher.md 的 DeepEP 示例固定为 EP=2。")
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
def _run_xtuner_deepep_case(parallel_info: ParallelInfo) -> dict[str, Any]:
    DeepEPDispatcher = _import_deepep_dispatcher()

    case = CASES[parallel_info.ep_rank]
    hidden_states = torch.zeros(
        (len(case.token_values), HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=parallel_info.device,
    )
    hidden_states[:, 0] = torch.tensor(case.token_values, dtype=torch.bfloat16, device=parallel_info.device)
    topk_ids = torch.tensor(case.topk_ids, dtype=torch.long, device=parallel_info.device)
    topk_weights = torch.tensor(case.topk_weights, dtype=torch.float32, device=parallel_info.device)

    dispatcher = DeepEPDispatcher(
        n_routed_experts=N_ROUTED_EXPERTS,
        training_dtype="bf16",
        process_group=parallel_info.ep_group,
    )

    # 对应文档 1：DeepEP source 侧不做 route-copy 展开，只保留原始 token。
    pre_dispatched = dispatcher.dispatch_preprocess(hidden_states=hidden_states, topk_ids=topk_ids)

    # 对应文档 2：DeepEP dispatch 按 token->rank 发送 hidden、local topk ids 和 topk weights。
    dispatched = dispatcher.dispatch(
        pre_dispatched=pre_dispatched,
        topk_weights=topk_weights,
        decoding=False,
    )

    # 对应文档 3：receiver rank 内按 recv_topk_idx 展开成 local expert grouped 顺序。
    post_dispatched = dispatcher.dispatch_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
    )

    # 用 expert id 改写输出，确保 DeepEP 在 combine 前的 topK 加权折叠也被验证。
    experts_out = _mock_local_experts(
        hidden_states=post_dispatched["hidden_states"],
        tokens_per_expert=post_dispatched["tokens_per_expert"],
        ep_rank=parallel_info.ep_rank,
    )

    # 对应文档 5：expert rank 上先用 recv_topk_weights 做加权折叠，回到 dispatch 后的 source-token 顺序。
    pre_combined = dispatcher.combine_preprocess(
        hidden_states=experts_out,
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        decoding=False,
    )

    # 对应文档 6：DeepEP combine 复用 dispatch handle，把 partial output 送回 source rank 并 SUM。
    combined = dispatcher.combine(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        decoding=False,
    )

    # DeepEP 的 topK 加权已经在 combine_preprocess 完成；这里主要是等待 event 并返回 hidden。
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
        "pre_topk_ids": pre_dispatched["topk_ids"],
        "dispatch_hidden": dispatched["hidden_states"],
        "dispatch_topk_ids": dispatched["topk_ids"],
        "dispatch_topk_weights": dispatched["topk_weights"],
        "num_recv_tokens_per_expert_list": dispatched["num_recv_tokens_per_expert_list"],
        "post_hidden": post_dispatched["hidden_states"],
        "post_row_ids_map": post_dispatched["row_ids_map"],
        "tokens_per_expert": post_dispatched["tokens_per_expert"],
        "experts_out": experts_out,
        "pre_combine_hidden": pre_combined["hidden_states"],
        "combine_hidden": combined["hidden_states"],
        "post_combine_hidden": post_combined["hidden_states"],
    }


def _import_deepep_dispatcher() -> Any:
    try:
        from xtuner.v1.module.dispatcher.deepep import DeepEPDispatcher
    except Exception as exc:
        raise RuntimeError(
            "DeepEPDispatcher 导入失败，请确认当前 conda 环境中的 deep_ep/deep_ep_cpp "
            f"与 CUDA/PyTorch ABI 匹配。原始错误：{exc}"
        ) from exc
    return DeepEPDispatcher


def _mock_local_experts(
    *,
    hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    ep_rank: int,
) -> torch.Tensor:
    local_expert_ids = torch.arange(EXPERTS_PER_RANK, dtype=torch.float32, device=hidden_states.device)
    local_expert_ids = torch.repeat_interleave(local_expert_ids, tokens_per_expert.to(torch.long))
    global_expert_ids = ep_rank * EXPERTS_PER_RANK + local_expert_ids
    experts_out = hidden_states.to(torch.float32)
    experts_out[:, 0] += global_expert_ids * EXPERT_OUTPUT_SCALE
    return experts_out.to(hidden_states.dtype)


def _validate(parallel_info: ParallelInfo, snapshots: dict[str, Any]) -> None:
    expected = EXPECTED[parallel_info.ep_rank]
    error: AssertionError | None = None

    try:
        if os.getenv("XTUNER_EP_DEBUG", "0") == "1":
            _print_snapshots(parallel_info, snapshots)
        _assert_tensor_close(parallel_info, "pre_hidden", snapshots["pre_hidden"], expected.pre_hidden, first_col=True)
        _assert_tensor_close(parallel_info, "pre_topk_ids", snapshots["pre_topk_ids"], _flatten(expected.topk_ids))
        _assert_tensor_close(
            parallel_info,
            "dispatch_hidden",
            snapshots["dispatch_hidden"],
            expected.dispatch_hidden,
            first_col=True,
        )
        _assert_tensor_close(
            parallel_info,
            "dispatch_topk_ids",
            snapshots["dispatch_topk_ids"],
            expected.dispatch_topk_ids,
        )
        _assert_tensor_close(
            parallel_info,
            "dispatch_topk_weights",
            snapshots["dispatch_topk_weights"],
            expected.dispatch_topk_weights,
            atol=1e-4,
        )
        _assert_list_equal(
            parallel_info,
            "num_recv_tokens_per_expert_list",
            snapshots["num_recv_tokens_per_expert_list"],
            expected.num_recv_tokens_per_expert_list,
        )
        _assert_tensor_close(
            parallel_info,
            "post_hidden",
            snapshots["post_hidden"],
            expected.post_hidden,
            first_col=True,
        )
        _assert_tensor_close(
            parallel_info,
            "post_row_ids_map",
            snapshots["post_row_ids_map"],
            expected.post_row_ids_map,
        )
        _assert_tensor_close(
            parallel_info,
            "tokens_per_expert",
            snapshots["tokens_per_expert"],
            expected.tokens_per_expert,
        )
        _assert_tensor_close(
            parallel_info,
            "experts_out",
            snapshots["experts_out"],
            expected.experts_out,
            atol=3.0,
            first_col=True,
        )
        _assert_tensor_close(
            parallel_info,
            "pre_combine_hidden",
            snapshots["pre_combine_hidden"],
            expected.pre_combine_hidden,
            atol=3.0,
            first_col=True,
        )
        _assert_tensor_close(
            parallel_info,
            "combine_hidden",
            snapshots["combine_hidden"],
            expected.combine_hidden,
            atol=3.0,
            first_col=True,
        )
        _assert_tensor_close(
            parallel_info,
            "post_combine_hidden",
            snapshots["post_combine_hidden"],
            expected.post_combine_hidden,
            atol=3.0,
            first_col=True,
        )
    except AssertionError as exc:
        error = exc

    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=parallel_info.device)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)

    if failed.item() != 0:
        if error is not None:
            raise error
        raise AssertionError("其他 rank 的 xtuner_ep_dispatcher.md DeepEP 示例校验失败。")

    if parallel_info.global_rank == 0:
        print("xtuner_ep_dispatcher.md EP=2 DP=4 DeepEP 示例校验通过。")


def _assert_tensor_close(
    parallel_info: ParallelInfo,
    name: str,
    actual: torch.Tensor,
    expected: tuple[float, ...] | tuple[int, ...],
    *,
    atol: float = 0.0,
    first_col: bool = False,
) -> None:
    # 文档只跟踪 activation 行来源，不展开 H 维；脚本用第一列承载 token 标识。
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
            f"ep_rank={parallel_info.ep_rank} 的 {name} 不符合 xtuner_ep_dispatcher.md DeepEP 示例："
            f"actual={actual_1d.cpu().tolist()}, expected={expected_tensor.cpu().tolist()}"
        ) from exc


def _assert_list_equal(parallel_info: ParallelInfo, name: str, actual: list[int], expected: tuple[int, ...]) -> None:
    if actual != list(expected):
        raise AssertionError(
            f"global_rank={parallel_info.global_rank}, dp_rank={parallel_info.dp_rank}, "
            f"ep_rank={parallel_info.ep_rank} 的 {name} 不符合 xtuner_ep_dispatcher.md DeepEP 示例："
            f"actual={actual}, expected={expected}"
        )


def _flatten(values: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    return tuple(item for row in values for item in row)


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
