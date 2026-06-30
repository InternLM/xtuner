from __future__ import annotations

import os
import unittest
from typing import Literal, TypeAlias

# 本测试关注 DeepEP + ExpertTP 的真实 grouped-GEMM 训练路径；
# 与既有 engine ExpertTP 测试一致，用 Cutlass 后端规避本地 Triton TMA 兼容性差异。
os.environ.setdefault("XTUNER_USE_CUTLASS_GROUP_GEMM", "1")

import torch
import torch.distributed as dist
from mmengine.utils import is_installed
from torch.testing._comparison import default_tolerances

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.module.dispatcher.deepep import DeepEPDispatcher
from xtuner.v1.module.dispatcher.torch_all2all import TorchAll2AllDispatcher

from .test_moe_train_engine_tpep import (
    _build_tiny_moe_cfg,
    _copy_matching_engine_weights,
    _get_local_param_grad,
    _get_param_grad,
    _get_tpep_grouped_linear,
    _make_engine_input,
    _run_train_step_items_without_clip,
    _run_one_step_with_norm,
    _run_train_step_without_clip,
    _slice_tpep_weight,
    _sync_engine_weights,
    _zero_non_expert_grads,
)

BF16_RTOL, BF16_ATOL = default_tolerances(torch.bfloat16)
BF16_GRAD_ATOL = BF16_ATOL * 2
TopKExpansion: TypeAlias = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool, int]


def _assert_bf16_training_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # 中文注释：梯度矩阵经过 grouped-GEMM 与 TP/EP 规约，近 0 元素会出现极小累加顺序差异；
    # 这里仍以 torch.testing 的 bf16 默认精度为基准，只给梯度绝对误差留 2 倍余量。
    torch.testing.assert_close(
        actual.to(torch.bfloat16),
        expected.to(torch.bfloat16),
        atol=BF16_GRAD_ATOL,
        rtol=BF16_RTOL,
    )


def _build_engine(
    *,
    dispatcher: Literal["all2all", "deepep"],
    ep_size: int,
    expert_tp_size: int,
    intra_layer_micro_batch: int = 1,
) -> TrainEngine:
    moe_cfg = _build_tiny_moe_cfg(ep_size=ep_size, expert_tp_size=expert_tp_size)
    moe_cfg.dispatcher = dispatcher
    optim_cfg = AdamWConfig()
    fsdp_cfg = FSDPConfig(
        ep_size=ep_size,
        cpu_offload=False,
    )
    return TrainEngine(
        model_cfg=moe_cfg,
        optim_cfg=optim_cfg,
        fsdp_cfg=fsdp_cfg,
        intra_layer_micro_batch=intra_layer_micro_batch,
    )


def _record_deepep_expert_tp_domino_stages(
    engine: TrainEngine,
) -> tuple[dict[str, list[str]], list[TopKExpansion]]:
    stages: dict[str, list[str]] = {
        "async_op_true": [],
    }
    topk_expansions: list[TopKExpansion] = []

    for layer in engine.model.layers.values():
        dispatcher = layer.dispatcher
        assert isinstance(dispatcher, DeepEPDispatcher)
        assert dispatcher._tp_size > 1
        assert dispatcher._process_group is not None
        assert dispatcher._process_group.size() == dispatcher._ep_size * dispatcher._tp_size
        assert dispatcher._virtual_n_experts == dispatcher._n_routed_experts * dispatcher._tp_size

        for stage_name in (
            "dispatch_preprocess",
            "dispatch",
            "dispatch_postprocess",
            "combine_preprocess",
            "combine",
            "combine_postprocess",
        ):
            original_stage = getattr(dispatcher, stage_name)

            def stage_wrapper(
                *args,
                _original_stage=original_stage,
                _stage_name=stage_name,
                _dispatcher=dispatcher,
                **kwargs,
            ):
                if kwargs.get("async_op", False):
                    stages["async_op_true"].append(_stage_name)
                original_topk_ids = kwargs.get("topk_ids")
                result = _original_stage(*args, **kwargs)

                if _stage_name == "dispatch_preprocess":
                    assert isinstance(original_topk_ids, torch.Tensor)
                    # 中文注释：当前 DeePEP ExpertTP 不再有独立 _expert_tp helper；
                    # dispatch_preprocess 会把物理 expert topK 扩展成 virtual expert topK。
                    topk_expansions.append(
                        (
                            tuple(original_topk_ids.shape),
                            tuple(result["topk_ids"].shape),
                            tuple(result["topk_weights"].shape),
                            result["topk_weights"].requires_grad,
                            _dispatcher._tp_size,
                        )
                    )
                return result

            setattr(dispatcher, stage_name, stage_wrapper)

    return stages, topk_expansions


def _assert_domino_deepep_expert_tp_collective_stages(
    stages: dict[str, list[str]],
    topk_expansions: list[TopKExpansion],
) -> None:
    assert set(stages["async_op_true"]) == {
        "dispatch_preprocess",
        "dispatch",
        "dispatch_postprocess",
        "combine_preprocess",
        "combine",
        "combine_postprocess",
    }
    assert topk_expansions
    for expansion in topk_expansions:
        original_topk_shape, expanded_topk_shape, expanded_weight_shape, _, tp_size = expansion
        assert expanded_topk_shape[:-1] == original_topk_shape[:-1]
        assert expanded_topk_shape[-1] == original_topk_shape[-1] * tp_size
        assert expanded_weight_shape == expanded_topk_shape
    # 中文注释：checkpoint wrapper 的第一次 forward 可能在 no_grad 下记录到不可微扩展；
    # 只要重算 forward 存在可微 topK weight 扩展，就覆盖了 DeepEP virtual TP 的 backward 路径。
    assert any(expansion[3] for expansion in topk_expansions)


@unittest.skipIf(
    not torch.cuda.is_available() or not is_installed("deep_ep"),
    "CUDA/NCCL and DeepEP are required for real DeepEP ExpertTP TrainEngine validation.",
)
class TestMoETrainEngineDeepEPExpertTP(DeterministicDDPTestCase):
    def test_deepep_matches_all2all_with_same_expert_tp_topology(self) -> None:
        pg = self.create_pg("cuda")

        ep_size = 2
        expert_tp_size = 2
        engine_all2all = _build_engine(
            dispatcher="all2all",
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
        )
        engine_all2all.init_model_weights()

        engine_deepep = _build_engine(
            dispatcher="deepep",
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
        )
        engine_deepep.init_model_weights()
        _copy_matching_engine_weights(engine_all2all, engine_deepep)
        dist.barrier()

        assert isinstance(engine_all2all.model.layers["0"].dispatcher, TorchAll2AllDispatcher)
        assert isinstance(engine_deepep.model.layers["0"].dispatcher, DeepEPDispatcher)
        assert engine_all2all.model.ep_mesh is not None
        assert engine_deepep.model.ep_mesh is not None
        assert engine_all2all.model.expert_tp_mesh is not None
        assert engine_deepep.model.expert_tp_mesh is not None
        assert engine_all2all.model.ep_mesh.size() == engine_deepep.model.ep_mesh.size() == ep_size
        assert (
            engine_all2all.model.expert_tp_mesh.size()
            == engine_deepep.model.expert_tp_mesh.size()
            == expert_tp_size
        )
        assert type(engine_all2all.optimizer) is type(engine_deepep.optimizer)
        assert len(engine_all2all.optimizer.param_groups) == len(engine_deepep.optimizer.param_groups)
        assert [
            len(group["params"]) for group in engine_all2all.optimizer.param_groups
        ] == [len(group["params"]) for group in engine_deepep.optimizer.param_groups]

        device = torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
        input_ids, labels = _make_engine_input(device=device, seed_offset=dist.get_rank())
        loss_cfg = CELossConfig()

        loss_deepep, _, norm_deepep = _run_one_step_with_norm(engine_deepep, loss_cfg, input_ids, labels)
        loss_all2all, _, norm_all2all = _run_one_step_with_norm(engine_all2all, loss_cfg, input_ids, labels)

        torch.testing.assert_close(
            torch.tensor(loss_deepep),
            torch.tensor(loss_all2all),
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        gate_grad_deepep = _get_param_grad(engine_deepep, "layers.0.gate.weight")
        gate_grad_all2all = _get_param_grad(engine_all2all, "layers.0.gate.weight")
        _assert_bf16_training_close(gate_grad_deepep, gate_grad_all2all)
        torch.testing.assert_close(
            norm_deepep,
            norm_all2all,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    def test_deepep_expert_tp_matches_single_model_baseline(self) -> None:
        pg = self.create_pg("cuda")

        ep_size = 2
        expert_tp_size = 2
        engine_ref = _build_engine(
            dispatcher="all2all",
            ep_size=1,
            expert_tp_size=1,
        )
        engine_ref.init_model_weights()

        engine_deepep = _build_engine(
            dispatcher="deepep",
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
        )
        engine_deepep.init_model_weights()
        _sync_engine_weights(engine_ref, engine_deepep)
        dist.barrier()

        assert isinstance(engine_deepep.model.layers["0"].dispatcher, DeepEPDispatcher)
        assert engine_deepep.model.ep_mesh is not None
        assert engine_deepep.model.expert_tp_mesh is not None
        assert engine_deepep.model.ep_mesh.size() == ep_size
        assert engine_deepep.model.expert_tp_mesh.size() == expert_tp_size

        device = torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
        input_ids, labels = _make_engine_input(device=device, seed_offset=dist.get_rank())
        loss_cfg = CELossConfig()

        loss_deepep, _, norm_deepep = _run_one_step_with_norm(engine_deepep, loss_cfg, input_ids, labels)
        loss_ref, _, norm_ref = _run_one_step_with_norm(engine_ref, loss_cfg, input_ids, labels)

        torch.testing.assert_close(
            torch.tensor(loss_deepep),
            torch.tensor(loss_ref),
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        gate_grad_deepep = _get_param_grad(engine_deepep, "layers.0.gate.weight")
        gate_grad_ref = _get_param_grad(engine_ref, "layers.0.gate.weight")
        _assert_bf16_training_close(gate_grad_deepep, gate_grad_ref)

        for module_suffix, fused_gate_up in (
            ("layers.0.experts.fused_w1w3", True),
            ("layers.0.experts.fused_w2", False),
        ):
            ref_grad = _get_param_grad(engine_ref, f"{module_suffix}.weight")
            deepep_grad = _get_local_param_grad(engine_deepep, f"{module_suffix}.weight")
            deepep_module = _get_tpep_grouped_linear(engine_deepep, module_suffix)
            expected_deepep_grad = _slice_tpep_weight(deepep_module, ref_grad, fused_gate_up=fused_gate_up)
            _assert_bf16_training_close(deepep_grad, expected_deepep_grad)

        torch.testing.assert_close(
            norm_deepep,
            norm_ref,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    def test_deepep_expert_tp_expert_only_grad_norm_matches_single_model_baseline(self) -> None:
        pg = self.create_pg("cuda")

        ep_size = 2
        expert_tp_size = 2
        engine_ref = _build_engine(
            dispatcher="all2all",
            ep_size=1,
            expert_tp_size=1,
        )
        engine_ref.init_model_weights()

        engine_deepep = _build_engine(
            dispatcher="deepep",
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
        )
        engine_deepep.init_model_weights()
        _sync_engine_weights(engine_ref, engine_deepep)
        dist.barrier()

        device = torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
        input_ids, labels = _make_engine_input(device=device, seed_offset=dist.get_rank())
        loss_cfg = CELossConfig()

        _run_train_step_without_clip(engine_deepep, loss_cfg, input_ids, labels)
        _run_train_step_without_clip(engine_ref, loss_cfg, input_ids, labels)
        # 中文注释：expert-only norm 单独验证 EP 和 ExpertTP shard 的 norm-square 汇总语义。
        _zero_non_expert_grads(engine_deepep)
        _zero_non_expert_grads(engine_ref)
        expert_norm_deepep = engine_deepep.clip_grad_norm(do_clip=False).detach().float().cpu()
        expert_norm_ref = engine_ref.clip_grad_norm(do_clip=False).detach().float().cpu()
        torch.testing.assert_close(
            expert_norm_deepep,
            expert_norm_ref,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    def test_deepep_expert_tp_domino_micro_batch_matches_sync_baseline(self) -> None:
        pg = self.create_pg("cuda")

        ep_size = 2
        expert_tp_size = 2
        engine_ref = _build_engine(
            dispatcher="deepep",
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
        )
        engine_ref.init_model_weights()

        engine_domino = _build_engine(
            dispatcher="deepep",
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
            intra_layer_micro_batch=2,
        )
        engine_domino.init_model_weights()
        _copy_matching_engine_weights(engine_ref, engine_domino)
        stages, topk_expansions = _record_deepep_expert_tp_domino_stages(engine_domino)
        dist.barrier()

        device = torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
        batches = [
            _make_engine_input(device=device, seed_offset=dist.get_rank() * 2),
            _make_engine_input(device=device, seed_offset=dist.get_rank() * 2 + 1),
        ]
        loss_cfg = CELossConfig()

        loss_domino = _run_train_step_items_without_clip(engine_domino, loss_cfg, batches)
        norm_domino = engine_domino.clip_grad_norm(do_clip=False).detach().float().cpu()
        gate_grad_domino = _get_param_grad(engine_domino, "layers.0.gate.weight")

        loss_ref = _run_train_step_items_without_clip(engine_ref, loss_cfg, batches)
        norm_ref = engine_ref.clip_grad_norm(do_clip=False).detach().float().cpu()
        gate_grad_ref = _get_param_grad(engine_ref, "layers.0.gate.weight")

        _assert_domino_deepep_expert_tp_collective_stages(stages, topk_expansions)
        torch.testing.assert_close(
            torch.tensor(loss_domino),
            torch.tensor(loss_ref),
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )
        torch.testing.assert_close(
            norm_domino,
            norm_ref,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )
        _assert_bf16_training_close(gate_grad_domino, gate_grad_ref)

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @property
    def world_size(self) -> int:
        return 4

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
