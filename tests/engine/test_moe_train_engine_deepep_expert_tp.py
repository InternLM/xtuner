from __future__ import annotations

import os
import unittest
from typing import Literal

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
    _get_param_grad,
    _make_engine_input,
    _run_one_step_with_norm,
)

BF16_RTOL, BF16_ATOL = default_tolerances(torch.bfloat16)


def _build_engine(
    *,
    dispatcher: Literal["all2all", "deepep"],
    ep_size: int,
    expert_tp_size: int,
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
    )


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
        torch.testing.assert_close(
            gate_grad_deepep,
            gate_grad_all2all,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )
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

    @property
    def world_size(self) -> int:
        return 4

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
