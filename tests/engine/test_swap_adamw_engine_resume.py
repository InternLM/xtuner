"""SwapAdamW(swap_optimizer=True)的 resume 行为测试。

TestSwapAdamWResume
    test_resume_after_dcp_matches_continuous_training: DCP resume 后继续训练与连续训练等价。

TestSwapAdamWLoadStateDict
    test_load_state_dict_applies_moments_on_pinned_host: 加载后数值/step/超参正确且仍在 pinned host;空 state 清零历史。
    test_load_state_dict_rejects_incompatible_state: state key 或 shape 不匹配时报错而不是静默加载。

resume 用例固定以 world_size=2 运行:只有多 rank 才能验证"每个 rank 只保存自己的 shard",
需要至少 2 张卡。
"""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.optim.swap_adamw import SwapAdamW
from xtuner.v1.utils.device import get_device


QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
DEVICE = get_device()
# Deterministic training data: fixed token ids and labels for every step.
INPUT_IDS = [
    151644, 872, 198, 113948, 99553, 11, 53982, 9370, 1773, 11, 100158, 12068, 101765, 11, 104545, 11,
    104591, 4850, 102021, 11, 105074, 1773, 30748, 11, 78398, 104592, 30, 151645, 151644, 77091, 198,
]  # fmt: skip
SEQ_LEN = 32


def _build_engine(lr: float) -> TrainEngine:
    moe_cfg = get_model_config_from_hf(Path(QWEN3_MOE_PATH))
    moe_cfg.dispatcher = None
    moe_cfg.ep_size = 1
    moe_cfg.compile_cfg = False
    moe_cfg.num_hidden_layers = 2
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1)
    return TrainEngine(model_cfg=moe_cfg, optim_cfg=AdamWConfig(lr=lr, swap_optimizer=True), fsdp_cfg=fsdp_cfg)


def _run_one_train_step(engine: TrainEngine) -> None:
    input_ids = torch.tensor([INPUT_IDS[:SEQ_LEN]], dtype=torch.long)
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    loss_cfg = CELossConfig()
    loss_ctx = loss_cfg.build(data={"shifted_labels": input_ids.clone().to(DEVICE)}, sp_mesh=None)
    loss_ctx_list = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])
    engine.train_step([ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx_list[0]})])
    engine.step_optimizer(engine.clip_grad_norm())


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _fresh_state_dict(optimizer: SwapAdamW) -> dict[str, Any]:
    # state_dict() 返回优化器内部 dict 与 pinned buffer 本身;只复制容器,便于测试改写 id/key。
    state_dict = optimizer.state_dict()
    return {
        "state": {param_id: dict(state) for param_id, state in state_dict["state"].items()},
        "param_groups": [dict(group) for group in state_dict["param_groups"]],
    }


class TestSwapAdamWResume(DeterministicDDPTestCase):
    """引擎级 resume 等价性:`TrainEngine.save_dcp` / `TrainEngine.load_dcp`。"""

    def test_resume_after_dcp_matches_continuous_training(self) -> None:
        # 行为:训 1 步 → save_dcp → 新引擎 load_dcp → 再训 1 步,与同一引擎连训 2 步等价(权重/moment/step/lr)
        self.create_pg("cuda")
        torch.accelerator.set_device_index(int(dist.get_rank()))

        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            weights_dir = Path(syncdir[0]) / "weights"

            torch.manual_seed(1234)
            engine = _build_engine(1e-4)
            engine.init_model_weights()
            _run_one_train_step(engine)
            dist.barrier()
            engine.save_dcp(weights_dir=weights_dir)
            dist.barrier()
            _run_one_train_step(engine)

            torch.manual_seed(1234)
            engine2 = _build_engine(1e-3)
            engine2.init_model_weights()
            engine2.load_dcp(weights_dir=weights_dir)
            dist.barrier()
            # 超参恢复:加载后 lr 应回到 checkpoint 里的 1e-4,而不是构建时的 1e-3
            assert all(group["lr"] == 1e-4 for group in engine2.optimizer.param_groups)
            _run_one_train_step(engine2)

            # 优化器状态:数值与连续训练一致、step 续上,且 moment 仍是带全局布局的 pinned DTensor
            # (普通张量会让 DCP 把每个 rank 的 shard 当成整张量保存;不在 pinned host 则 swap 白做)
            opt_state = engine.optimizer.state_dict()["state"]
            opt_state2 = engine2.optimizer.state_dict()["state"]
            assert set(opt_state) == set(opt_state2) != set()
            # state_dict() 跨 group 全局连续编号,用同一套编号把 param id 映射回参数
            params = [p for group in engine2.optimizer.param_groups for p in group["params"]]
            for param_id, state2 in opt_state2.items():
                param = params[param_id]
                assert state2["step"].item() == 2, "step counter must reach 2 after resume + 1 step"
                for key in ("exp_avg", "exp_avg_sq"):
                    moment = state2[key]
                    # moment 必须是带全局 shape/placements 且驻留 pinned host 的 DTensor,否则 DCP 会按错误的 shard 写盘
                    assert isinstance(moment, DTensor)
                    assert moment.shape == param.shape, f"{key} shape diverges from its parameter"
                    assert moment.placements == param.placements, f"{key} placements diverge from its parameter"
                    local2 = moment.to_local()
                    assert local2.device.type == "cpu" and local2.is_pinned()
                    torch.testing.assert_close(local2, _local(opt_state[param_id][key]))

            # 模型权重:与连续训练逐参数一致
            state_dict, state_dict2 = engine.model.state_dict(), engine2.model.state_dict()
            assert set(state_dict) == set(state_dict2)
            for key, weight in state_dict.items():
                torch.testing.assert_close(_local(state_dict2[key]), _local(weight), msg=f"weight mismatch in {key}")

    @property
    def world_size(self) -> int:
        return 2


class TestSwapAdamWLoadStateDict(TestCase):
    """优化器级 `SwapAdamW.load_state_dict`(torch optimizer 协议,不经过 DCP)。"""

    def setUp(self) -> None:
        self.optimizer = SwapAdamW(torch.nn.Linear(8, 4, device=DEVICE).parameters(), lr=1e-4)

    def test_load_state_dict_applies_moments_on_pinned_host(self) -> None:
        # 行为:加载把数值/step/超参写进既有 pinned buffer;空 state(engine 的 load_states=False 路径)清零历史
        optimizer = self.optimizer
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                state[key].fill_(1.0)
            state["step"].fill_(3.0)

        source = _fresh_state_dict(optimizer)
        for state in source["state"].values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value * 2
            state["step"] = torch.tensor(7.0)
        for group in source["param_groups"]:
            group["lr"] = 5e-4
        optimizer.load_state_dict(source)

        loaded = optimizer.state_dict()["state"]
        for param_id, state in loaded.items():
            assert state["step"].item() == 7.0, "step counter must be loaded"
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(state[key], source["state"][param_id][key], rtol=0, atol=0)
                assert state[key].device.type == "cpu" and state[key].is_pinned(), f"{key} must stay on pinned host"
        assert all(group["lr"] == 5e-4 for group in optimizer.param_groups)

        source["state"] = {}
        optimizer.load_state_dict(source)
        for state in optimizer.state_dict()["state"].values():
            assert state["step"].item() == 0.0, "step counter must be reset"
            assert torch.count_nonzero(state["exp_avg"]).item() == 0, "moment must be reset"

    def test_load_state_dict_rejects_incompatible_state(self) -> None:
        # 行为:state key 多/少或 shape 不匹配时抛 ValueError,而不是静默跳过或被 copy_ 广播
        optimizer = self.optimizer
        source = _fresh_state_dict(optimizer)
        param_id = next(iter(source["state"]))
        local_shape = tuple(source["state"][param_id]["exp_avg"].shape)

        source["state"][param_id]["unknown_key"] = torch.zeros(1)
        with pytest.raises(ValueError, match="incompatible optimizer state keys"):
            optimizer.load_state_dict(source)

        source = _fresh_state_dict(optimizer)
        source["state"][param_id].pop("exp_avg_sq")
        with pytest.raises(ValueError, match="incompatible optimizer state keys"):
            optimizer.load_state_dict(source)

        source = _fresh_state_dict(optimizer)
        source["state"][param_id]["exp_avg"] = torch.zeros(*local_shape, 1)
        with pytest.raises(ValueError, match="incompatible optimizer state shape"):
            optimizer.load_state_dict(source)
