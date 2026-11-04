# Copyright (c) OpenMMLab. All rights reserved.
"""Muon 优化器布局契约与 checkpoint 兼容的行为测试。

TestMuonMatrixLayoutContract          NS 布局契约（公共入口，纯 CPU）
    test_callback_receives_contract_layout    flatten/batch/expert 布局下回调收到契约形状且输出还原
    test_invalid_expert_layout_is_rejected    非法 expert 布局在公共入口触发断言
TestNewtonSchulzLayoutEquivalence     NS 数值等价（单进程 GPU）
    test_batched_and_flattened_ns_match_per_matrix  batch 与逐矩阵、flatten 与等价 2D 的结果逐位一致
TestMuonConvLayout                    Muon.step 端到端（2 卡 DDP）
    test_flattened_convolutions_match_matrix_updates  分片/复制布局下 3D/4D/5D 与等价二维参数更新一致
    test_existing_policies_keep_their_ratios          step 后 NS 矩阵行列与 lr_ratio 所用行列一致
    test_legacy_checkpoint_restores_logical_matrix_ratio  忽略旧 checkpoint 比例并保留动量及训练状态
    test_updated_group_configuration_matches_fresh_optimizer  配置变化后与同配置新优化器更新一致
    test_dcp_checkpoint_restores_groups_and_refreshes_update_ratio  DCP 磁盘往返恢复后继续更新并写回正确比例
TestMuonDerivedState                  optimizer state 派生字段行为（纯 CPU）
    test_initial_state_is_lazy                          未 step 时 state 延迟创建，param_groups 保留全部参数
    test_legacy_state_is_preserved_without_mutating_checkpoint  旧 state 原样恢复且不改写 checkpoint
    test_invalid_muonsplit_is_rejected_at_initialization        非法 MuonSplit 配置在初始化时拒绝
"""

import copy
import math
import shutil
import tempfile
from typing import Literal
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.optim.muon import Muon, muon_update_newton_schulz, zeropower_via_newtonschulz5


def _matrix_lr_ratio(rows: int, columns: int, mode: Literal["rms_norm", "spectral_norm", "none"]) -> float:
    if mode == "none":
        return 1.0
    if mode == "spectral_norm":
        return math.sqrt(rows / columns)
    return 0.2 * math.sqrt(max(rows, columns))


class TestMuonMatrixLayoutContract:
    """NS 布局契约：muon_update_newton_schulz 按 flatten/num_experts 把参数 reshape 成 2D/3D 交给回调。"""

    @pytest.mark.parametrize(
        ("shape", "flatten", "num_experts", "expected"),
        [
            ((64, 128), False, 1, (64, 128)),  # 普通 2D：单矩阵
            ((32, 3, 2, 2), True, 1, (32, 12)),  # flatten：高维摊平成单矩阵
            ((512, 128), False, 8, (8, 64, 128)),  # expert：拼接 2D 展开成 batch
            ((2, 3, 4, 5), False, 1, (6, 4, 5)),  # 非 flatten：前导维作 batch
        ],
    )
    def test_callback_receives_contract_layout(
        self, shape: tuple[int, ...], flatten: bool, num_experts: int, expected: tuple[int, ...]
    ) -> None:
        # 回调必须收到契约布局，且返回值被 reshape 回原始形状
        received: list[tuple[int, ...]] = []

        def recording_func(x: torch.Tensor, epsilon: float) -> torch.Tensor:
            received.append(tuple(x.shape))
            return x

        X = torch.randn(shape)
        out = muon_update_newton_schulz(
            X,
            newton_schulz_func=recording_func,
            flatten=flatten,
            epsilon=1e-7,
            num_experts=num_experts,
        )
        assert received == [expected]
        assert tuple(out.shape) == shape

    @pytest.mark.parametrize(
        ("shape", "num_experts", "message"),
        [
            ((16, 6, 4), 2, "concatenated 2D expert layout"),  # expert 组必须是拼接 2D
            ((15, 4), 2, "not divisible by num_experts"),  # 行数必须整除 expert 数
        ],
    )
    def test_invalid_expert_layout_is_rejected(self, shape: tuple[int, ...], num_experts: int, message: str) -> None:
        # 非法 expert 布局必须在公共入口响铃，而不是静默丢维度
        with pytest.raises(AssertionError, match=message):
            muon_update_newton_schulz(
                torch.randn(shape),
                newton_schulz_func=lambda x, epsilon: x,
                flatten=False,
                epsilon=1e-7,
                num_experts=num_experts,
            )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestNewtonSchulzLayoutEquivalence:
    """NS 数值等价：batch 布局与逐矩阵、flatten 布局与等价 2D 的结果逐位一致。"""

    def test_batched_and_flattened_ns_match_per_matrix(self) -> None:
        # batch NS 必须等于逐矩阵 NS 的堆叠；flatten conv 必须等于其 2D 等价矩阵
        batched = torch.randn(3, 5, 4, device="cuda", dtype=torch.float32)
        batched_result = muon_update_newton_schulz(
            batched,
            newton_schulz_func=zeropower_via_newtonschulz5,
            flatten=False,
            epsilon=1e-7,
            num_experts=1,
        )
        per_matrix = torch.stack(
            [zeropower_via_newtonschulz5(batched[i], epsilon=1e-7) for i in range(batched.size(0))]
        )
        torch.testing.assert_close(batched_result, per_matrix, atol=0, rtol=0)

        conv = torch.randn(8, 2, 3, 4, device="cuda", dtype=torch.float32)
        matrix = conv.reshape(8, -1)
        conv_result = muon_update_newton_schulz(
            conv,
            newton_schulz_func=zeropower_via_newtonschulz5,
            flatten=True,
            epsilon=1e-7,
            num_experts=1,
        )
        matrix_result = muon_update_newton_schulz(
            matrix,
            newton_schulz_func=zeropower_via_newtonschulz5,
            flatten=True,
            epsilon=1e-7,
            num_experts=1,
        )
        torch.testing.assert_close(conv_result.reshape_as(matrix), matrix_result, atol=0, rtol=0)


@pytest.mark.gpu
class TestMuonConvLayout(DeterministicDDPTestCase):
    """Muon.step 端到端：形状/策略/checkpoint 变化下的更新行为（2 卡 DDP）。"""

    @property
    def world_size(self) -> int:
        return 2

    def test_flattened_convolutions_match_matrix_updates(self) -> None:
        """覆盖两卡分片以及 FP32 ignored 卷积使用的 Replicate/local NS 路径。"""
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("conv.fsdp",))
        for shape, placement in (
            ((64, 1, 4), Shard(0)),
            ((64, 1, 4), Replicate()),
            ((32, 2, 3), Shard(0)),
            ((64, 2, 3, 4), Shard(0)),
            ((32, 3, 2, 2, 2), Shard(0)),
            ((32, 3, 2, 2, 2), Replicate()),
        ):
            matrix_shape = (shape[0], math.prod(shape[1:]))
            initial_weight = torch.randn(matrix_shape, device="cuda")
            gradient = torch.randn_like(initial_weight)
            for mode in ("rms_norm", "spectral_norm", "none"):
                matrix = torch.nn.Parameter(distribute_tensor(initial_weight.clone(), mesh, [placement]))
                convolution = torch.nn.Parameter(
                    distribute_tensor(initial_weight.reshape(shape).clone(), mesh, [placement])
                )
                matrix.grad = distribute_tensor(gradient, mesh, [placement])
                convolution.grad = distribute_tensor(gradient.reshape(shape), mesh, [placement])
                optimizer = Muon(
                    [matrix, convolution],
                    lr=0.01,
                    mu=0.95,
                    weight_decay=0.1,
                    nesterov=True,
                    flatten=True,
                    adjust_lr=mode,
                )
                optimizer.step()
                expected_ratio = _matrix_lr_ratio(*matrix_shape, mode)
                assert all(state["lr_ratio"] == expected_ratio for state in optimizer.state_dict()["state"].values())
                matrix_result = matrix.full_tensor()
                convolution_result = convolution.full_tensor().reshape(matrix_shape)
                assert torch.isfinite(matrix_result).all()
                assert not torch.equal(matrix_result, initial_weight), "优化器必须实际执行更新"
                torch.testing.assert_close(
                    convolution_result, matrix_result, atol=0, rtol=0, msg=f"{shape}, {placement}, {mode}"
                )

    def test_existing_policies_keep_their_ratios(self) -> None:
        """Muon.step must orthogonalize with the same rows/cols used by lr_ratio."""
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("conv.fsdp",))
        # label, param shape, group extras, optional MuonSplit sizes, expected NS (rows, cols)
        policies = (
            ("unflattened", (16, 6, 4), {"flatten": False}, None, (6, 4)),
            ("expert", (16, 4), {"num_experts": 2}, None, (8, 4)),
            ("muonsplit", (8, 4), {}, (4, 4), (4, 4)),
        )
        # 同源性契约与 adjust_lr 模式无关（各模式的公式差异由 flatten conv 用例覆盖），单模式即可
        for label, shape, group_extras, split_sizes, matrix_rc in policies:
            ns_matrix_shapes: list[tuple[int, int]] = []

            def track_newton_schulz(x, epsilon, _shapes=ns_matrix_shapes):
                _shapes.append((int(x.shape[-2]), int(x.shape[-1])))
                return zeropower_via_newtonschulz5(x, epsilon=epsilon)

            param = torch.nn.Parameter(distribute_tensor(torch.randn(shape, device="cuda"), mesh, [Replicate()]))
            param.grad = distribute_tensor(torch.randn(shape, device="cuda"), mesh, [Replicate()])
            optimizer = Muon(
                [{"params": [param], **group_extras}],
                flatten=True,
                adjust_lr="rms_norm",
                use_triton=False,
                weight_decay=0.0,
                newton_schulz_func=track_newton_schulz,
                muon_split_sizes={param: split_sizes} if split_sizes is not None else None,
            )
            optimizer.step()

            assert ns_matrix_shapes, f"{label}: Newton-Schulz was not invoked"
            assert all(rc == matrix_rc for rc in ns_matrix_shapes), (
                f"{label}: NS matrix shapes {ns_matrix_shapes} != {matrix_rc}"
            )
            expected_ratio = 1.0 if split_sizes is not None else _matrix_lr_ratio(*matrix_rc, "rms_norm")
            assert all(state["lr_ratio"] == expected_ratio for state in optimizer.state_dict()["state"].values()), (
                f"{label}: state lr_ratio mismatch"
            )

    def test_legacy_checkpoint_restores_logical_matrix_ratio(self) -> None:
        """忽略旧比例，恢复参数组和动量/step，继续更新与二维参考一致。"""
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("conv.fsdp",))
        for shape in ((64, 1, 4), (32, 2, 3, 4)):
            matrix_shape = (shape[0], math.prod(shape[1:]))
            for mode in ("rms_norm", "spectral_norm", "none"):
                legacy_weight = torch.nn.Parameter(
                    distribute_tensor(torch.randn(shape, device="cuda"), mesh, [Shard(0)])
                )
                legacy_weight.grad = distribute_tensor(torch.randn(shape, device="cuda"), mesh, [Shard(0)])
                legacy_optimizer = Muon(
                    [legacy_weight],
                    lr=0.01,
                    mu=0.95,
                    weight_decay=0.1,
                    nesterov=True,
                    flatten=True,
                    adjust_lr=mode,
                )
                # 模拟旧版本真实训练一步，生成包含旧比例、动量和 step 的 checkpoint。
                legacy_ratio = _matrix_lr_ratio(shape[-2], shape[-1], mode)
                with patch.object(legacy_optimizer, "_param_lr_ratio", return_value=legacy_ratio):
                    legacy_optimizer.step()
                checkpoint = copy.deepcopy(legacy_optimizer.state_dict())
                state_key = checkpoint["param_groups"][0]["params"][0]
                checkpoint["state"][state_key]["lr_ratio"] = legacy_ratio
                checkpoint_weight = legacy_weight.full_tensor().detach().clone()
                checkpoint_momentum = legacy_optimizer.state[legacy_weight]["momentum"].full_tensor().clone()
                assert checkpoint_momentum.count_nonzero() > 0

                resumed_weight = torch.nn.Parameter(distribute_tensor(checkpoint_weight.clone(), mesh, [Shard(0)]))
                # 故意使用不同构造配置；恢复比例必须依据 load 后的配置。
                resumed_optimizer = Muon([resumed_weight], lr=0.5, mu=0.1, flatten=False, adjust_lr="none")
                resumed_optimizer.load_state_dict(copy.deepcopy(checkpoint))
                assert resumed_optimizer.state[resumed_weight]["lr_ratio"] == legacy_ratio
                assert resumed_optimizer.param_groups[0]["flatten"] is True
                assert resumed_optimizer.param_groups[0]["adjust_lr"] == mode
                assert resumed_optimizer.param_groups[0]["step"] == checkpoint["param_groups"][0]["step"]
                torch.testing.assert_close(
                    resumed_optimizer.state[resumed_weight]["momentum"].full_tensor(),
                    checkpoint_momentum,
                    atol=0,
                    rtol=0,
                )

                # 二维参考保留同一 checkpoint 状态，仅将权重/动量 reshape 成逻辑矩阵。
                reference_weight = torch.nn.Parameter(
                    distribute_tensor(checkpoint_weight.reshape(matrix_shape).clone(), mesh, [Shard(0)])
                )
                reference_checkpoint = copy.deepcopy(checkpoint)
                state_key = reference_checkpoint["param_groups"][0]["params"][0]
                reference_checkpoint["state"][state_key]["momentum"] = distribute_tensor(
                    checkpoint_momentum.reshape(matrix_shape), mesh, [Shard(0)]
                )
                reference_optimizer = Muon([reference_weight])
                reference_optimizer.load_state_dict(reference_checkpoint)
                next_gradient = torch.randn(matrix_shape, device="cuda")
                resumed_weight.grad = distribute_tensor(next_gradient.reshape(shape), mesh, [Shard(0)])
                reference_weight.grad = distribute_tensor(next_gradient, mesh, [Shard(0)])
                resumed_optimizer.step()
                reference_optimizer.step()
                torch.testing.assert_close(
                    resumed_weight.full_tensor().reshape(matrix_shape),
                    reference_weight.full_tensor(),
                    atol=0,
                    rtol=0,
                    msg=f"{shape}, {mode}",
                )
                assert checkpoint["state"][state_key]["lr_ratio"] == legacy_ratio
                expected_ratio = _matrix_lr_ratio(*matrix_shape, mode)
                assert resumed_optimizer.state[resumed_weight]["lr_ratio"] == expected_ratio
                assert reference_optimizer.state[reference_weight]["lr_ratio"] == expected_ratio
                assert resumed_optimizer.state_dict()["state"][state_key]["lr_ratio"] == expected_ratio

    def test_updated_group_configuration_matches_fresh_optimizer(self) -> None:
        """比例随当前参数组配置变化，实际更新不读取初始化时的缓存。"""
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("conv.fsdp",))
        initial = torch.randn(32, 3, 2, 2, 2, device="cuda")
        gradient = torch.randn_like(initial)
        changed = torch.nn.Parameter(distribute_tensor(initial.clone(), mesh, [Shard(0)]))
        reference = torch.nn.Parameter(distribute_tensor(initial.clone(), mesh, [Shard(0)]))
        optimizer = Muon([changed], flatten=False, adjust_lr="none", weight_decay=0.0)
        optimizer.state[changed]["lr_ratio"] = -99.0
        optimizer.param_groups[0].update(flatten=True, adjust_lr="spectral_norm")
        reference_optimizer = Muon([reference], flatten=True, adjust_lr="spectral_norm", weight_decay=0.0)
        changed.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
        reference.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(changed.full_tensor(), reference.full_tensor(), rtol=0, atol=0)
        expected_ratio = _matrix_lr_ratio(32, 24, "spectral_norm")
        assert optimizer.state[changed]["lr_ratio"] == expected_ratio
        assert reference_optimizer.state[reference]["lr_ratio"] == expected_ratio

    def test_dcp_checkpoint_restores_groups_and_refreshes_update_ratio(self) -> None:
        """DCP 恢复保留旧比例，继续更新时重算并写回正确比例。"""
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("conv.fsdp",))
        model = torch.nn.Module()
        model.register_parameter(
            "weight", torch.nn.Parameter(distribute_tensor(torch.randn(32, 3, 2, 2, device="cuda"), mesh, [Shard(0)]))
        )
        model.register_parameter(
            "untouched",
            torch.nn.Parameter(
                distribute_tensor(torch.ones(4, 4, device="cuda"), mesh, [Shard(0)]), requires_grad=False
            ),
        )
        optimizer = Muon(model.parameters(), flatten=True, adjust_lr="rms_norm", weight_decay=0.0)
        model.weight.grad = torch.randn_like(model.weight)
        optimizer.step()
        state = get_optimizer_state_dict(model, optimizer)
        expected_ratio = _matrix_lr_ratio(32, 12, "rms_norm")
        assert state["state"]["weight"]["lr_ratio"] == expected_ratio
        state["state"]["weight"]["lr_ratio"] = 1234.0

        checkpoint_dirs = [tempfile.mkdtemp(prefix="muon-dcp-") if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(checkpoint_dirs)
        checkpoint_dir = checkpoint_dirs[0]
        try:
            dcp.save({"optimizer": state}, checkpoint_id=checkpoint_dir)
            loaded = copy.deepcopy(state)
            loaded["state"]["weight"]["momentum"].zero_()
            loaded["state"]["weight"]["lr_ratio"] = 0.0
            dcp.load({"optimizer": loaded}, checkpoint_id=checkpoint_dir)

            resumed_model = torch.nn.Module()
            for name, param in model.named_parameters():
                resumed_model.register_parameter(name, torch.nn.Parameter(param.detach().clone(), param.requires_grad))
            resumed_optimizer = Muon(resumed_model.parameters(), flatten=False, adjust_lr="none", weight_decay=0.5)
            set_optimizer_state_dict(resumed_model, resumed_optimizer, loaded)
            assert resumed_optimizer.param_groups[0]["flatten"] is True
            assert resumed_optimizer.param_groups[0]["adjust_lr"] == "rms_norm"
            assert resumed_optimizer.param_groups[0]["step"] == optimizer.param_groups[0]["step"]
            assert resumed_optimizer.state[resumed_model.untouched] == {}
            assert resumed_optimizer.state[resumed_model.weight]["lr_ratio"] == 1234.0
            torch.testing.assert_close(
                resumed_optimizer.state[resumed_model.weight]["momentum"].full_tensor(),
                optimizer.state[model.weight]["momentum"].full_tensor(),
                rtol=0,
                atol=0,
            )
            gradient = torch.randn_like(model.weight)
            model.weight.grad = gradient
            resumed_model.weight.grad = gradient.clone()
            optimizer.step()
            resumed_optimizer.step()
            torch.testing.assert_close(resumed_model.weight.full_tensor(), model.weight.full_tensor(), rtol=0, atol=0)
            assert loaded["state"]["weight"]["lr_ratio"] == 1234.0
            assert resumed_optimizer.state[resumed_model.weight]["lr_ratio"] == expected_ratio
            assert optimizer.state[model.weight]["lr_ratio"] == expected_ratio
            assert (
                get_optimizer_state_dict(resumed_model, resumed_optimizer)["state"]["weight"]["lr_ratio"]
                == expected_ratio
            )
        finally:
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(checkpoint_dir)


class TestMuonDerivedState:
    def test_initial_state_is_lazy(self) -> None:
        trainable = torch.nn.Parameter(torch.ones(8, 3, 2, 2))
        untouched = torch.nn.Parameter(torch.ones(4, 4), requires_grad=False)
        optimizer = Muon([trainable, untouched], flatten=True)
        state = optimizer.state_dict()
        assert state["state"] == {}
        assert len(state["param_groups"][0]["params"]) == 2

    def test_legacy_state_is_preserved_without_mutating_checkpoint(self) -> None:
        param = torch.nn.Parameter(torch.ones(8, 3, 2, 2))
        optimizer = Muon([param], flatten=True, adjust_lr="rms_norm")
        checkpoint = copy.deepcopy(optimizer.state_dict())
        key = checkpoint["param_groups"][0]["params"][0]
        momentum = torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape)
        checkpoint["state"][key] = {"lr_ratio": -123.0, "momentum": momentum}
        checkpoint["param_groups"][0]["step"] = 7
        restored = Muon([param], flatten=False, adjust_lr="none")
        restored.load_state_dict(checkpoint)
        assert restored.param_groups[0]["flatten"] is True
        assert restored.param_groups[0]["adjust_lr"] == "rms_norm"
        assert restored.param_groups[0]["step"] == 7
        torch.testing.assert_close(restored.state[param]["momentum"], momentum, rtol=0, atol=0)
        assert restored.state[param]["lr_ratio"] == -123.0
        assert restored.state_dict()["state"][key]["lr_ratio"] == -123.0
        assert checkpoint["state"][key]["lr_ratio"] == -123.0

    @pytest.mark.parametrize(
        ("shape", "num_experts", "split_sizes", "message"),
        [
            ((8, 4, 2), 1, (4, 4), "regular 2D"),
            ((8, 4), 2, (4, 4), "regular 2D"),
            ((8, 4), 1, (), "Invalid MuonSplit"),
            ((8, 4), 1, (0, 8), "Invalid MuonSplit"),
            ((8, 4), 1, (-1, 8), "Invalid MuonSplit"),
            ((8, 4), 1, (4, 5), "exceed parameter shape"),
        ],
    )
    def test_invalid_muonsplit_is_rejected_at_initialization(
        self, shape: tuple[int, ...], num_experts: int, split_sizes: tuple[int, ...], message: str
    ) -> None:
        param = torch.nn.Parameter(torch.ones(shape))
        with pytest.raises(ValueError, match=message):
            Muon([{"params": [param], "num_experts": num_experts}], muon_split_sizes={param: split_sizes})
