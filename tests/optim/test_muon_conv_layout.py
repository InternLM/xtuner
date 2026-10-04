# Copyright (c) OpenMMLab. All rights reserved.
"""Muon 卷积参数的逻辑矩阵形状回归测试。

TestMuonConvLayout
    test_flattened_convolutions_match_matrix_updates  分片/复制布局下 3D/4D 与等价二维参数更新一致
    test_existing_policies_keep_their_ratios          expert、非 flatten 和 MuonSplit 保持既有规则
    test_legacy_checkpoint_restores_logical_matrix_ratio  旧 checkpoint 迁移比例并保留动量及训练状态
"""

import copy
import math
from typing import Literal

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.optim.muon import Muon


def _matrix_lr_ratio(rows: int, columns: int, mode: Literal["rms_norm", "spectral_norm", "none"]) -> float:
    if mode == "none":
        return 1.0
    if mode == "spectral_norm":
        return math.sqrt(rows / columns)
    return 0.2 * math.sqrt(max(rows, columns))


@pytest.mark.gpu
class TestMuonConvLayout(DeterministicDDPTestCase):
    """仅改变参数存储形状，不应改变 flatten=True 的普通 Muon 更新。"""

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
                expected_ratio = _matrix_lr_ratio(*matrix_shape, mode)
                assert optimizer.state[matrix]["lr_ratio"] == expected_ratio
                assert optimizer.state[convolution]["lr_ratio"] == expected_ratio, (shape, placement, mode)
                optimizer.step()
                matrix_result = matrix.full_tensor()
                convolution_result = convolution.full_tensor().reshape(matrix_shape)
                assert torch.isfinite(matrix_result).all()
                assert not torch.equal(matrix_result, initial_weight), "优化器必须实际执行更新"
                torch.testing.assert_close(
                    convolution_result, matrix_result, atol=0, rtol=0, msg=f"{shape}, {placement}, {mode}"
                )

    def test_existing_policies_keep_their_ratios(self) -> None:
        """按参数组读取 flatten，保留 expert 形状规则和 MuonSplit 的块内缩放。"""
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("conv.fsdp",))
        for mode in ("rms_norm", "spectral_norm", "none"):
            unflattened = torch.nn.Parameter(distribute_tensor(torch.zeros(16, 6, 4, device="cuda"), mesh, [Shard(0)]))
            expert = torch.nn.Parameter(distribute_tensor(torch.zeros(16, 6, 4, device="cuda"), mesh, [Shard(0)]))
            split = torch.nn.Parameter(distribute_tensor(torch.zeros(8, 4, device="cuda"), mesh, [Shard(0)]))
            optimizer = Muon(
                [
                    {"params": [unflattened], "flatten": False},
                    {"params": [expert], "num_experts": 2},
                    {"params": [split]},
                ],
                flatten=True,
                adjust_lr=mode,
                muon_split_sizes={split: (4, 4)},
            )
            optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            assert optimizer.state[unflattened]["lr_ratio"] == _matrix_lr_ratio(6, 4, mode)
            assert optimizer.state[expert]["lr_ratio"] == _matrix_lr_ratio(3, 4, mode)
            assert optimizer.state[split]["lr_ratio"] == 1.0

    def test_legacy_checkpoint_restores_logical_matrix_ratio(self) -> None:
        """加载旧比例后仅迁移 LR ratio，保留动量和 step，继续更新与二维参考一致。"""
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
                legacy_optimizer.state[legacy_weight]["lr_ratio"] = legacy_ratio
                legacy_optimizer.step()
                checkpoint = copy.deepcopy(legacy_optimizer.state_dict())
                checkpoint_weight = legacy_weight.full_tensor().detach().clone()
                checkpoint_momentum = legacy_optimizer.state[legacy_weight]["momentum"].full_tensor().clone()
                assert checkpoint_momentum.count_nonzero() > 0

                resumed_weight = torch.nn.Parameter(distribute_tensor(checkpoint_weight.clone(), mesh, [Shard(0)]))
                # 故意使用不同构造配置；恢复比例必须依据 load 后的配置。
                resumed_optimizer = Muon([resumed_weight], lr=0.5, mu=0.1, flatten=False, adjust_lr="none")
                resumed_optimizer.load_state_dict(copy.deepcopy(checkpoint))
                expected_ratio = _matrix_lr_ratio(*matrix_shape, mode)
                assert resumed_optimizer.state[resumed_weight]["lr_ratio"] == expected_ratio
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
                reference_checkpoint["state"][state_key]["lr_ratio"] = expected_ratio
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
