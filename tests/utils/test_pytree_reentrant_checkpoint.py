"""Pytree reentrant checkpoint 的梯度行为测试。

TestPytreeReentrantCheckpoint
    test_nested_inputs_preserve_both_gradient_paths: 嵌套输入在 checkpoint 内外复用时梯度正确汇合。
"""

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

from xtuner.v1.model.utils import checkpoint_wrapper, pytree_reentrant_checkpoint


class NestedTensorBlock(nn.Module):
    def forward(self, direct: torch.Tensor, nested: list[torch.Tensor]) -> torch.Tensor:
        return direct * nested[0]


class TestPytreeReentrantCheckpoint:
    def test_nested_inputs_preserve_both_gradient_paths(self):
        # 验证嵌套 Tensor 在 checkpoint 内外同时使用时不会重复反传旧 graph，且梯度正确相加。
        direct_source = torch.tensor([2.0], requires_grad=True)
        nested_source = torch.tensor([5.0], requires_grad=True)
        direct = direct_source * 2
        nested = nested_source * 3
        block = checkpoint_wrapper(
            NestedTensorBlock(),
            checkpoint_impl=CheckpointImpl.REENTRANT,
            checkpoint_fn=pytree_reentrant_checkpoint,
        )

        loss = block(direct, nested=[nested]).sum() + nested.square().sum()
        loss.backward()

        torch.testing.assert_close(direct_source.grad, torch.tensor([30.0]))
        torch.testing.assert_close(nested_source.grad, torch.tensor([102.0]))
