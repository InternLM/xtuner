import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

from xtuner.v1.model.utils import checkpoint_wrapper, pytree_reentrant_checkpoint


class NestedTensorBlock(nn.Module):
    def forward(self, direct: torch.Tensor, nested: list[torch.Tensor]) -> torch.Tensor:
        return direct * nested[0]


def test_pytree_reentrant_checkpoint_preserves_nested_input_gradients():
    direct_source = torch.tensor([2.0], requires_grad=True)
    nested_source = torch.tensor([5.0], requires_grad=True)
    direct = direct_source * 2
    nested = nested_source * 3
    block = checkpoint_wrapper(
        NestedTensorBlock(),
        checkpoint_impl=CheckpointImpl.REENTRANT,
        checkpoint_fn=pytree_reentrant_checkpoint,
    )

    # nested 模拟 micro2 的 [future_embedding]。同一个 Tensor 有两种用途：
    # 1. checkpoint_output 在 checkpoint 内使用它；
    # 2. nested.square() 在 checkpoint 外使用它。
    # 修复前第一条路径会提前使用原 graph，导致第二条路径再次 backward 时失败。
    checkpoint_output = block(direct, nested=[nested])
    loss = checkpoint_output.sum() + nested.square().sum()
    loss.backward()

    # direct_source=x、nested_source=y 时，checkpoint 内输出为 (2x)*(3y)=6xy。
    # y=5，所以对 x 的梯度为 6y=30。
    torch.testing.assert_close(direct_source.grad, torch.tensor([30.0]))
    # x=2，所以 checkpoint 内对 y 的梯度为 6x=12；外层 (3y)^2 对 y 的
    # 梯度为 18y=90。两条路径的梯度相加得到 12+90=102。
    torch.testing.assert_close(nested_source.grad, torch.tensor([102.0]))
