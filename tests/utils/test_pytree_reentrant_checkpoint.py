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

    checkpoint_output = block(direct, nested=[nested])
    loss = checkpoint_output.sum() + nested.square().sum()
    loss.backward()

    torch.testing.assert_close(direct_source.grad, torch.tensor([30.0]))
    torch.testing.assert_close(nested_source.grad, torch.tensor([102.0]))
