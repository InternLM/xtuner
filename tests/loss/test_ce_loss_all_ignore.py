import torch
import torch.nn as nn

from xtuner.v1.loss.ce_loss import CELossConfig, CELossKwargs, LMHeadLossContext


class _LigerReference(nn.Module):
    """Small differentiable stand-in for Liger's reduction='sum' output."""

    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def forward(
        self,
        head_weight: torch.Tensor,
        hidden_states: torch.Tensor,
        shifted_labels: torch.Tensor,
    ) -> torch.Tensor:
        self.called = True
        valid = (shifted_labels.flatten() != -100).nonzero().flatten()
        return (
            hidden_states.reshape(-1, hidden_states.shape[-1])[valid].sum() + head_weight.sum() * 0
        ).float()


def _build_context() -> LMHeadLossContext:
    context = LMHeadLossContext.__new__(LMHeadLossContext)
    nn.Module.__init__(context)
    context.loss_cfg = CELossConfig(mode="liger", loss_reduction="token")
    context.liger_loss_fct = _LigerReference()
    return context


def test_liger_all_ignore_calibrates_to_differentiable_zero() -> None:
    context = _build_context()

    hidden_states = torch.randn(1, 4, 3, requires_grad=True)
    head_weight = torch.randn(5, 3, requires_grad=True)
    loss_kwargs = CELossKwargs(
        shifted_labels=torch.full((1, 4), context.loss_cfg.ignore_idx),
        loss_weight=torch.zeros(1, 4),
    )

    loss, (logits, extra_info) = context.chunk_mode(hidden_states, head_weight, None, loss_kwargs)
    loss.backward()

    assert logits is None
    assert extra_info == {}
    assert context.liger_loss_fct.called
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(hidden_states.grad, torch.zeros_like(hidden_states))
    torch.testing.assert_close(head_weight.grad, torch.zeros_like(head_weight))
