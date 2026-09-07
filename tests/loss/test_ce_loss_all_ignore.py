import torch
import torch.nn as nn

from xtuner.v1.loss.ce_loss import CELossConfig, CELossKwargs, LMHeadLossContext


class _UnexpectedLigerCall(nn.Module):
    """Liger must never be invoked for an all-ignore context."""

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
        raise AssertionError("Liger must not run for an all-ignore batch")


def _build_context() -> LMHeadLossContext:
    context = LMHeadLossContext.__new__(LMHeadLossContext)
    nn.Module.__init__(context)
    context.loss_cfg = CELossConfig(mode="liger", loss_reduction="token")
    context.liger_loss_fct = _UnexpectedLigerCall()
    return context


def test_liger_all_ignore_returns_differentiable_zero() -> None:
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
    assert not context.liger_loss_fct.called
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(hidden_states.grad, torch.zeros_like(hidden_states))
    torch.testing.assert_close(head_weight.grad, torch.zeros_like(head_weight))


def test_liger_all_ignore_loss_is_finite_and_float32() -> None:
    context = _build_context()

    hidden_states = torch.randn(1, 4, 3, dtype=torch.bfloat16, requires_grad=True)
    head_weight = torch.randn(5, 3, dtype=torch.bfloat16, requires_grad=True)
    loss_kwargs = CELossKwargs(
        shifted_labels=torch.full((1, 4), context.loss_cfg.ignore_idx),
        loss_weight=torch.zeros(1, 4),
    )

    loss, _ = context.chunk_mode(hidden_states, head_weight, None, loss_kwargs)

    assert torch.isfinite(loss).all()
    assert loss.dtype == torch.float32


def test_liger_mixed_ignore_still_calls_liger() -> None:
    """The bypass must not change behavior for contexts with valid tokens."""
    context = LMHeadLossContext.__new__(LMHeadLossContext)
    nn.Module.__init__(context)
    context.loss_cfg = CELossConfig(mode="liger", loss_reduction="token")
    context.liger_loss_fct = _MixedLigerReference()
    context.liger_loss_fct.called = False

    hidden_states = torch.randn(1, 4, 3, requires_grad=True)
    head_weight = torch.randn(5, 3, requires_grad=True)
    loss_kwargs = CELossKwargs(
        shifted_labels=torch.tensor([[0, 1, -100, 4]]),
        loss_weight=torch.tensor([[0.5, 0.5, 0.0, 0.5]]),
    )

    loss, _ = context.chunk_mode(hidden_states, head_weight, None, loss_kwargs)

    assert context.liger_loss_fct.called
    torch.testing.assert_close(loss, context.liger_loss_fct.expected_loss)


class _MixedLigerReference(nn.Module):
    """Stands in for the real Liger kernel with reduction='sum'."""

    def __init__(self) -> None:
        super().__init__()
        self.called = False
        self.expected_loss: torch.Tensor | None = None

    def forward(
        self,
        head_weight: torch.Tensor,
        hidden_states: torch.Tensor,
        shifted_labels: torch.Tensor,
    ) -> torch.Tensor:
        self.called = True
        valid = (shifted_labels.flatten() != -100).nonzero().flatten()
        liger_sum = hidden_states.reshape(-1, hidden_states.shape[-1])[valid].sum().float()
        self.expected_loss = liger_sum * 0.5
        return liger_sum
