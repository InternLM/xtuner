import torch

from xtuner.v1.rl.loss import ValueLossConfig, ValueLossContext, value_loss


class TestValueLoss:
    def test_mse_value_loss_uses_masked_weights(self) -> None:
        loss = value_loss(
            values=torch.tensor([[2.0, 99.0]]),
            returns=torch.tensor([[0.0, 0.0]]),
            loss_weight=torch.tensor([[0.5, 0.0]]),
            loss_type="mse",
        )

        torch.testing.assert_close(loss, torch.tensor(1.0))

    def test_clipped_value_loss_anchors_to_old_values(self) -> None:
        loss = value_loss(
            values=torch.tensor([[0.0]]),
            old_values=torch.tensor([[1.0]]),
            returns=torch.tensor([[0.0]]),
            loss_weight=torch.tensor([[1.0]]),
            loss_type="clipped",
            value_clip=0.2,
        )

        torch.testing.assert_close(loss, torch.tensor(0.32))

    def test_context_calibrates_by_valid_tokens_across_micro_batches(self) -> None:
        config = ValueLossConfig(loss_type="mse")
        first = config.build(
            {
                "returns": torch.tensor([[0.0, 0.0]]),
                "value_mask": torch.tensor([[True, False]]),
            }
        )
        second = config.build(
            {
                "returns": torch.tensor([[1.0, 1.0]]),
                "value_mask": torch.tensor([[True, True]]),
            }
        )
        assert first is not None and second is not None

        contexts = ValueLossContext.build_batches([first, second])

        assert contexts[0].loss_kwargs.global_valid_count is not None
        assert contexts[0].loss_kwargs.global_valid_count.item() == 3.0
        torch.testing.assert_close(contexts[0].loss_kwargs.loss_weight, torch.tensor([[1 / 3, 0.0]]))
        torch.testing.assert_close(contexts[1].loss_kwargs.loss_weight, torch.tensor([[1 / 3, 1 / 3]]))

    def test_context_applies_scalar_head_without_sigmoid(self) -> None:
        config = ValueLossConfig(loss_type="mse")
        context = config.build(
            {
                "returns": torch.tensor([[3.0]]),
                "value_mask": torch.tensor([[True]]),
            }
        )
        assert context is not None
        ValueLossContext.build_batches([context])

        loss, (values, _) = context(
            hidden_states=torch.tensor([[[2.0, -1.0]]]),
            head_weight=torch.tensor([[2.0, 1.0]]),
        )

        torch.testing.assert_close(values, torch.tensor([[3.0]]))
        torch.testing.assert_close(loss, torch.tensor(0.0))
