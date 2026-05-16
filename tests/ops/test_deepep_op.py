from unittest.mock import MagicMock, patch

import torch

from xtuner.v1.ops.comm.deepep_op import dispatch_backward


class TestDispatchBackwardEmptyTokens:
    """Test dispatch_backward handles zero-token dispatch in EP scenario."""

    def test_empty_grad_recv_does_not_crash(self):
        """Construct the scenario where an EP rank receives zero tokens.

        In EP (Expert Parallelism), a rank may receive no tokens in dispatch
        when no token's topk routing points to the experts owned by this rank.
        In backward, both grad_recv_x and grad_recv_topk_weights are empty
        tensors. The original code ``grad_recv_topk_weights[0].shape[-1]``
        crashed with IndexError on empty tensors.
        """
        hidden_size = 128
        topk = 2
        num_experts = 4

        grad_recv_x = torch.empty(0, hidden_size)
        grad_recv_topk_weights = torch.empty(0, topk)

        mock_buffer = MagicMock()
        mock_event = MagicMock()
        mock_buffer.combine.return_value = (
            torch.empty(0, hidden_size),
            torch.empty(0, topk),
            mock_event,
        )

        with patch(
            "xtuner.v1.ops.comm.deepep_op.get_low_latency_buffer",
            return_value=mock_buffer,
        ):
            combined_x, combined_weights, event = dispatch_backward(
                grad_recv_x,
                grad_recv_topk_weights,
                num_experts,
                handle=(),
                group=MagicMock(),
            )

        assert combined_x.shape == (0, hidden_size)
        assert combined_weights.shape == (0, topk)
