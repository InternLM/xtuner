from unittest.mock import MagicMock, patch

import pytest
import torch

from xtuner.v1.ops.comm.deepep_op import dispatch_backward, get_low_latency_buffer


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


class TestLowLatencyBufferQPs:
    """The QP count must satisfy the check in DeepEP's internode normal kernels.

    ``internode.cu`` device-asserts ``num_qps == num_sms // 2 or num_qps >= num_sms``. With 256 experts on EP16 the
    old ``max(num_experts // ep_size, num_sms // 2)`` gave 16 QPs for 20 SMs, so the first cross-node dispatch
    trapped with "CUDA error: unspecified launch failure".
    """

    @pytest.mark.parametrize("ep_size", [8, 16, 32, 64])
    def test_num_qps_satisfies_internode_kernel_check(self, ep_size):
        num_sms, num_experts = 20, 256
        mock_buffer_cls = MagicMock(num_sms=num_sms)
        mock_buffer_cls.get_low_latency_rdma_size_hint.return_value = 0
        for get_config in (mock_buffer_cls.get_dispatch_config, mock_buffer_cls.get_combine_config):
            get_config.return_value.get_nvl_buffer_size_hint.return_value = 0
            get_config.return_value.get_rdma_buffer_size_hint.return_value = 0
        group = MagicMock()
        group.size.return_value = ep_size

        with (
            patch("xtuner.v1.ops.comm.deepep_op.Buffer", mock_buffer_cls),
            patch("xtuner.v1.ops.comm.deepep_op._buffer", None),
        ):
            get_low_latency_buffer(group, hidden=6144, num_experts=num_experts)

        num_qps = mock_buffer_cls.call_args.kwargs["num_qps_per_rank"]
        assert num_qps == num_sms // 2 or num_qps >= num_sms
        assert num_qps >= num_experts // ep_size
