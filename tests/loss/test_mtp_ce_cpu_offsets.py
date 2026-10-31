import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.mtp_loss import MTPLossConfig


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [1, 7])
def test_mtp_ce_build_matches_gpu_boundaries_and_uses_key_layout(depth: int):
    seq_ctx = SequenceContext.from_input_ids(
        tuple(torch.zeros(1, length, dtype=torch.long) for length in (3, 5, 7)), device="cpu"
    ).to("cuda")
    labels = torch.arange(15, device="cuda")[None, :]
    config = MTPLossConfig(mtp_depth=depth)

    with_cpu_offsets = config.build({"seq_ctx": seq_ctx, "shifted_labels": labels})
    gpu_reference = config.build({"seq_ctx": seq_ctx.copy(cu_seq_lens_q_cpu=None), "shifted_labels": labels})
    assert with_cpu_offsets is not None and gpu_reference is not None
    torch.testing.assert_close(
        with_cpu_offsets.loss_kwargs.shifted_labels,
        gpu_reference.loss_kwargs.shifted_labels,
        atol=0,
        rtol=0,
    )

    # MTP must still follow key boundaries when they differ from query boundaries.
    different_k = seq_ctx.copy(cu_seq_lens_k=torch.tensor([0, 4, 15], dtype=torch.int32, device="cuda"))
    with_cpu_offsets = config.build({"seq_ctx": different_k, "shifted_labels": labels})
    gpu_reference = config.build({"seq_ctx": different_k.copy(cu_seq_lens_q_cpu=None), "shifted_labels": labels})
    assert with_cpu_offsets is not None and gpu_reference is not None
    torch.testing.assert_close(
        with_cpu_offsets.loss_kwargs.shifted_labels,
        gpu_reference.loss_kwargs.shifted_labels,
        atol=0,
        rtol=0,
    )
