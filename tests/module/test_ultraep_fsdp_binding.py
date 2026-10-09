import pytest
import torch

from xtuner.v1.module.ultraep.fsdp_expert_binding import (
    writeback_fsdp_unsharded_expert_gradients,
)


def test_writeback_binds_bf16_staging_without_accumulating_old_grad():
    first = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    second = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    staging_first = torch.full((2, 3), 2, dtype=torch.bfloat16)
    staging_second = torch.full((2, 3), 3, dtype=torch.bfloat16)

    writeback_fsdp_unsharded_expert_gradients(
        (first, second), (staging_first, staging_second)
    )

    assert first.grad.data_ptr() == staging_first.data_ptr()
    assert second.grad.data_ptr() == staging_second.data_ptr()
    assert torch.equal(first.grad, staging_first)
    assert torch.equal(second.grad, staging_second)


def test_writeback_rejects_non_bf16_staging():
    first = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    second = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="staging must be BF16"):
        writeback_fsdp_unsharded_expert_gradients(
            (first, second),
            (torch.zeros(2, 3, dtype=torch.float32), torch.zeros(2, 3, dtype=torch.bfloat16)),
        )
