from types import SimpleNamespace

import torch
import torch.nn.functional as F

from xtuner.v1.loss.mtp_loss import MTPE2ETVLossConfig, MTPE2ETVLossContext
from xtuner.v1.utils.device import get_device


DEVICE = torch.device(get_device())


def _build_context(*, detach_head: bool = False) -> MTPE2ETVLossContext:
    # Two packed samples with lengths 5 and 3. For gamma=2, only positions
    # [0, 1, 2] and [5] have a complete two-token speculative horizon.
    shifted_labels = torch.arange(8, device=DEVICE).unsqueeze(0)
    seq_ctx = SimpleNamespace(cu_seq_lens_k=torch.tensor([0, 5, 8], dtype=torch.int32, device=DEVICE))
    cfg = MTPE2ETVLossConfig(
        mode="chunk",
        chunk_size=2,
        loss_reduction="token",
        num_steps=2,
        detach_mtp_lm_head_weight=detach_head,
    )
    ctx = cfg.build({"shifted_labels": shifted_labels, "seq_ctx": seq_ctx})
    assert ctx is not None
    return MTPE2ETVLossContext.build_batches(
        [ctx],
        cu_seq_lens_list=[seq_ctx.cu_seq_lens_k],
    )[0]


def test_e2e_tv_matches_expected_acceptance_length_and_detaches_teacher():
    torch.manual_seed(0)
    target = torch.randn(1, 8, 4, device=DEVICE, requires_grad=True)
    drafts = [torch.randn(1, 8, 4, device=DEVICE, requires_grad=True) for _ in range(2)]
    head_weight = torch.randn(7, 4, device=DEVICE, requires_grad=True)
    ctx = _build_context()

    actual, _ = ctx.forward((target, drafts), head_weight)

    valid_positions = torch.tensor([0, 1, 2, 5], device=DEVICE)
    target_positions = (
        torch.tensor([1, 2, 3, 6], device=DEVICE),
        torch.tensor([2, 3, 4, 7], device=DEVICE),
    )
    overlaps = []
    for draft, positions in zip(drafts, target_positions):
        p = F.softmax(F.linear(target[0, positions].detach(), head_weight.detach()).float(), dim=-1)
        q = F.softmax(F.linear(draft[0, valid_positions], head_weight).float(), dim=-1)
        overlaps.append(torch.minimum(p, q).sum(dim=-1))
    alphas = torch.stack(overlaps, dim=-1)
    expected = (1.0 - torch.cumprod(alphas, dim=-1).mean(dim=-1)).mean()

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert target.grad is None
    assert all(draft.grad is not None and torch.count_nonzero(draft.grad) > 0 for draft in drafts)
    assert head_weight.grad is not None and torch.count_nonzero(head_weight.grad) > 0


def test_e2e_tv_can_detach_shared_lm_head():
    torch.manual_seed(1)
    target = torch.randn(1, 8, 4, device=DEVICE, requires_grad=True)
    drafts = [torch.randn(1, 8, 4, device=DEVICE, requires_grad=True) for _ in range(2)]
    head_weight = torch.randn(7, 4, device=DEVICE, requires_grad=True)
    ctx = _build_context(detach_head=True)

    loss, _ = ctx.forward((target, drafts), head_weight)
    loss.backward()

    assert target.grad is None
    assert head_weight.grad is None
    assert all(draft.grad is not None and torch.count_nonzero(draft.grad) > 0 for draft in drafts)
