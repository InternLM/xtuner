"""BF16 MTP CE/TV losses with a completely unsupervised SP shard."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.loss import mtp_loss
from xtuner.v1.loss.mtp_loss import MTPE2ETVLossConfig, MTPE2ETVLossContext, MTPLossConfig, MTPLossContext


@pytest.fixture(autouse=True)
def _cpu_loss_device(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mtp_loss, "DEVICE", torch.device("cpu"))


@pytest.mark.parametrize("mode", ["eager", "chunk"])
@pytest.mark.parametrize("detach_head", [False, True])
def test_mtp_ce_empty_bf16_shard_has_fp32_loss_and_zero_gradients(mode: str, detach_head: bool):
    torch.manual_seed(0)
    hidden = torch.randn(1, 18, 4, dtype=torch.bfloat16, requires_grad=True)
    head = torch.randn(7, 4, dtype=torch.bfloat16, requires_grad=True)
    seq_ctx = SimpleNamespace(cu_seq_lens_k=torch.tensor([0, 10, 18], dtype=torch.int32))
    cfg = MTPLossConfig(mode=mode, chunk_size=2, mtp_depth=7, detach_mtp_lm_head_weight=detach_head)
    ctx = cfg.build({"shifted_labels": torch.arange(18).remainder(7).unsqueeze(0), "seq_ctx": seq_ctx})
    assert ctx is not None
    ctx = MTPLossContext.build_batches([ctx], cu_seq_lens_list=[seq_ctx.cu_seq_lens_k])[0]
    nonempty_loss, _ = ctx.forward(hidden, head)
    ctx.loss_kwargs.shifted_labels.fill_(-100)
    ctx.loss_kwargs.loss_weight.zero_()
    empty_loss, _ = ctx.forward(hidden, head)

    assert empty_loss.dtype == nonempty_loss.dtype == torch.float32
    assert empty_loss.item() == 0
    empty_loss.backward()
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) == 0
    if detach_head:
        assert head.grad is None
    else:
        assert head.grad is not None and torch.count_nonzero(head.grad) == 0


def _mixed_sp_worker(rank: int, rendezvous: str) -> None:
    mtp_loss.DEVICE = torch.device("cpu")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("sp",))
        labels = torch.cat([torch.arange(10).remainder(7), torch.full((10,), -100)]).unsqueeze(0)
        seq_ctx = SimpleNamespace(cu_seq_lens_k=torch.tensor([0, 20], dtype=torch.int32))
        data = {"shifted_labels": labels, "seq_ctx": seq_ctx}
        for objective in ("ce", "e2e_tv"):
            torch.manual_seed(1)
            target = torch.randn(1, 10, 4, dtype=torch.bfloat16, requires_grad=True)
            drafts = [torch.randn_like(target, requires_grad=True) for _ in range(7)]
            head = torch.randn(7, 4, dtype=torch.bfloat16, requires_grad=True)
            if objective == "ce":
                losses = []
                for depth, draft in enumerate(drafts, start=1):
                    cfg = MTPLossConfig(mode="chunk", chunk_size=2, mtp_depth=depth)
                    ctx = cfg.build(data, sp_mesh=mesh)
                    assert ctx is not None
                    ctx = MTPLossContext.build_batches([ctx], [seq_ctx.cu_seq_lens_k], sp_mesh=mesh)[0]
                    if rank == 1:
                        assert torch.count_nonzero(ctx.loss_kwargs.loss_weight) == 0
                    losses.append(ctx.forward(draft, head)[0])
                loss = torch.stack(losses).mean() * 0.1
            else:
                cfg = MTPE2ETVLossConfig(mode="chunk", chunk_size=2, num_steps=7)
                ctx = cfg.build(data, sp_mesh=mesh)
                assert ctx is not None
                ctx = MTPE2ETVLossContext.build_batches([ctx], [seq_ctx.cu_seq_lens_k], sp_mesh=mesh)[0]
                if rank == 1:
                    assert torch.count_nonzero(ctx.loss_kwargs.loss_weight) == 0
                loss = ctx.forward((target, drafts), head)[0] * 0.1

            assert loss.dtype == torch.float32 and torch.isfinite(loss) and loss.item() > 0
            gathered = [torch.empty_like(loss) for _ in range(2)]
            dist.all_gather(gathered, loss.detach())
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
            loss.backward()
            assert target.grad is None
            for tensor in [head, *drafts]:
                assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
                assert (torch.count_nonzero(tensor.grad) == 0).item() == (rank == 1)
    finally:
        dist.destroy_process_group()


def test_mtp_seven_depth_ce_and_tv_agree_across_mixed_empty_sp_ranks(tmp_path: Path):
    mp.spawn(_mixed_sp_worker, args=(str(tmp_path / "mixed-mtp-shards"),), nprocs=2, join=True)
