"""Local MTP buffers must preserve packed rolling values, gradients and lifetimes."""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.mtp import MTPBlock, MTPConfig
from xtuner.v1.module.mtp.utils import local_packed_future_embeddings, roll_packed_tensor


BOUNDARIES = torch.tensor([0, 2, 2, 9, 10, 17, 20], dtype=torch.int32)


@pytest.mark.parametrize("depth", range(1, 8))
@pytest.mark.parametrize("shard_start", [0, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_local_future_matches_full_roll_at_pack_and_sp_boundaries(depth, shard_start, dtype):
    source = torch.arange(60, dtype=dtype).reshape(1, 20, 3).requires_grad_()
    reference_source = source.detach().clone().requires_grad_()
    local = local_packed_future_embeddings(source, BOUNDARIES, depth=depth, shard_start=shard_start, shard_size=4)
    reference = roll_packed_tensor(reference_source, BOUNDARIES, shifts=-depth, dim=1)
    torch.testing.assert_close(local, reference[:, shard_start : shard_start + 4], rtol=0, atol=0)
    assert local.is_contiguous() and local.storage_offset() == 0
    assert local.untyped_storage().nbytes() == local.numel() * local.element_size()
    assert local.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    local.float().square().sum().backward()
    reference[:, shard_start : shard_start + 4].float().square().sum().backward()
    torch.testing.assert_close(source.grad, reference_source.grad, rtol=0, atol=0)


class _Mesh:
    def __init__(self, rank):
        self.rank = rank

    def size(self):
        return 5

    def get_local_rank(self):
        return self.rank

    def get_group(self):
        return None


def _context(embeddings, rank=None):
    start = 0 if rank is None else rank * 4
    size = 20 if rank is None else 4
    ids = torch.arange(20).unsqueeze(0)
    return SequenceContext(
        input_ids=ids[:, start : start + size],
        inputs_embeds=embeddings,
        raw_input_ids=ids,
        cu_seq_lens_q=BOUNDARIES,
        cu_seq_lens_k=BOUNDARIES,
        max_length_q=7,
        max_length_k=7,
        num_padding=3 if rank in (None, 4) else 0,
        position_ids=ids[:, start : start + size],
        sequence_parallel_mesh=None if rank is None else _Mesh(rank),
        shard_start=start,
        shard_size=size,
    )


class _CheckpointLayer(nn.Module):
    def __init__(self, recompute):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.recompute = recompute
        self.futures = []
        self.contexts = []

    def _compute(self, hidden, future, ctx):
        return (self.weight * hidden + future).tanh() * ctx.mask.unsqueeze(-1)

    def forward(self, hidden_states, *, future_embeddings, position_embeddings, seq_ctx):
        multi = isinstance(hidden_states, list)
        states = hidden_states if multi else [hidden_states]
        futures = future_embeddings if multi else [future_embeddings]
        contexts = seq_ctx if multi else [seq_ctx]
        self.futures.append(futures)
        self.contexts.append(contexts)
        outputs = [
            checkpoint(self._compute, h, e, ctx, use_reentrant=False) if self.recompute else self._compute(h, e, ctx)
            for h, e, ctx in zip(states, futures, contexts)
        ]
        result = dict(hidden_states=outputs if multi else outputs[0])
        result.update(
            {
                key: [None] * len(states) if multi else None
                for key in ("router_logits", "router_weights", "router_topk_ids")
            }
        )
        return result


def _run(optimized, recompute, micro_batches, rank=None, detach=False):
    layer = _CheckpointLayer(recompute)
    block = MTPBlock(
        mtp_config=MTPConfig(
            num_layers=7, share_weights=True, use_local_future_embeddings=optimized, detach_mtp_inputs=detach
        ),
        mtp_layers=[layer],
    )
    # Values stay small so every prediction depth contributes measurable gradients.
    full = torch.arange(60, dtype=torch.float64).reshape(1, 20, 3) / 100
    source = (
        full.clone().requires_grad_() if rank is None else full[:, rank * 4 : rank * 4 + 4].clone().requires_grad_()
    )
    contexts = [_context(source, rank) for _ in range(micro_batches)]
    states = [torch.full_like(source, 0.2 + mb / 10, requires_grad=True) for mb in range(micro_batches)]
    positions = [(torch.empty(0), torch.empty(0)) for _ in contexts]
    with patch(
        "xtuner.v1.data_proto.sequence_context.gather_for_sequence_parallel",
        return_value=full.detach().clone(),
    ) as gather:
        outputs = block(
            states[0] if micro_batches == 1 else states,
            embed_tokens_fn=lambda _: pytest.fail("Supplied embeddings must not be re-embedded"),
            position_embeddings=positions[0] if micro_batches == 1 else positions,
            seq_ctx=contexts[0] if micro_batches == 1 else contexts,
        )
    groups = [outputs] if micro_batches == 1 else outputs
    sum(out["hidden_states"].square().mean() for group in groups for out in group).backward()
    return layer, source, states, groups, gather.call_count, contexts


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("micro_batches", [1, 2])
@pytest.mark.parametrize("rank", [None, 0, 4])
def test_seven_depths_preserve_outputs_and_backward_with_checkpoint(recompute, micro_batches, rank):
    old_layer, old_source, old_states, old_outputs, old_gathers, _ = _run(False, recompute, micro_batches, rank)
    layer, source, states, outputs, gathers, contexts = _run(True, recompute, micro_batches, rank)
    assert gathers == old_gathers == (0 if rank is None else micro_batches)
    torch.testing.assert_close(layer.weight.grad, old_layer.weight.grad)
    for state, old_state, group, old_group in zip(states, old_states, outputs, old_outputs):
        torch.testing.assert_close(state.grad, old_state.grad)
        for out, old_out in zip(group, old_group):
            torch.testing.assert_close(out["hidden_states"], old_out["hidden_states"])
    if rank is None:
        torch.testing.assert_close(source.grad, old_source.grad)
    else:
        # Existing SP all_gather is not autograd-aware. A1 preserves that behavior.
        assert source.grad is old_source.grad is None
    for depth in range(7):
        for mb in range(micro_batches):
            assert layer.contexts[depth][mb] is contexts[mb]
            future = layer.futures[depth][mb]
            torch.testing.assert_close(future, old_layer.futures[depth][mb], rtol=0, atol=0)
            assert future.is_contiguous() and future.storage_offset() == 0
            assert future.untyped_storage().nbytes() == future.numel() * future.element_size()
    # Hold all seven inputs across backward; each is an independent local allocation.
    for mb in range(micro_batches):
        pointers = {futures[mb].untyped_storage().data_ptr() for futures in layer.futures}
        assert len(pointers) == 7


def test_local_cache_preserves_single_microbatch_input_detach():
    old_layer, old_source, old_states, _, _, _ = _run(False, True, 1, detach=True)
    layer, source, states, _, _, _ = _run(True, True, 1, detach=True)
    assert source.grad is old_source.grad is states[0].grad is old_states[0].grad is None
    torch.testing.assert_close(layer.weight.grad, old_layer.weight.grad)


def test_token_id_only_inputs_keep_legacy_rolling_when_optimization_is_enabled():
    ctx = SequenceContext.from_input_ids((torch.arange(20).unsqueeze(0),), device="cpu")
    layer = _CheckpointLayer(False)
    block = MTPBlock(
        mtp_config=MTPConfig(num_layers=7, share_weights=True, use_local_future_embeddings=True),
        mtp_layers=[layer],
    )
    block(
        torch.zeros(1, 20, 3),
        embed_tokens_fn=lambda ids: ids.double().unsqueeze(-1).expand(-1, -1, 3),
        position_embeddings=(torch.empty(0), torch.empty(0)),
        seq_ctx=ctx,
    )
    for depth, contexts in enumerate(layer.contexts, start=1):
        expected = torch.arange(20).roll(-depth)
        expected[-depth:] = 0
        torch.testing.assert_close(contexts[0].input_ids, expected.unsqueeze(0))


def _distributed_cache_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("sp",))
        boundaries = torch.tensor([0, 2, 2, 7, 13, 17, 20], dtype=torch.int32)
        full = torch.arange(60, dtype=torch.float64).reshape(1, 20, 3) / 100
        reference = None
        for optimized in (False, True):
            layer = _CheckpointLayer(True)
            block = MTPBlock(
                mtp_config=MTPConfig(num_layers=7, share_weights=True, use_local_future_embeddings=optimized),
                mtp_layers=[layer],
            )
            source = full[:, rank * 10 : rank * 10 + 10].clone().requires_grad_()
            state = torch.full_like(source, 0.2, requires_grad=True)
            ctx = SequenceContext(
                input_ids=None,
                inputs_embeds=source,
                raw_input_ids=torch.arange(20).unsqueeze(0),
                cu_seq_lens_q=boundaries,
                cu_seq_lens_k=boundaries,
                max_length_q=6,
                max_length_k=6,
                sequence_parallel_mesh=mesh,
                position_ids=torch.arange(rank * 10, rank * 10 + 10).unsqueeze(0),
                shard_start=rank * 10,
                shard_size=10,
            )

            def cpu_gather(tensor, dim, sp_group):
                # The production helper requires CUDA. Use its same plain all_gather
                # semantics over two real CPU ranks, without adding an autograd edge.
                shards = [torch.empty_like(tensor) for _ in range(2)]
                dist.all_gather(shards, tensor.detach().contiguous(), group=sp_group)
                return torch.cat(shards, dim=dim).contiguous()

            with patch(
                "xtuner.v1.data_proto.sequence_context.gather_for_sequence_parallel", side_effect=cpu_gather
            ) as gather:
                outputs = block(
                    state,
                    embed_tokens_fn=lambda _: None,
                    position_embeddings=(torch.empty(0), torch.empty(0)),
                    seq_ctx=ctx,
                )
                sum(out["hidden_states"].square().mean() for out in outputs).backward()
                assert gather.call_count == 1
            assert source.grad is None and state.grad is not None
            for depth, futures in enumerate(layer.futures, start=1):
                expected = roll_packed_tensor(full, boundaries, shifts=-depth, dim=1)
                torch.testing.assert_close(futures[0], expected[:, rank * 10 : rank * 10 + 10], rtol=0, atol=0)
            if reference is None:
                reference = (layer.weight.grad.clone(), state.grad.clone(), [out["hidden_states"] for out in outputs])
            else:
                torch.testing.assert_close(layer.weight.grad, reference[0])
                torch.testing.assert_close(state.grad, reference[1])
                for out, old_out in zip(outputs, reference[2]):
                    torch.testing.assert_close(out["hidden_states"], old_out)
                assert all(contexts[0] is ctx for contexts in layer.contexts)
                assert all(
                    futures[0].untyped_storage().nbytes() == source.numel() * source.element_size()
                    for futures in layer.futures
                )
    finally:
        dist.destroy_process_group()


def test_distributed_local_future_cache_matches_full_roll_and_checkpoint_backward(tmp_path):
    mp.spawn(_distributed_cache_worker, args=(str(tmp_path / "local-future-cache"),), nprocs=2, join=True)
