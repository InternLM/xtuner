"""Real distributed logprob layout and numerical regression.

Run with pytest (four CPU ranks), or torchrun this file with --device cuda.
"""

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from xtuner.v1.loss.rl_loss import LogProbConfig, LogProbContext


class TestLogProbSPNumerical:
    def test_gloo_layout_and_logprobs(self, tmp_path: Path) -> None:
        context = mp.spawn(_worker, args=(4, f"file://{tmp_path / 'rendezvous'}", "cpu"), nprocs=4, join=False)
        try:
            deadline = time.monotonic() + 180
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Distributed logprob regression exceeded 180 seconds")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)


def _oracle(hidden: torch.Tensor, head: torch.Tensor, bias: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = torch.nn.functional.linear(hidden, head, bias).float()
    return logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)


def _check(mesh: DeviceMesh, device: torch.device) -> None:
    size = mesh.size()
    rank = dist.get_rank(mesh.get_group())
    vocab, width, batch = 127, 5, 2
    head = torch.arange(vocab * width, device=device, dtype=torch.float32).reshape(vocab, width).sin() / 7
    bias = torch.arange(vocab, device=device, dtype=torch.float32).cos() / 11
    assert LogProbConfig().build({}, sp_mesh=mesh, device=device) is None
    for length in (2 * size, 2 * size + 1, 1):
        labels = (torch.arange(batch * length, device=device).reshape(batch, length) * 7 + 3) % vocab
        if length > 1:
            labels[:, 1] = -100
        source = labels.clone()
        data = {"shifted_labels": labels}
        shard_size = (length + size - 1) // size
        padded_length = shard_size * size
        padded_labels = torch.full((batch, padded_length), -100, device=device, dtype=labels.dtype)
        padded_labels[:, :length] = labels
        hidden = torch.arange(batch * length * width, device=device, dtype=torch.float32)
        hidden = hidden.reshape(batch, length, width).cos() / 3
        padded_hidden = torch.zeros(batch, padded_length, width, device=device)
        padded_hidden[:, :length] = hidden
        start = rank * shard_size
        stop = start + shard_size
        expected_labels = padded_labels[:, start:stop]
        local_hidden = padded_hidden[:, start:stop].contiguous()
        expected = _oracle(padded_hidden, head, bias, padded_labels)
        contexts = []
        for mode in ("eager", "chunk"):
            ctx = LogProbConfig(mode=mode, chunk_size=2).build(data, sp_mesh=mesh, device=device)
            assert ctx is not None
            torch.testing.assert_close(ctx.loss_kwargs.shifted_labels, expected_labels, rtol=0, atol=0)
            actual, _ = ctx.forward(local_hidden, head, bias)
            torch.testing.assert_close(actual, expected[:, start:stop], rtol=1e-5, atol=1e-5)
            gathered = [torch.empty_like(actual) for _ in range(size)]
            dist.all_gather(gathered, actual, group=mesh.get_group())
            torch.testing.assert_close(torch.cat(gathered, dim=1)[:, :length], expected[:, :length])
            contexts.append(ctx)
        for _ in range(2):
            batched = LogProbContext.build_batches(contexts, sp_mesh=mesh)
            for before, after in zip(contexts, batched):
                assert before is after and after.batch_size == 2
                torch.testing.assert_close(after.loss_kwargs.shifted_labels, expected_labels, rtol=0, atol=0)
        local_ctx = LogProbConfig().build({"shifted_labels": expected_labels}, device=device)
        assert local_ctx is not None
        torch.testing.assert_close(local_ctx.loss_kwargs.shifted_labels, expected_labels, rtol=0, atol=0)
        torch.testing.assert_close(data["shifted_labels"], source, rtol=0, atol=0)
        assert data["shifted_labels"] is labels and set(data) == {"shifted_labels"}


def _worker(rank: int, world_size: int, init_method: str, device_type: str) -> None:
    torch.set_num_threads(1)
    device = torch.device(device_type)
    if device_type == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=150),
    )
    try:
        _check(init_device_mesh(device_type, (world_size,)), device)
        subgroup_mesh = init_device_mesh(device_type, (world_size // 2, 2), mesh_dim_names=("dp", "sp"))
        _check(subgroup_mesh["sp"], device)
        singleton_mesh = init_device_mesh(device_type, (world_size, 1), mesh_dim_names=("dp", "sp"))
        _check(singleton_mesh["sp"], device)
        if rank == 0:
            print(f"PASS: logprob {device_type} {world_size} ranks; labels/eager/chunk/subgroups/padding", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), "env://", args.device)
