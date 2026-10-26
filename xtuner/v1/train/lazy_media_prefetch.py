"""Per-SP-leader prefetch, bounded to current plus one future batch.

The supplied decoder identity is computed on the training thread from its SP mesh.
No distributed collective runs in this executor. A running media read cannot be
cancelled: close waits for it; launcher termination handles a hung external read.
"""

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterator

from xtuner.v1.datasets.lazy_media import materialize_image_plan


def lazy_media_prefetch_enabled() -> bool:
    return os.environ.get("XTUNER_LAZY_MEDIA_PREFETCH", "0").lower() in ("1", "true", "yes", "on")


def _iter_media_plans(batch: Any):
    if isinstance(batch, dict) and "seq_ctx" in batch:
        plan = getattr(batch["seq_ctx"], "media_plan", None)
        if plan is not None:
            yield plan
    elif isinstance(batch, (list, tuple)):
        for item in batch:
            yield from _iter_media_plans(item)


def kickoff_lazy_media(batch: Any, *, decode_here: bool) -> None:
    if not decode_here:
        return
    # Called inside the single producer thread; completed futures preserve the
    # consumption API and defer decode errors to the main-thread group handshake.
    for plan in _iter_media_plans(batch):
        future = Future()
        plan["_future"] = future
        try:
            future.set_result(materialize_image_plan(plan))
        except Exception as exc:
            future.set_exception(exc)


def take_media_future(plan: dict) -> Future | None:
    return plan.pop("_future", None)


def lookahead_iter(base: Iterator[Any], *, decode_here: bool, count: int, allow_short: bool = False) -> Iterator[Any]:
    if count <= 0:
        return
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lazy-media-lookahead")
    pending = None
    exhausted = object()

    def fetch():
        try:
            batch = next(base)
        except StopIteration:
            if allow_short:
                return exhausted
            raise
        kickoff_lazy_media(batch, decode_here=decode_here)
        return batch

    try:
        pending = pool.submit(fetch)
        for index in range(count):
            batch = pending.result()
            if batch is exhausted:
                break
            pending = pool.submit(fetch) if index + 1 < count else None
            yield batch
            del batch
    finally:
        if pending is not None:
            pending.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        close = getattr(base, "close", None)
        if close is not None:
            close()


def shared_materialize(plan, group, device, patch_dim):
    """Main-thread SP protocol; CPU Gloo and CUDA NCCL use the same code."""
    import torch
    import torch.distributed as dist

    media_dtype = plan.get("media_dtype", torch.float32)
    images = plan["images"]
    rows = sum(entry["grid"][0] * entry["grid"][1] * entry["grid"][2] for entry in images)
    src = dist.get_process_group_ranks(group)[0]
    error = None
    # Allocate after checking the leader result, so decode failures don't force
    # receivers to allocate full media tensors just to learn about the error.
    status = torch.zeros(1, dtype=torch.int32, device=device)
    local_pixel = local_grid = None
    if dist.get_rank(group) == 0:
        try:
            future = take_media_future(plan)
            local_pixel, local_grid = future.result() if future is not None else materialize_image_plan(plan)
            if local_pixel.dtype != media_dtype or local_grid.dtype != torch.int64:
                raise RuntimeError("Unexpected lazy media dtype")
            if tuple(local_pixel.shape) != (rows, patch_dim) or tuple(local_grid.shape) != (len(images), 3):
                raise RuntimeError("Lazy media shape differs from the SP plan")
        except Exception as exc:
            error = exc
            status.fill_(1)
    dist.broadcast(status, src=src, group=group)
    if status.item():
        raise RuntimeError(f"SP media decoder rank {src} failed") from error
    # Allocation/copy errors are synchronized before any rank enters pixel broadcast.
    try:
        pixels = torch.empty((rows, patch_dim), dtype=media_dtype, device=device)
        grid = torch.empty((len(images), 3), dtype=torch.int64, device=device)
        if dist.get_rank(group) == 0:
            pixels.copy_(local_pixel)
            grid.copy_(local_grid)
    except Exception as exc:
        error = exc
        status.fill_(1)
    dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
    if status.item():
        raise RuntimeError("SP media allocation/copy failed") from error
    dist.broadcast(pixels, src=src, group=group)
    dist.broadcast(grid, src=src, group=group)
    return pixels, grid
