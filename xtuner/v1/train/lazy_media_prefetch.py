"""V2c: one-step-ahead lazy media materialization (bounded lookahead prefetch).

The trainer's data loop is serial: the batch for step N+1 is fetched only after step N
finishes, so the lazy media decode would otherwise sit at the head of every forward.
This module moves two things onto a single background thread:

1. fetching the next batch from the dataloader (one slot ahead, bounded);
2. kicking off materialize_image_plan for every media plan in that batch.

The result is attached as a ``concurrent.futures.Future`` ON THE PLAN DICT (plans travel
by reference through SequenceContext split/copy, so the future reaches the consumption
point regardless of context re-wrapping). Consumption stays on the main thread in
Glm53ForConditionalGeneration._prepare_llm_inputs — in particular all NCCL collectives
(V2b broadcast) remain on the main thread, because concurrent collectives on the same
process group from two threads are not safe.

Gating:
- XTUNER_LAZY_MEDIA_PREFETCH=0 (default) keeps the V2a/V2b behavior exactly.
- With XTUNER_LAZY_MEDIA_SHARED=1 (V2b), only global rank 0 kicks off decoding; other
  ranks receive the exact bits via the consumption-time broadcast. Note: this rank-0
  shortcut assumes DP1 (the SP group covers the world, as in the frozen GPU3 contract);
  under DP>1 every data replica needs its own decoding rank, so the shared+prefetch
  combination falls back to all-rank decode there (see _should_decode_here).

Failure semantics: a decode error propagates through future.result() at consumption and
fails the run with the original exception — no fake samples, no silent skips.
"""

import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterator

import torch.distributed as dist

from xtuner.v1.datasets.mllm_tokenize_fn.glm53_vl_tokenize_fn import (
    lazy_media_shared_decode,
    materialize_image_plan,
)

PREFETCH_ENV = "XTUNER_LAZY_MEDIA_PREFETCH"

# One worker serializes plan materializations; each materialization itself uses the V2a
# bounded thread pool, so the total decode thread count stays at that pool's size.
_MATERIALIZER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lazy-media-mat")
_LOOKAHEAD_SLOT = 1


def lazy_media_prefetch_enabled() -> bool:
    """Return whether one-step-ahead media prefetching is enabled for this process."""
    return os.environ.get(PREFETCH_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _should_decode_here() -> bool:
    # V2b mode: a single decoding rank per SP group. With DP1 the SP group is the whole
    # world, so global rank 0 is the right decoder. Otherwise decode locally everywhere
    # (the consumption-time broadcast of V2b is then skipped for want of a shared decoder
    # in every group; the local V2a path consumes the future instead).
    if lazy_media_shared_decode() and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _iter_media_plans(batch: Any):
    if isinstance(batch, dict) and "seq_ctx" in batch:
        plan = getattr(batch["seq_ctx"], "media_plan", None)
        if plan is not None:
            yield plan
    elif isinstance(batch, (list, tuple)):
        for item in batch:
            yield from _iter_media_plans(item)


def kickoff_lazy_media(batch: Any) -> None:
    """Start materializing every media plan in ``batch`` (no-op without plans)."""
    if not lazy_media_prefetch_enabled():
        return
    decode_here = _should_decode_here()
    for plan in _iter_media_plans(batch):
        if "_future" in plan or not decode_here:
            continue
        plan["_future"] = _MATERIALIZER.submit(materialize_image_plan, plan)


def take_media_future(plan: dict) -> Future | None:
    """Detach and return the plan's materialization future, if any."""
    return plan.pop("_future", None)


_DONE = object()


def lookahead_iter(base: Iterator[Any]) -> Iterator[Any]:
    """Iterate ``base`` one batch ahead on a background thread, kicking off decodes."""
    slot: queue.Queue = queue.Queue(maxsize=_LOOKAHEAD_SLOT)

    def worker() -> None:
        try:
            for fetched in base:
                kickoff_lazy_media(fetched)
                slot.put(fetched)
            slot.put(_DONE)
        except BaseException as exc:  # surfaced to the main loop below
            slot.put(exc)

    threading.Thread(target=worker, daemon=True, name="lazy-media-lookahead").start()
    while True:
        item = slot.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item
