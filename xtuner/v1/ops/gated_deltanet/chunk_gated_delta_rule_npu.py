"""GatedDeltaNet chunk kernel backed by fla_npu AscendC custom operators.

Port of the MindSpeed-MM reference implementation
(``fsdp/ops/gdn/flash_chunk_gated_delta_rule.py``, the kernel behind the 2B
FSDP run), minus the activation-offload and training-context machinery that
XTuner does not use.  The three cheap Triton helpers stay on Triton exactly
as in the reference (chunk-local cumsum, WY KKT scores, tril inverse); the
eight heavy forward/backward ops run as AscendC kernels under
``torch.ops.npu.*``.

The fla_npu package owns loading: ``import fla_npu`` prepares the embedded
OPP, promotes the torch_npu libraries into the global symbol scope, and
``torch.ops.load_library()``es its own AscendC extension, so once it returns
every ``torch.ops.npu.*`` schema is registered.

Selected by ``get_chunk_gated_delta_rule_fn`` when XTUNER_NPU_GDN_ASCENDC=1
(the default); any load failure falls back to XTuner's Triton pipeline.
"""

import os
import sys
from typing import Any, Optional

import torch
from mindspeed.ops.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # type: ignore[import-untyped]
from mindspeed.ops.triton.cumsum import chunk_local_cumsum  # type: ignore[import-untyped]
from mindspeed.ops.triton.solve_tril import solve_tril  # type: ignore[import-untyped]


_l2norm_fused = None
_l2norm_checked = False


def _fused_l2norm():
    """Resolve the fla_npu triton l2norm once; None keeps the inline fallback."""
    global _l2norm_fused, _l2norm_checked
    if not _l2norm_checked:
        _l2norm_checked = True
        try:
            from fla_npu.ops.triton import l2norm  # type: ignore[import-untyped]

            _l2norm_fused = l2norm
        except Exception as exc:
            print(f"[l2norm] fla_npu l2norm unavailable ({exc!r}); inline fallback", flush=True)
            _l2norm_fused = None
    return _l2norm_fused


def warm_l2norm():
    """Resolve the l2norm path at wiring time and log which one is live.

    The lazy resolver caches its first result forever, so a transient
    fla_npu import failure would silently pin the inline fallback for
    every later step. Calling this once from the GDN getter surfaces that
    at startup instead.
    """
    fused = _fused_l2norm()
    print(
        f"[l2norm] {'fla_npu triton' if fused is not None else 'inline fallback'}",
        flush=True,
    )
    return fused


_ASCENDC_OPS = (
    "npu_recompute_w_u_fwd",
    "npu_chunk_gated_delta_rule_fwd_h",
    "npu_chunk_fwd_o",
    "npu_chunk_bwd_dv_local",
    "npu_chunk_gated_delta_rule_bwd_dhu",
    "npu_chunk_bwd_dqkwg",
    "npu_prepare_wy_repr_bwd_da",
    "npu_prepare_wy_repr_bwd_full",
)


def _op(name):
    try:
        return getattr(torch.ops.npu, name)
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            f"torch.ops.npu.{name} is not registered; the fla_npu AscendC extension did not load"
        ) from exc


def load_ascendc_ops():
    # The whole loader is the import: fla_npu owns embedded-OPP preparation,
    # torch_npu global-scope pinning and the extension load itself. _op only
    # resolves (and fails loudly if a schema is missing); repeat calls are
    # cheap module-cache hits.
    import fla_npu  # type: ignore[import-untyped]  # noqa: F401 -- the import itself performs the load

    for name in _ASCENDC_OPS:
        _op(name)


def prepare_chunk_indices(cu_seqlens: list, chunk_size: int) -> list:
    """Flattened [sequence_id, chunk_id, ...] pairs covering every chunk."""
    indices = []
    for i in range(len(cu_seqlens) - 1):
        length = cu_seqlens[i + 1] - cu_seqlens[i]
        if length <= 0:
            continue
        for chunk_id in range((length + chunk_size - 1) // chunk_size):
            indices.append(i)
            indices.append(chunk_id)
    return indices


# Host mirrors of the packed-batch cu_seqlens tensors currently in flight,
# keyed by data_ptr and capped to the last few allocations. The dataloader
# allocates a fresh cu_seq_lens_q per micro-batch and every GDN layer (12
# forwards, 12 checkpoint replays, 12 autograd backwards per step) sees
# either that tensor or a detach copy sharing its storage; each cache entry
# holds a strong reference, which pins the allocation, so a matching
# data_ptr is guaranteed to carry the same values. This turns per-layer
# ``tolist()`` device syncs into one sync per fresh allocation (one per
# micro-batch per step).
_CU_SEQLENS_HOST_CACHE: dict = {}
# Sized for the worst split (one main tensor + its slab mirrors; the slab
# split seeds one entry per slab via _seed_cu_seqlens_host below).
_CU_SEQLENS_HOST_CACHE_MAX = 8


def _cu_seqlens_host(cu_seqlens: Optional[torch.Tensor]) -> Optional[list]:
    if cu_seqlens is None:
        return None
    key = cu_seqlens.data_ptr()
    entry = _CU_SEQLENS_HOST_CACHE.get(key)
    if entry is None or entry[0] is not cu_seqlens and entry[0].shape != cu_seqlens.shape:
        host = cu_seqlens.tolist()
        # Store the tensor beside the list: the strong reference pins the
        # allocation, so a later tensor can never be handed this data_ptr
        # while the entry lives (no stale-hit aliasing).
        _seed_cu_seqlens_host(cu_seqlens, host)
        return host
    return entry[1]


def _seed_cu_seqlens_host(cu_seqlens: torch.Tensor, host: list) -> None:
    """Insert a host mirror, evicting the oldest entry at the cap.

    Every insert site goes through here: the slab split seeds one entry per
    slab per fresh allocation, so an uncapped insert would leak one small
    entry per micro-batch per step.
    """
    if len(_CU_SEQLENS_HOST_CACHE) >= _CU_SEQLENS_HOST_CACHE_MAX:
        _CU_SEQLENS_HOST_CACHE.pop(next(iter(_CU_SEQLENS_HOST_CACHE)))
    _CU_SEQLENS_HOST_CACHE[cu_seqlens.data_ptr()] = (cu_seqlens, host)


# Pinned staging buffers for the small host-built index lists, pooled per
# (dtype, length) and reused across misses: fixed-memory buffers are never
# re-allocated per step. A buffer is refilled only after its last H2D has
# landed (event.query(), non-blocking -- a miss during the transient
# in-flight window allocates a fresh buffer instead of ever blocking):
# stream FIFO orders device work against itself, but it cannot order a
# host-side refill against the copy still reading the buffer, and an
# in-flight buffer freed back to the host allocator would be reused by the
# next pinned allocation -- the freed-then-reused corruption form. All
# copies are issued on the current compute stream, which is also the only
# stream every consumer runs on (the eq/cumsum/stack chain and the triton
# helpers here, the AscendC ops for cu_seqlens, and the autograd replay):
# device-side ordering is stream FIFO, the event gates only the host
# refill, and no device-side wait is ever inserted.
_PINNED_STAGE_POOL: dict = {}
_PINNED_STAGE_POOL_MAX = 8


def _h2d_pinned(values: list, dtype: torch.dtype, device) -> torch.Tensor:
    """Copy a small host-built index list to ``device`` without a sync H2D.

    ``torch.tensor(values, device=...)`` is a pageable H2D: libruntime
    stages it behind a rtStreamSynchronize that parks the launching host
    thread and drains the stream queue (the 16-ranks-wait-on-one slow-step
    pattern). A pooled pinned staging buffer + ``non_blocking=True`` copy
    keeps the queue live; see the pool block above for the ordering and
    lifetime argument.
    """
    entries = _PINNED_STAGE_POOL.setdefault((dtype, len(values)), [])
    src: torch.Tensor | None = None
    ev: Any = None
    for cand in entries:
        if cand[1].query():
            src, ev = cand
            break
    if src is None:
        # Every pooled buffer's copy is still in flight (a transient window
        # at copy latency; the same (dtype, len) recurs across the GDN
        # layers of one step): allocate a fresh buffer rather than reuse or
        # block. In-flight entries are never evicted -- only completed ones
        # beyond the cap are dropped, which is a safe pinned free.
        src = torch.tensor(values, dtype=dtype, pin_memory=True)
        # torch.npu.Event, not torch.cuda.Event: the latter is only the npu
        # event after transfer_to_npu maps the namespace (get_device does it
        # lazily), while torch.npu is the real class in every import order.
        ev = torch.npu.Event()  # type: ignore[attr-defined]
        entries.insert(0, (src, ev))
        live = [cand for cand in entries if not cand[1].query()]
        spare = [cand for cand in entries if cand[1].query()]
        entries[:] = (live + spare)[:_PINNED_STAGE_POOL_MAX]
    else:
        src.copy_(torch.tensor(values, dtype=dtype))
    out = src.to(device, non_blocking=True)
    ev.record()
    return out


# Device-tensor chunk indices for the mindspeed triton helpers (cumsum,
# KKT scores, tril inverse), keyed by (cu_seqlens data_ptr, chunk_size).
# mindspeed's own prepare_chunk_indices derives them from
# ``prepare_lens(cu_seqlens).tolist()`` -- a host-blocking D2H that waits
# out the whole queued compute -- behind a single-identity-slot cache the
# three helpers keep evicting for each other (different chunk_size per
# helper) and that the autograd backward's saved-tensor wrappers never hit
# (fresh python object, same storage). That was 112 syncs per step here
# (3 per GDN forward + 1 per backward, 28 GDN calls), each stalling the
# launching thread until the stream drained. This builder reuses the
# already-cached host mirror (one D2H per fresh allocation, see
# _cu_seqlens_host) and rebuilds the [NT, 2] index tensor once per
# (allocation, chunk_size); the strong reference beside the entry pins the
# cu_seqlens allocation, so a data_ptr match is the same values.
_DEVICE_CHUNK_INDICES_CACHE: dict = {}
_DEVICE_CHUNK_INDICES_CACHE_MAX = 8


def _prepare_chunk_indices_device(cu_seqlens: Optional[torch.Tensor], chunk_size: int) -> Optional[torch.Tensor]:
    if cu_seqlens is None:
        return None
    key = (cu_seqlens.data_ptr(), chunk_size)
    entry = _DEVICE_CHUNK_INDICES_CACHE.get(key)
    if entry is not None:
        return entry[1]
    host = _cu_seqlens_host(cu_seqlens)
    assert host is not None  # cu_seqlens was checked non-None above
    flat: list[int] = []
    for i in range(len(host) - 1):
        chunks = (host[i + 1] - host[i] + chunk_size - 1) // chunk_size
        flat.extend(range(chunks))
    idx = _h2d_pinned(flat, cu_seqlens.dtype, cu_seqlens.device)
    # The eq/cumsum stack promotes to int64; the original builder ends with
    # .to(cu_seqlens) (int32 in this pipeline), which also keeps the triton
    # kernel specialization identical to the unpatched arms.
    out = torch.stack([idx.eq(0).cumsum(0) - 1, idx], 1).to(cu_seqlens)
    if len(_DEVICE_CHUNK_INDICES_CACHE) >= _DEVICE_CHUNK_INDICES_CACHE_MAX:
        _DEVICE_CHUNK_INDICES_CACHE.pop(next(iter(_DEVICE_CHUNK_INDICES_CACHE)))
    _DEVICE_CHUNK_INDICES_CACHE[key] = (cu_seqlens, out)
    return out


# Swap the cached builder into the three helper namespaces. The helpers
# pass their own (differing) chunk_size, so the cache needs no knowledge
# of their BT derivations. The AscendC path is fully covered (these three
# triton helpers are its entire triton surface); the frozen triton
# fallback also picks up the patched builder via its kkt import, which is
# harmless (the builder is caller-agnostic and value-identical).
# XTUNER_NPU_GDN_DEVICE_INDICES=0 restores the mindspeed helpers verbatim.
# The three helper modules are
# already loaded by the from-imports above; patch through sys.modules
# rather than re-importing them mid-file.
_ms_kkt_module: Any = sys.modules["mindspeed.ops.triton.chunk_scaled_dot_kkt"]
_ms_cumsum_module: Any = sys.modules["mindspeed.ops.triton.cumsum"]
_ms_solve_tril_module: Any = sys.modules["mindspeed.ops.triton.solve_tril"]

if os.environ.get("XTUNER_NPU_GDN_DEVICE_INDICES", "1") == "1":
    _ms_cumsum_module.prepare_chunk_indices = _prepare_chunk_indices_device
    _ms_kkt_module.prepare_chunk_indices = _prepare_chunk_indices_device
    _ms_solve_tril_module.prepare_chunk_indices = _prepare_chunk_indices_device


# Host-list chunk indices for the AscendC ops, same keying and pinning
# discipline as the device cache above: prepare_chunk_indices rebuilds the
# flat [sequence_id, chunk_id, ...] list from scratch on every fwd/bwd call
# (24+ calls per step per packed batch), so cache it per
# (cu_seqlens allocation, chunk_size).
_CHUNK_INDICES_HOST_CACHE: dict = {}
_CHUNK_INDICES_HOST_CACHE_MAX = 8


def _prepare_chunk_indices_host(cu_seqlens: Optional[torch.Tensor], chunk_size: int) -> Optional[list]:
    if cu_seqlens is None:
        return None
    key = (cu_seqlens.data_ptr(), chunk_size)
    entry = _CHUNK_INDICES_HOST_CACHE.get(key)
    if entry is not None:
        return entry[1]
    host = _cu_seqlens_host(cu_seqlens)
    assert host is not None  # cu_seqlens was checked non-None above
    indices = prepare_chunk_indices(host, chunk_size)
    if len(_CHUNK_INDICES_HOST_CACHE) >= _CHUNK_INDICES_HOST_CACHE_MAX:
        _CHUNK_INDICES_HOST_CACHE.pop(next(iter(_CHUNK_INDICES_HOST_CACHE)))
    # The tensor beside the list pins the allocation (see _CU_SEQLENS_HOST_CACHE).
    _CHUNK_INDICES_HOST_CACHE[key] = (cu_seqlens, indices)
    return indices


# The NPU SequenceContext keeps cu_seq_lens_q on CPU; every layer call used
# to make a fresh H2D copy (delta() wrapper), which both multiplied the tiny
# copies and made each layer's tolist() see a different device allocation.
# This cache performs the conversion once per packed-batch CPU tensor (the
# strong reference pins the CPU allocation, so a matching data_ptr is the
# same values), and every GDN layer, checkpoint replay and autograd backward
# then shares one device tensor and one host mirror.
_CU_SEQLENS_DEVICE_CACHE: dict = {}
_CU_SEQLENS_DEVICE_CACHE_MAX = 4
_EMPTY_SEQ_LOGGED = False


def prepare_cu_seqlens(cu_seqlens: Optional[torch.Tensor], device) -> Optional[torch.Tensor]:
    if cu_seqlens is None:
        return None
    key = cu_seqlens.data_ptr()
    entry = _CU_SEQLENS_DEVICE_CACHE.get(key)
    if entry is not None and entry[0] is cu_seqlens:
        return entry[1]
    if entry is not None and entry[0].shape == cu_seqlens.shape:
        # Same address and shape while the pinned original is alive: a
        # detach view of the same storage, i.e. the same values.
        return entry[1]
    # The AscendC VariableLenTiling processor requires strictly increasing
    # boundaries and rejects zero-length sequences (EZ1001 "seqlens[2] should
    # be larger than seqlens[1]"; hit at step 30 of the 50-step phase, whose
    # data stream carries an empty document). An empty sequence consumes no
    # tokens, and every non-empty sequence starts from zero state when
    # initial_state is None, so dropping its duplicate boundary is exact.
    # The filtered host list is seeded into the mirror cache beside the new
    # device tensor, so the fwd/bwd lookups stay sync-free.
    host = cu_seqlens.tolist()
    filtered = host
    if any(b < a for a, b in zip(host, host[1:])):
        raise ValueError(f"cu_seqlens not non-decreasing: {host}")
    if any(b == a for a, b in zip(host, host[1:])):
        filtered = [host[0]] + [b for a, b in zip(host, host[1:]) if b > a]
        global _EMPTY_SEQ_LOGGED
        if not _EMPTY_SEQ_LOGGED:
            _EMPTY_SEQ_LOGGED = True
            print(
                f"[gdn_ascendc] cu_seqlens: dropped {len(host) - len(filtered)} zero-length sequence(s) from {host}",
                flush=True,
            )
    device_tensor = _h2d_pinned(filtered, cu_seqlens.dtype, device)
    if len(_CU_SEQLENS_DEVICE_CACHE) >= _CU_SEQLENS_DEVICE_CACHE_MAX:
        _CU_SEQLENS_DEVICE_CACHE.pop(next(iter(_CU_SEQLENS_DEVICE_CACHE)))
    _CU_SEQLENS_DEVICE_CACHE[key] = (cu_seqlens, device_tensor)
    # Seed the host mirror unconditionally (filtered is host when nothing
    # was dropped): the fwd/bwd helpers look the DEVICE tensor up by
    # data_ptr, so seeding here also removes the one remaining
    # per-fresh-allocation D2H for the common no-empty-sequence packs.
    _seed_cu_seqlens_host(device_tensor, filtered)
    return device_tensor


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64,
):
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    # Obtain the WY representation; u is the updated value.
    A = chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)

    cu_seqlens1 = _cu_seqlens_host(cu_seqlens)
    chunk_indices = _prepare_chunk_indices_host(cu_seqlens, chunk_size)

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    A = A.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()

    w, u = _op("npu_recompute_w_u_fwd")(
        k, v, beta, A, chunk_size, g=g, gk=None, cu_seqlens=cu_seqlens1, chunk_indices=chunk_indices
    )
    h, v_new, final_state = _op("npu_chunk_gated_delta_rule_fwd_h")(
        k,
        w,
        u,
        g,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens1,
        chunk_indices=chunk_indices,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    o = _op("npu_chunk_fwd_o")(
        q, k, v_new, h, scale, g=g, cu_seqlens=cu_seqlens1, chunk_indices=chunk_indices, chunk_size=chunk_size
    )

    # The AscendC chain is H-major ([B, H, T, D]); hand the contiguous mirrors
    # straight to the caller so forward can save them for the backward, which
    # then runs the whole chain copy-free (it used to re-transpose every saved
    # tensor per call). Only o crosses back to the module's T-major contract.
    # Memory note (measured, review agent 2026-09-20): the mirrors total
    # ~14.5KB/token -- 1.77-3.53 GiB gross at 131-262K tokens (net +1.51-3.03 GiB
    # after subtracting the eliminated transient backward-entry copies) -- and now
    # live from forward until backward instead of being transient as before. In
    # production checkpointing (recompute_ratio=1.0) each layer's mirrors are
    # dropped right after its no-grad forward pass, and backward peak only gains
    # the one layer being recomputed: measured +95.7 MiB at 32K tokens
    # (~+0.77 GiB extrapolated at 262K). Single-graph bwd peak is flat (the
    # removed per-call transposes offset the mirrors). Confirm via the next
    # cluster run's max_memory.
    o = o.transpose(1, 2).contiguous()
    return o, final_state, q, k, v, g, beta, A


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64,
):
    # q/k/v/g/beta/A arrive as the H-major mirrors saved by forward -- contiguous
    # [B, H, T, D] by construction. Implicit precondition: this backward does NO
    # layout/contiguity normalization (the old entry re-transposed every saved
    # tensor), so feeding it T-major or strided tensors would run the whole chain
    # on silently wrong layouts. Only do (and the returned grads) cross the
    # layout boundary.
    cu_seqlens1 = _cu_seqlens_host(cu_seqlens)
    chunk_indices = _prepare_chunk_indices_host(cu_seqlens, chunk_size)

    do = do.transpose(1, 2).contiguous()

    w, u = _op("npu_recompute_w_u_fwd")(
        k, v, beta, A, chunk_size, g=g, gk=None, cu_seqlens=cu_seqlens1, chunk_indices=chunk_indices
    )
    h, v_new, final_state = _op("npu_chunk_gated_delta_rule_fwd_h")(
        k,
        w,
        u,
        g,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens1,
        chunk_indices=chunk_indices,
        output_final_state=False,
        chunk_size=chunk_size,
    )

    dv = _op("npu_chunk_bwd_dv_local")(
        q,
        k,
        do,
        g,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens1,
        chunk_indices=chunk_indices,
        scale=scale,
        chunk_size=chunk_size,
    )
    dh, dh0, dv = _op("npu_chunk_gated_delta_rule_bwd_dhu")(
        q,
        k,
        w,
        do,
        dv,
        g=g,
        gK=None,
        h0=initial_state,
        dht=dht,
        cu_seqlens=cu_seqlens1,
        chunk_indices=chunk_indices,
        scale=scale,
        chunk_size=chunk_size,
    )
    dq, dk, dw, dg = _op("npu_chunk_bwd_dqkwg")(
        q,
        k,
        v_new,
        g,
        h,
        do,
        dh,
        dv,
        chunk_size,
        chunk_indices=chunk_indices,
        scale=scale,
        cu_seqlens=cu_seqlens1,
    )

    dA = _op("npu_prepare_wy_repr_bwd_da")(
        k, v, beta, A, dw, dv, g, cu_seqlens=cu_seqlens1, chunk_indices=chunk_indices, chunk_size=chunk_size
    )
    dk2, dv, db, dg2 = _op("npu_prepare_wy_repr_bwd_full")(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens1,
        chunk_indices=chunk_indices,
    )

    # Accumulate in the chain's H-major layout before the single transpose out.
    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")

    # Gradients return to the module's T-major contract; the reverse cumsum
    # (head_first=False triton kernel) also consumes T-major.
    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()
    dv = dv.transpose(1, 2).contiguous()
    db = db.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, head_first=False)
    return dq, dk, dv, db, dg, dh0


# Long packed graphs blow the Ascend launch-grid limit (65535 blocks) in the
# triton solve_tril_64x64_kernel: its flattened grid is chunk_count x
# num_heads, ~67k at 262144 tokens (96-node job-36 and job-37 both died there
# with EE1003 "coreDim ... less than or equal to 65535"), while 131072 tokens
# (~33k blocks) is the validated ceiling (job-35). Documents are independent
# under cu_seqlens (each starts from a zero state), so the identical kernel
# runs on contiguous document slabs instead; the three triton helpers and the
# eight AscendC ops all shrink with them, forward and backward alike. 0
# disables the split.
def _split_max_tokens_from_env() -> int:
    raw = os.environ.get("XTUNER_NPU_GDN_SPLIT_MAX_TOKENS")
    if not raw:
        return 131072
    try:
        return int(raw)
    except ValueError as exc:
        # RuntimeError on purpose: the getter below (ascendc_chunk_gated_delta_rule_fn)
        # catch-all turns any ImportError into "AscendC unavailable"; a named
        # message here keeps a bad knob value diagnosable in that fallback log line.
        raise RuntimeError(f"XTUNER_NPU_GDN_SPLIT_MAX_TOKENS={raw!r} is not an integer") from exc


_GDN_SLAB_SPLIT_MAX_TOKENS = _split_max_tokens_from_env()
_SLAB_SPLIT_CACHE: dict = {}
_SLAB_SPLIT_CACHE_MAX = 4


def _split_varlen_slabs(cu_seqlens: torch.Tensor) -> list:
    """Contiguous ``[t_lo, t_hi)`` token slabs of whole documents.

    Returns ``[(t_lo, t_hi, slab_cu, slab_host), ...]`` where ``slab_cu`` is
    the device cu_seqlens slice re-based to the slab start and ``slab_host``
    its matching host mirror. Every slab stays <=
    ``_GDN_SLAB_SPLIT_MAX_TOKENS`` unless a single document alone exceeds it.
    Cached per
    cu_seqlens allocation: all 12 GDN layers of a packed batch hand in the
    same device tensor, so the split happens once per micro-batch, not once
    per layer. The host boundaries come from the mirror cache seeded by
    prepare_cu_seqlens (sync-free); each slab cu tensor is seeded into the
    host-mirror cache beside the split, so ``_cu_seqlens_host`` and the
    device chunk-index builder stay sync-free for the slab tensors too.
    """
    key = cu_seqlens.data_ptr()
    entry = _SLAB_SPLIT_CACHE.get(key)
    if entry is not None and entry[0] is cu_seqlens:
        return entry[1]
    cu_host = _cu_seqlens_host(cu_seqlens)
    assert cu_host is not None  # the wrapper only splits non-None cu_seqlens
    slabs = []
    doc_lo = 0
    for i in range(1, len(cu_host) - 1):
        # Close the slab [doc_lo, i) before doc i would push it past the cap,
        # so every slab stays <= the cap unless a single document alone
        # exceeds it (that document then becomes its own slab).
        if cu_host[i + 1] - cu_host[doc_lo] > _GDN_SLAB_SPLIT_MAX_TOKENS:
            t_lo, t_hi = cu_host[doc_lo], cu_host[i]
            slab_cu = cu_seqlens[doc_lo : i + 1] - t_lo
            slab_host = [v - t_lo for v in cu_host[doc_lo : i + 1]]
            slabs.append((t_lo, t_hi, slab_cu, slab_host))
            _seed_cu_seqlens_host(slab_cu, slab_host)
            doc_lo = i
    t_lo = cu_host[doc_lo]
    slab_cu = cu_seqlens[doc_lo:] - t_lo
    slab_host = [v - t_lo for v in cu_host[doc_lo:]]
    slabs.append((t_lo, cu_host[-1], slab_cu, slab_host))
    _seed_cu_seqlens_host(slab_cu, slab_host)
    if len(_SLAB_SPLIT_CACHE) >= _SLAB_SPLIT_CACHE_MAX:
        _SLAB_SPLIT_CACHE.pop(next(iter(_SLAB_SPLIT_CACHE)))
    _SLAB_SPLIT_CACHE[key] = (cu_seqlens, slabs)
    return slabs


class ChunkGatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_size):
        o, final_state, q, k, v, g, beta, A = chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_size
        )
        # Save the H-major contiguous mirrors: the backward AscendC chain
        # consumes exactly these layouts, so it runs copy-free.
        ctx.save_for_backward(q, k, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q, k, v, g, beta, A, ctx.scale, initial_state, do, dht, cu_seqlens, ctx.chunk_size
        )
        # One grad per forward input, in order: q, k, v, g, beta, scale,
        # initial_state, output_final_state, cu_seqlens, chunk_size.
        return dq, dk, dv, dg, db, None, dh0, None, None, None


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
):
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype}, k current type is {k.dtype}, v current type is {v.dtype}, "
            "they should be equal"
        )
    if q.dtype == torch.float32:
        raise ValueError("ChunkGatedDeltaRule does not support float32. Please use bfloat16.")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using cu_seqlens. "
            "Please flatten variable-length inputs before processing."
        )
    if (output_final_state or initial_state is not None) and cu_seqlens is not None:
        # The delta() wrapper below rejects state-carrying calls whose
        # cu_seqlens carries empty documents before its prepare_cu_seqlens
        # filtering could misalign per-sequence state rows, so through that
        # path this never fires; it guards direct callers that bypass the
        # wrapper and hand in a raw cu_seqlens.
        bounds = cu_seqlens.tolist()
        if any(b == a for a, b in zip(bounds, bounds[1:])):
            raise NotImplementedError(
                "empty-sequence filtering is incompatible with "
                "output_final_state/initial_state; duplicate cu_seqlens "
                f"boundaries present: {bounds}"
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        fused = _fused_l2norm()
        if fused is not None:
            # FLA triton l2norm: one kernel per pass instead of the six-op
            # fp32 decomposition; measured within 1 bf16 ulp of the inline
            # expression at the [1, T, H, D] call-site shape.
            q = fused(q, eps=1e-6)
            k = fused(k, eps=1e-6)
        else:

            def l2norm(x, dim=-1, eps=1e-6):
                # The FLA reference (and the fused triton kernel above)
                # computes the norm in float32 regardless of input dtype.
                original_dtype = x.dtype
                x = x.float()
                inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
                return (x * inv_norm).to(original_dtype)

            q = l2norm(q)
            k = l2norm(k)

    # State-carrying calls stay on the single-graph path (the split slices
    # per-document state rows only when they are not needed): training always
    # arrives here with initial_state=None / output_final_state=False.
    if (
        cu_seqlens is not None
        and initial_state is None
        and not output_final_state
        and 0 < _GDN_SLAB_SPLIT_MAX_TOKENS < q.shape[1]
    ):
        o_slabs = []
        for t_lo, t_hi, slab_cu, _slab_host in _split_varlen_slabs(cu_seqlens):
            slab_o, _ = ChunkGatedDeltaRule.apply(
                q[:, t_lo:t_hi],
                k[:, t_lo:t_hi],
                v[:, t_lo:t_hi],
                g[:, t_lo:t_hi],
                beta[:, t_lo:t_hi],
                scale,
                None,
                False,
                slab_cu,
                chunk_size,
            )
            o_slabs.append(slab_o)
        return torch.cat(o_slabs, dim=1), None

    o, final_state = ChunkGatedDeltaRule.apply(
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_size
    )
    return o, final_state


def ascendc_chunk_gated_delta_rule_fn():
    """Resolve the AscendC GDN callable, or None to keep the Triton fallback.

    Loading is the import itself -- fla_npu's wheel __init__ prepares the
    embedded OPP, pins the torch_npu libraries into the global symbol scope
    and torch.ops.load_library()es its AscendC extension. Any load failure
    (wheel absent, schema missing) returns None so the dispatcher keeps
    XTuner's Triton pipeline.
    """
    try:
        load_ascendc_ops()
        warm_l2norm()
        print("[gdn_ascendc] GDN kernel: fla_npu AscendC", flush=True)
    except Exception as exc:
        print(f"[gdn_ascendc] GDN kernel: fla_npu AscendC unavailable ({exc}); Triton fallback", flush=True)
        return None

    def delta(q, k, v, **kwargs):
        # XTuner's NPU SequenceContext keeps these small metadata tensors
        # on CPU for its flash-attention binding; the AscendC path needs
        # them on device. prepare_cu_seqlens converts once per packed
        # batch and shares the device tensor (plus a host mirror) across
        # every GDN layer, checkpoint replay and autograd backward. It
        # also drops zero-length sequence boundaries -- exact only for
        # the training path (initial_state=None, no final state), so a
        # state-carrying call carrying an empty document fails loudly
        # here instead of getting its per-sequence state rows silently
        # misaligned.
        if kwargs.get("cu_seqlens") is not None:
            cu = kwargs["cu_seqlens"]
            if kwargs.get("initial_state") is not None or kwargs.get("output_final_state"):
                bounds = cu.tolist()  # CPU metadata tensor: no device sync
                if any(b == a for a, b in zip(bounds, bounds[1:])):
                    raise NotImplementedError(
                        "empty-sequence filtering is incompatible with "
                        "initial_state/output_final_state; duplicate "
                        f"cu_seqlens boundaries: {bounds}"
                    )
            kwargs["cu_seqlens"] = prepare_cu_seqlens(cu, q.device)
        return chunk_gated_delta_rule(q, k, v, **kwargs)

    return delta
