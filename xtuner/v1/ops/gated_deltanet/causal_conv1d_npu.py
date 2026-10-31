"""NPU causal depthwise conv for GatedDeltaNet: AscendC fused, unmasked, four-tap.

Three paths behind one dispatcher (``causal_conv1d``):

* fused -- the fla_npu AscendC causal_conv1d, one fused varlen kernel per
  direction (masks document boundaries natively via query_start_loc). Two
  backends drive the same embedded-OPP kernels: "stable" calls the decoupled
  runtime's prefill adapter host-metadata path (``npu_causal_conv1d_fn`` with
  ``query_start_loc_cpu``; 26.10+ packaging, bitwise-identical probe-verified;
  falls back to the deprecated raw ``npu_causal_conv1d`` entry on wheels
  without the adapter), "legacy" calls the torch.ops.npu dispatcher names and
  splits the backward per document (26.6 packaging). A one-shot numeric probe
  engages the path; any failure pins the dispatcher to the unmasked/four-tap
  fallback.
* unmasked (``XTUNER_NPU_CONV=1``) -- run the conv unmasked and subtract the
  cross-document contributions at the affected rows, with per-tap
  host mirrors so the backward skips the readbacks.
* four-tap -- the masked per-lag torch loop, the last fallback.

Selected by ``get_causal_conv1d_fn`` when XTUNER_NPU_CONV=1 (the
default); otherwise XTuner's own causal_conv1d_fn stays in place.

``XTUNER_NPU_GDN_API`` does not select the conv backend (it only governs
the chunk GDN op resolution in chunk_gated_delta_rule_npu); the conv
backend is detected automatically and printed once at resolve time.
"""

import os

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


_CONV_UNMASKED = bool(int(os.environ.get("XTUNER_NPU_CONV", "1")))
_DEBUG_CONV_CALLS = [0]


def _strict_kernels() -> bool:
    return os.environ.get("XTUNER_NPU_STRICT_KERNELS", "0") == "1"


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


class _ConvCorrections:
    """Per-pack document-boundary correction indices for the unmasked conv.

    Keyed by the CALLER's original seq_idx tensor (identity + width): in the real pipeline SequenceContext.to() pins
    cu_seq_lens_q on CPU, so gen_seq_idx builds a CPU tensor once per SequenceContext and every GDN layer hands that
    same object to causal_conv1d (no device copy -- the cache derives everything from the host tensor).  Keying by a
    per-call device copy would miss on every layer; keying by the stable source hits once per pack.  The entry pins the
    source tensor, so a data_ptr match is the same values.
    """

    def __init__(self, max_entries=4):
        self._cache = {}
        self._max = max_entries

    def get(self, seq_idx, src, width, device=None):
        # ``device`` (the compute device) places the correction index tensors
        # even though seq_idx/src stay on CPU; default keeps the historical
        # seq_idx.device placement.
        key = (src.data_ptr(), width)
        entry = self._cache.get(key)
        if entry is not None and entry[0] is src:
            return entry[1]
        if entry is not None and entry[0].shape == src.shape:
            # Same address and shape while the pinned original is alive: a
            # detach view of the same storage, i.e. the same values.
            return entry[1]
        if src.device.type == "cpu":
            s = src[0]
        else:
            s = seq_idx[0].to("cpu")
        T = s.numel()
        starts = (s[1:] != s[:-1]).nonzero(as_tuple=True)[0] + 1
        # Offsets are bounded by each document's own length: positions past a
        # short document belong to later documents, and the pairs they would
        # generate are already produced by those documents' own starts, so
        # emitting them here would subtract the same term twice.
        starts_list = starts.tolist()
        bounds = [0] + starts_list + [T]
        lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        dev = seq_idx.device if device is None else device
        groups = []
        for o in range(width - 1):
            lags = list(range(o + 1, width))
            tgt, srcs, valid = [], [], []
            cols = [width - 1 - lag for lag in lags]
            for st, dl in zip(starts_list, lens[1:]):
                if o >= dl:
                    continue
                t = st + o
                if t < T:
                    row, vrow = [], []
                    for lag in lags:
                        sc = t - lag
                        row.append(sc if sc >= 0 else 0)
                        vrow.append(sc >= 0)
                    tgt.append(t)
                    srcs.append(row)
                    valid.append(vrow)
            if tgt:
                t_tgt = torch.tensor(tgt, device=dev, dtype=torch.long)
                t_src = torch.tensor(srcs, device=dev, dtype=torch.long)
                t_valid = torch.tensor(valid, device=dev, dtype=torch.bool)
                t_cols = torch.tensor(cols, device=dev, dtype=torch.long)
                groups.append((t_tgt, t_src, t_valid, t_cols))
                if len(_CONV_GROUP_HOST) >= _CONV_GROUP_HOST_MAX:
                    _CONV_GROUP_HOST.pop(next(iter(_CONV_GROUP_HOST)))
                # Per-tap "any valid row" comes from the python lists the
                # tensors were built from -- no device readback needed. The
                # tensor beside the lists pins the allocation, so a later
                # tensor can never be handed this data_ptr while the entry
                # lives (no stale-hit aliasing).
                _CONV_GROUP_HOST[t_cols.data_ptr()] = (
                    t_cols,
                    cols,
                    [any(vrow[j] for vrow in valid) for j in range(len(cols))],
                )
        out = tuple(groups) if groups else ()
        if len(self._cache) >= self._max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[src.data_ptr()] = (src, out)
        return out


# Host mirrors for the unmasked backward's per-tap loop, keyed by the cols tensor's
# data_ptr. The group tensors are built from python lists at pack-build time,
# so keeping those lists beside the device tensors lets the backward skip the
# per-layer ``cols.tolist()`` (a host-blocking D2H that waits out the whole
# queued stream -- 14 layers x 2 micro-batches x groups per step, and the
# same again for ``valid[:, j].any()`` scalar reads). The strong reference in
# the owning cache pins the allocation, so a data_ptr match is the same
# values; a miss (evicted entry) falls back to the sync path, still correct.
_CONV_GROUP_HOST: dict = {}
# The cap must exceed one step's worst-case live entries, or the backward
# falls into the sync fallback mid-step: unmasked builds up to width-1 correction
# groups per conv layer per micro-batch, so ilmb=2 inserts ~7 x 13 x 2 ~ 182
# mirrors per step and the old cap of 16 thrashed hundreds of host-blocking
# ``tolist()`` readbacks into every backward. Entries are a 7-element tensor
# plus python lists, so a four-digit cap pins kilobytes; entries age out FIFO
# across steps once the per-step population stops growing.
_CONV_GROUP_HOST_MAX = 1024


_conv_corrections = _ConvCorrections()
# Dense short-document packs make the correction gathers grow with the
# document count (rows x taps x channels); past this many correction rows
# they move more bytes than the masked taps and the four-tap wins again.
_CONV_UNMASKED_MAX_ROWS = 512
_unmasked_logged = [False, False]


class _CausalConv1dUnmasked(torch.autograd.Function):
    """Depthwise causal conv run unmasked, cross-document terms subtracted.

    The masked four-tap costs a full-tensor mask multiply and a pad per tap; convolving across document boundaries and
    subtracting exactly the cross-document contributions at the affected rows reaches the same result with fewer full-
    tensor passes and no pads.
    """

    @staticmethod
    def forward(ctx, x, weight, corr, activation):
        width = weight.shape[-1]
        y = x * weight[:, -1].to(x.dtype)
        for lag in range(1, width):
            if lag >= x.shape[1]:
                break
            y[:, lag:] += x[:, :-lag] * weight[:, -1 - lag].to(x.dtype)
        for tgt, src, valid, cols in corr:
            w_rows = weight.t()[cols].to(x.dtype)
            contrib = x[:, src] * w_rows
            contrib = contrib * valid.unsqueeze(-1).to(x.dtype)
            y[:, tgt] -= contrib.sum(2)
        y_pre = y
        if activation in ("silu", "swish"):
            y = F.silu(y)
        ctx.save_for_backward(x, weight, y_pre, *sum((list(g) for g in corr), []))
        ctx.activation = activation
        return y

    @staticmethod
    def backward(ctx, dy):
        saved = ctx.saved_tensors
        x, weight, y_pre = saved[0], saved[1], saved[2]
        groups, idx = [], 3
        while idx < len(saved):
            groups.append(saved[idx : idx + 4])
            idx += 4
        width = weight.shape[-1]
        if ctx.activation in ("silu", "swish"):
            dxs = torch.ops.aten.silu_backward(dy, y_pre)
        else:
            dxs = dy
        dx = dxs * weight[:, -1].to(x.dtype)
        dw = torch.empty(weight.shape, dtype=torch.float32, device=weight.device)
        dw[:, -1] = (dxs * x).sum((0, 1), dtype=torch.float32)
        for lag in range(1, width):
            if lag >= x.shape[1]:
                break
            dx[:, :-lag] += dxs[:, lag:] * weight[:, -1 - lag].to(x.dtype)
            dw[:, -1 - lag] = (dxs[:, lag:] * x[:, :-lag]).sum((0, 1), dtype=torch.float32)
        for tgt, src, valid, cols in groups:
            w_rows = weight.t()[cols].to(x.dtype)
            dt = dxs[:, tgt]
            xs = x[:, src]
            g = dt.unsqueeze(2) * w_rows * valid.unsqueeze(-1).to(x.dtype)
            srcf = src.reshape(-1)
            gf = g.reshape(1, -1, weight.shape[0])
            vf = valid.reshape(-1).to(x.dtype).unsqueeze(-1)
            dx.index_add_(1, srcf, -(gf * vf))
            mirror = _CONV_GROUP_HOST.get(cols.data_ptr())
            if mirror is None:
                # Evicted or foreign entry: rebuild the mirror (one sync) and
                # cache it, so later layers in the same pack skip the readback.
                mirror = (cols, cols.tolist(), [bool(v) for v in valid.any(0).tolist()])
                if len(_CONV_GROUP_HOST) >= _CONV_GROUP_HOST_MAX:
                    _CONV_GROUP_HOST.pop(next(iter(_CONV_GROUP_HOST)))
                _CONV_GROUP_HOST[cols.data_ptr()] = mirror
            _, cols_list, valid_any = mirror
            for j, col in enumerate(cols_list):
                if valid_any[j]:
                    vj = valid[:, j]
                    dw[:, col] -= (dt * xs[:, :, j] * vj.unsqueeze(-1).to(x.dtype)).sum((0, 1), dtype=torch.float32)
        return dx, dw.to(weight.dtype), None, None


class _ConvHostBounds:
    """Per-pack document boundaries [0, s1, ..., T] for the AscendC varlen
    conv.

    Same keying discipline as _ConvCorrections: the CALLER's original seq_idx
    tensor (identity, pinned by the entry) is the stable key -- in the real
    pipeline SequenceContext.to() leaves cu_seq_lens_q on CPU, so one host
    derivation serves every GDN layer in the pack with no device readback.
    """

    def __init__(self, max_entries=4):
        self._cache = {}
        self._max = max_entries

    def get(self, seq_idx, src):
        key = src.data_ptr()
        entry = self._cache.get(key)
        if entry is not None and (entry[0] is src or entry[0].shape == src.shape):
            return entry[1]
        s = src[0] if src.device.type == "cpu" else seq_idx[0].to("cpu")
        T = s.numel()
        starts = (s[1:] != s[:-1]).nonzero(as_tuple=True)[0] + 1
        bounds = [0] + starts.tolist() + [T]
        if len(self._cache) >= self._max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (src, bounds)
        return bounds


_conv_host_bounds = _ConvHostBounds()

# ---------------------------------------------------------------------------
# fla_npu AscendC causal_conv1d, resolved once per process to one of two
# backends for the SAME embedded-OPP kernels:
#   * "stable" -- fla_npu.ops.ascendc.npu_causal_conv1d{,_bwd}, the decoupled
#     runtime's raw host-metadata entries (the 26.10 wheel packaging drops the
#     PyTorch dispatcher .so, so torch.ops.npu.* cannot register there; the
#     stable entries drive the identical aclnn kernels). Detected via
#     fla_npu.load_legacy_torch_ops: 26.6-era wheels lack it and bind the
#     ascendc names to torch.ops forward shells instead, so a successful
#     ascendc import alone proves nothing. Verified numerically on
#     26.10.0+main.dev47d7c97: fwd/dx/dw bf16 few-ulp vs an autograd reference.
#   * "legacy" -- torch.ops.npu.npu_causal_conv1d{,_bwd} via load_ascendc_ops()
#     (26.6 wheels register on import). Multi-doc bwd dw corruption was
#     observed on one 26.6 build under the pre-host-list call form; with the
#     current host-list NTD ABI two 26.6.0 builds probe correct, but backward
#     still splits per document on this backend as the shock absorber.
# Launches are async on torch's current NPU stream; fwd passes a zero bias and
# per-sequence zero initial state, bwd runs in NTD layout. A one-shot NUMERIC
# probe (below) engages the path; any failure (missing backend, wrong output)
# pins the dispatcher to the unmasked/four-tap fallback.
#
# Env: CANN's set_env.sh must be sourced before import fla_npu; the package
# itself sets ASCEND_CUSTOM_OPP_PATH and loads its opapi libs at import time.
# ---------------------------------------------------------------------------

_OPS_ENSURED = [False]
_CONV_BACKEND = {"name": None, "fwd": None, "bwd": None}


def _ensure_conv_ops():
    """Resolve the causal-conv backend exactly once.

    The decoupled runtime is detected by ``fla_npu.load_legacy_torch_ops``:
    wheels that ship it (26.10+) expose genuine raw host-metadata entries on
    ``fla_npu.ops.ascendc`` ("stable"). Older wheels (26.6) bind those names
    to torch.ops.npu forward shells (verified: their __doc__ reads
    "Call torch.ops.npu.npu_causal_conv1d."), so importing them succeeds
    while meaning something else; on those we register the legacy torch.ops
    dispatcher names and split the backward per document.
    """
    if not _OPS_ENSURED[0]:
        try:
            import fla_npu

            decoupled = getattr(fla_npu, "load_legacy_torch_ops", None) is not None
        except ImportError:
            decoupled = False
        if decoupled:
            try:
                # 26.10+ packaging: the prefill adapter's host-metadata path
                # (query_start_loc_cpu) replaces the deprecated raw
                # npu_causal_conv1d compat entry (FutureWarning, removal
                # 2027/02); bitwise-identical on the dev wheel.
                from fla_npu.ops.ascendc import npu_causal_conv1d_fn as fwd, npu_causal_conv1d_bwd as bwd

                _CONV_BACKEND.update(name="stable", fwd=fwd, bwd=bwd, host_list=True)
            except ImportError:
                from fla_npu.ops.ascendc import npu_causal_conv1d as fwd, npu_causal_conv1d_bwd as bwd

                _CONV_BACKEND.update(name="stable", fwd=fwd, bwd=bwd, host_list=False)
        else:
            from .chunk_gated_delta_rule_npu import load_ascendc_ops

            load_ascendc_ops()
            assert hasattr(torch.ops.npu, "npu_causal_conv1d") and hasattr(torch.ops.npu, "npu_causal_conv1d_bwd"), (
                "fla_npu extension missing npu_causal_conv1d{,_bwd}"
            )
            _CONV_BACKEND.update(name="legacy")
        _OPS_ENSURED[0] = True
        print(f"[causal_conv1d] conv backend resolved: {_CONV_BACKEND['name']}", flush=True)
        if os.environ.get("XTUNER_NPU_GDN_API"):
            print(
                "[causal_conv1d] note: XTUNER_NPU_GDN_API no longer selects the conv backend; "
                "it only governs the chunk GDN op resolution",
                flush=True,
            )


def _conv_fwd_aclnn(x2d, weight_kc, cu, activation_mode=0):
    """AscendC causal conv forward.

    x2d [T, C] bf16, weight_kc [K, C] bf16
    (the transposed, conv-native layout), cu = host list of per-sequence
    token starts + total ([0, s1, s2, ..., T]). Returns pre-activation
    [T, C] bf16. Async launch on the current stream (stable: the raw
    host-metadata entry; legacy: torch.ops dispatch).
    """
    _ensure_conv_ops()
    x2d = x2d.contiguous()
    weight_kc = weight_kc.contiguous()
    C = x2d.shape[-1]
    K = weight_kc.shape[0]
    n_seq = len(cu) - 1
    if os.environ.get("XTUNER_NCP_DEBUG_CAUSAL", "0") == "1" and _DEBUG_CONV_CALLS[0] < 4:
        call_index = _DEBUG_CONV_CALLS[0]
        _DEBUG_CONV_CALLS[0] += 1
        print(
            f"[causal_conv_debug] call={call_index} "
            f"x={tuple(x2d.shape)} weight={tuple(weight_kc.shape)} states={n_seq}x{K}x{C} "
            f"qsl_len={len(cu)} qsl_first={cu[:4]} qsl_last={cu[-4:]} "
            f"x_dtype={x2d.dtype} weight_dtype={weight_kc.dtype}",
            flush=True,
        )
    bias = torch.zeros(C, dtype=x2d.dtype, device=x2d.device)
    # Zero initial history per sequence; tiling accepts [N, K, C] zeros (the
    # kernel only reads the last K-1 columns as pre-sequence context).
    states = torch.zeros(n_seq, K, C, dtype=x2d.dtype, device=x2d.device)
    if _CONV_BACKEND["name"] == "stable":
        if _CONV_BACKEND.get("host_list"):
            # Prefill adapter's host-metadata path: host-list query_start_loc_cpu,
            # string activation. Bitwise-identical to the deprecated raw
            # npu_causal_conv1d entry on the 26.10 dev wheel (probe: both
            # activation modes + doc-boundary identity). The adapter's
            # device-metadata path stays excluded (miscomputes on this wheel).
            return _CONV_BACKEND["fwd"](
                x2d,
                weight_kc,
                bias,
                conv_states=states,
                query_start_loc_cpu=list(cu),
                activation="none" if activation_mode == 0 else "silu",
            )
        # Older packaging without the fn adapter: the deprecated raw
        # host-metadata entry (no FutureWarning on those wheels).
        return _CONV_BACKEND["fwd"](
            x=x2d,
            weight=weight_kc,
            bias=bias,
            conv_states=states,
            query_start_loc=list(cu),
            activation_mode=activation_mode,
        )
    return torch.ops.npu.npu_causal_conv1d(
        x2d,
        weight_kc,
        bias,
        states,
        query_start_loc=list(cu),
        activation_mode=activation_mode,
    )


def _conv_bwd_aclnn(x2d, y3d, weight_kc, dy3d, cu, activation=1):
    """AscendC causal conv backward (NTD layout).

    x2d [T, C] bf16, y3d/dy3d [1, T, C] bf16 (y = saved pre-activation
    when activation=1: the op applies silu_backward internally; y3d may be
    None with activation=0), weight_kc [K, C], cu as in _conv_fwd_aclnn.
    Returns (dx [T, C], dw [K, C]) -- dw in the conv-native layout;
    transpose(0, 1) for the [C, K] torch layout.
    """
    _ensure_conv_ops()
    x2d = x2d.contiguous()
    weight_kc = weight_kc.contiguous()
    dy3d = dy3d.contiguous()
    if y3d is not None:
        y3d = y3d.contiguous()
    if _CONV_BACKEND["name"] == "stable":
        out = _CONV_BACKEND["bwd"](
            x=x2d,
            y=y3d,
            weight=weight_kc,
            dy=dy3d,
            initial_state=None,
            dht=None,
            query_start_loc=list(cu),
            activation=activation,
            input_layout="NTD",
        )
        return out[0], out[1]
    dx, dw, _db, _dh0 = torch.ops.npu.npu_causal_conv1d_bwd(
        x2d,
        y3d,
        weight_kc,
        dy3d,
        query_start_loc=list(cu),
        activation=activation,
        input_layout="NTD",
    )
    return dx, dw


_fused_state = {"probed": False, "ok": False}


def _fused_conv_available(device):
    """Lazy one-shot probe of the torch.ops fused conv path.

    Loads the fla_npu extension on first use; any failure (extension absent, tiling rejection) marks the fused path off
    for the process lifetime and the dispatcher falls back to unmasked/four-tap.
    """
    if not _fused_state["probed"]:
        _fused_state["probed"] = True
        try:
            T, C, K = 64, 128, 4

            def conv_ref(xr, wr, length):
                # fp32 depthwise causal conv reference, bf16-independent.
                return F.conv1d(
                    xr.float().unsqueeze(0).transpose(1, 2),
                    wr.float().transpose(0, 1).unsqueeze(1),
                    None,
                    padding=K - 1,
                    groups=C,
                )[..., :length].squeeze(0).transpose(0, 1)

            x = torch.randn(T, C, dtype=torch.bfloat16, device=device) * 0.5
            w = torch.randn(K, C, dtype=torch.bfloat16, device=device) * 0.5
            y = _conv_fwd_aclnn(x, w, [0, T])
            # NUMERIC forward gate: a finite-but-wrong kernel (the 26.10
            # npu_causal_conv1d_fn decode adapter computed garbage) must be
            # rejected here, not at loss time.
            ref = conv_ref(x, w, T)
            fwd_err = (y.float() - ref).abs().max().item()
            assert fwd_err < 0.15, f"conv fwd mismatch vs reference (max_err={fwd_err:.5f})"
            # Exercise the backward too (the raw-gradient, no-y route) and gate
            # it numerically through an autograd reference.
            dx, dw = _conv_bwd_aclnn(x, None, w, y.unsqueeze(0), [0, T], activation=0)
            torch.npu.synchronize()
            assert dx.shape == x.shape and torch.isfinite(dx.float()).all()
            assert dw.shape == w.shape and torch.isfinite(dw.float()).all()
            # enable_grad: the probe usually fires during model build under
            # no_grad, where even requires_grad inputs build no graph.
            with torch.enable_grad():
                xr = x.detach().clone().requires_grad_(True)
                wr = w.detach().clone().requires_grad_(True)
                yr = conv_ref(xr, wr, T)
                (yr * y.detach().float()).sum().backward()
            dx_err = (dx.float() - xr.grad.float()).abs().max().item()
            dw_err = (dw.float() - wr.grad.float()).abs().max().item()
            assert dx_err < 0.15 and dw_err < 0.15, f"conv bwd mismatch (dx_err={dx_err:.5f}, dw_err={dw_err:.5f})"
            # Training route: activation=1 with the saved pre-activation -- the
            # op applies silu_backward internally, a branch the raw-gradient
            # route above never executes. Gate it through the same reference.
            with torch.enable_grad():
                xs = x.detach().clone().requires_grad_(True)
                ws = w.detach().clone().requires_grad_(True)
                gs = (torch.randn(T, C, dtype=torch.float32, device=device) * 0.5).to(torch.bfloat16)
                F.silu(conv_ref(xs, ws, T)).backward(gradient=gs.float())
            dx_s, dw_s = _conv_bwd_aclnn(x, y.unsqueeze(0), w, gs.unsqueeze(0), [0, T], activation=1)
            torch.npu.synchronize()
            dxs_err = (dx_s.float() - xs.grad.float()).abs().max().item()
            dws_err = (dw_s.float() - ws.grad.float()).abs().max().item()
            assert dxs_err < 0.15 and dws_err < 0.15, (
                f"conv bwd silu-route mismatch (dx_err={dxs_err:.5f}, dw_err={dws_err:.5f})"
            )
            # Multi-doc backward gate: single-doc calls never execute the
            # kernel's multi-doc dw reduction -- the axis one 26.6 build
            # corrupted historically. Two docs at T>=256 cover it (each doc is
            # a zero-context conv, so the reference splits per document).
            T2, half = 512, 256
            cu2 = [0, half, T2]
            x2 = torch.randn(T2, C, dtype=torch.bfloat16, device=device) * 0.5
            y2 = _conv_fwd_aclnn(x2, w, cu2)
            with torch.enable_grad():
                x2r = x2.detach().clone().requires_grad_(True)
                w2r = w.detach().clone().requires_grad_(True)
                g2 = (torch.randn(T2, C, dtype=torch.float32, device=device) * 0.5).to(torch.bfloat16)
                ref2 = torch.cat([conv_ref(x2r[:half], w2r, half), conv_ref(x2r[half:], w2r, half)])
                (ref2 * g2.float()).sum().backward()
            dx2, dw2 = _conv_bwd_aclnn(x2, None, w, g2.unsqueeze(0), cu2, activation=0)
            torch.npu.synchronize()
            dx2_err = (dx2.float() - x2r.grad.float()).abs().max().item()
            dw2_err = (dw2.float() - w2r.grad.float()).abs().max().item()
            assert dx2_err < 0.15 and dw2_err < 0.15, (
                f"conv multi-doc bwd mismatch (dx_err={dx2_err:.5f}, dw_err={dw2_err:.5f})"
            )
            _fused_state["ok"] = True
            print(
                f"[causal_conv1d] conv fused engaged: fla_npu AscendC causal_conv1d "
                f"({_CONV_BACKEND['name']} backend, fwd_err={fwd_err:.5f})",
                flush=True,
            )
        except Exception as exc:
            if _strict_kernels():
                raise RuntimeError(f"strict NCP kernel mode rejected causal-conv fallback: {exc}") from exc
            print(
                f"[causal_conv1d] conv fused unavailable ({exc}); unmasked/tap fallback",
                flush=True,
            )
    return _fused_state["ok"]


class _CausalConv1dFused(torch.autograd.Function):
    """fla_npu AscendC causal_conv1d: one fused varlen op per direction.

    The AscendC kernel masks document boundaries natively via query_start_loc
    (per-sequence zero history), so the unmasked correction machinery disappears
    on the forward. silu runs inside the Function like the fla reference:
    backward hands the saved pre-activation to the bwd op with activation=1 and
    the op applies silu_backward internally. On the stable backend the bwd is a
    single varlen call (multi-doc dw verified on 26.10); the legacy 26.6
    backend splits per document (multi-doc dw corruption observed on one 26.6
    build; the split stays as the shock absorber for any wheel whose multi-doc
    reduction regresses).
    """

    @staticmethod
    def forward(ctx, x, weight, bounds, activation):
        ctx.weight_dtype = weight.dtype
        op_weight = weight.to(x.dtype).transpose(-1, -2).contiguous()
        op_x = x.reshape(-1, x.shape[-1]).contiguous()
        pre = _conv_fwd_aclnn(op_x, op_weight, bounds)
        ctx.save_for_backward(op_x, op_weight, pre)
        ctx.bounds = bounds
        ctx.activation = activation
        ctx.x_shape = tuple(x.shape)
        if activation in ("silu", "swish"):
            pre = F.silu(pre)
        return pre.reshape(ctx.x_shape)

    @staticmethod
    def backward(ctx, dy):
        op_x, op_weight, pre = ctx.saved_tensors
        # NTD dy carries a head-split leading dim ([N, T, Dh]); [1, T, C] is
        # the identity split for a channel-independent depthwise conv.
        op_dy = dy.reshape(1, -1, dy.shape[-1]).contiguous()
        act_bwd = 1 if ctx.activation in ("silu", "swish") else 0
        y3d = pre.unsqueeze(0) if act_bwd == 1 else None
        bounds = ctx.bounds
        if _CONV_BACKEND["name"] == "legacy" and len(bounds) > 2:
            # Multi-doc bwd dw corruption was observed on one 26.6 build (under
            # the pre-host-list call form; two 26.6.0 builds probe correct with
            # the current ABI). Splitting the bwd per document keeps every call
            # single-doc (verified correct up to T=4096) at one extra launch
            # per extra document -- still far below the ~40-pass unmasked
            # decomposition this fused path replaces.
            dx_parts = []
            dw_sum = None
            for n in range(len(bounds) - 1):
                lo, hi = bounds[n], bounds[n + 1]
                dx_n, dw_n = _conv_bwd_aclnn(
                    op_x[lo:hi],
                    y3d[:, lo:hi] if y3d is not None else None,
                    op_weight,
                    op_dy[:, lo:hi],
                    [0, hi - lo],
                    activation=act_bwd,
                )
                dx_parts.append(dx_n)
                dw_sum = dw_n if dw_sum is None else dw_sum + dw_n
            dx = torch.cat(dx_parts, dim=0)
            dw = dw_sum
        else:
            dx, dw = _conv_bwd_aclnn(
                op_x,
                y3d,
                op_weight,
                op_dy,
                bounds,
                activation=act_bwd,
            )
        return dx.reshape(ctx.x_shape), dw.transpose(0, 1).contiguous().to(ctx.weight_dtype), None, None


def causal_conv1d(x, weight, bias=None, activation="silu", seq_idx=None):
    """Channel-last causal depthwise convolution with document boundaries.

    fused path dispatches to the fla_npu AscendC op: one fused varlen kernel per
    direction instead of the ~40-pass unmasked tap/correction decomposition; it
    takes widths up to 4 and no bias. The unmasked branch runs without a mask and
    subtracts the cross-document terms afterwards (validated across
    mixed/short document layouts) and covers wider widths; the four-tap loop
    stays as the last fallback.
    """
    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError("expected B,T,C input and C,K kernel")
    weight = local(weight)
    if activation not in (None, "silu", "swish"):
        # Both paths validate: the unmasked apply() would otherwise ignore an
        # unsupported activation silently.
        raise ValueError("unsupported causal convolution activation")
    width = weight.shape[-1]
    # The AscendC kernel caps width at 4 (causal_conv1d_common.h MAX_WIDTH);
    # wider kernels fall to unmasked/four-tap, which take any width.
    if bias is None and width <= 4 and x.shape[0] == 1 and x.ndim == 3:
        if _fused_conv_available(x.device):
            if seq_idx is None:
                bounds = [0, x.shape[1]]
            else:
                bounds = _conv_host_bounds.get(seq_idx, seq_idx)
            return _CausalConv1dFused.apply(x, weight, bounds, activation)
    if _CONV_UNMASKED and _strict_kernels():
        raise RuntimeError(
            "strict NCP kernel mode requires fused causal_conv1d with no bias, "
            f"width<=4 and batch=1, got bias={bias is not None}, width={width}, batch={x.shape[0]}"
        )
    if _CONV_UNMASKED and bias is None and width <= 8:
        if seq_idx is None:
            return _CausalConv1dUnmasked.apply(x, weight, (), activation)
        assert x.shape[0] == 1, "unmasked corrections assume a single packed row"
        corr = _conv_corrections.get(seq_idx, seq_idx, width, device=x.device)
        rows = sum(g[0].shape[0] for g in corr)
        if rows <= _CONV_UNMASKED_MAX_ROWS:
            if not _unmasked_logged[0]:
                _unmasked_logged[0] = True
                print(
                    f"[causal_conv1d] conv unmasked engaged: {rows} correction rows for width {width}",
                    flush=True,
                )
            return _CausalConv1dUnmasked.apply(x, weight, corr, activation)
        if not _unmasked_logged[1]:
            _unmasked_logged[1] = True
            print(
                f"[causal_conv1d] conv unmasked dense-pack fallback: {rows} correction "
                f"rows exceeds cap {_CONV_UNMASKED_MAX_ROWS}",
                flush=True,
            )
    if seq_idx is not None:
        seq_idx = seq_idx.to(device=x.device)
    y = x * weight[:, -1].to(x.dtype)
    for lag in range(1, width):
        if lag >= x.shape[1]:
            continue
        term = x[:, :-lag] * weight[:, -1 - lag].to(x.dtype)
        if seq_idx is not None:
            term = term * (seq_idx[:, lag:] == seq_idx[:, :-lag]).unsqueeze(-1)
        y = y + F.pad(term, (0, 0, lag, 0))
    if bias is not None:
        y = y + local(bias)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y
