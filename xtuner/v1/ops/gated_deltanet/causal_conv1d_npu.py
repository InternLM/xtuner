"""NPU causal depthwise conv for GatedDeltaNet: AscendC fused, unmasked, four-tap.

Three paths behind one dispatcher (``causal_conv1d``):

* fused -- the fla_npu AscendC causal_conv1d via the torch extension
  ``torch.ops.npu.npu_causal_conv1d{,_bwd}`` (one fused varlen kernel per
  direction; masks document boundaries natively via query_start_loc). A
  one-shot probe engages the path; any failure pins the dispatcher to the
  unmasked/four-tap fallback.
* unmasked (``XTUNER_NPU_CONV=1``) -- run the conv unmasked and subtract the
  cross-document contributions at the affected rows, with per-tap
  host mirrors so the backward skips the readbacks.
* four-tap -- the masked per-lag torch loop, the last fallback.

Selected by ``get_causal_conv1d_fn`` when XTUNER_NPU_CONV=1 (the
default); otherwise XTuner's own causal_conv1d_fn stays in place.
"""

import os

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


_CONV_UNMASKED = bool(int(os.environ.get("XTUNER_NPU_CONV", "1")))


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


class _ConvCorrections:
    """Per-pack document-boundary correction indices for the unmasked conv.

    Keyed by the CALLER's original seq_idx tensor (identity + width): in the
    real pipeline SequenceContext.to() pins cu_seq_lens_q on CPU, so
    gen_seq_idx builds a CPU tensor once per SequenceContext and every GDN
    layer hands that same object to causal_conv1d (no device copy -- the
    cache derives everything from the host tensor).  Keying by a per-call
    device copy would miss on every layer; keying by the stable source hits
    once per pack.  The entry pins the source tensor, so a data_ptr match is
    the same values.
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

    The masked four-tap costs a full-tensor mask multiply and a pad per tap;
    convolving across document boundaries and subtracting exactly the
    cross-document contributions at the affected rows reaches the same result
    with fewer full-tensor passes and no pads.
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
    """Per-pack document boundaries [0, s1, ..., T] for the AscendC varlen conv.

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
# fla_npu AscendC causal_conv1d via the fla_npu torch extension:
# torch.ops.npu.npu_causal_conv1d{,_bwd}, registered by the wheel's
# custom_aclnn_extension_lib .so and loaded lazily through the gated_deltanet
# AscendC loader (npu.load_ascendc_ops) -- the same torch_npu-managed loader
# the AscendC GDN ops use. Launches are async on torch's current NPU stream;
# fwd passes a zero bias and per-sequence zero initial state, bwd runs in
# NTD layout. A one-shot probe (below) engages the path; any failure pins
# the dispatcher to the unmasked/four-tap fallback.
#
# Env: CANN's set_env.sh must be sourced before import fla_npu; the package
# itself sets ASCEND_CUSTOM_OPP_PATH and loads its opapi libs at import time.
# ---------------------------------------------------------------------------

_OPS_ENSURED = [False]


def _ensure_conv_ops():
    """Load the fla_npu torch extension exactly once (GDN's loader path).

    Registration goes through `import fla_npu` (load_ascendc_ops): the
    package's __init__ prepares the embedded OPP and torch.ops.load_library()es
    its own custom_aclnn_extension_lib -- the same .so the AscendC GDN ops
    use, so by the time a conv runs during training this is usually already
    done (idempotent).
    """
    if not _OPS_ENSURED[0]:
        from .chunk_gated_delta_rule_npu import load_ascendc_ops

        load_ascendc_ops()
        assert hasattr(torch.ops.npu, "npu_causal_conv1d") and hasattr(torch.ops.npu, "npu_causal_conv1d_bwd"), (
            "fla_npu extension missing npu_causal_conv1d{,_bwd}"
        )
        _OPS_ENSURED[0] = True


def _conv_fwd_aclnn(x2d, weight_kc, cu, activation_mode=0):
    """AscendC causal conv forward. x2d [T, C] bf16, weight_kc [K, C] bf16
    (the transposed, conv-native layout), cu = host list of per-sequence
    token starts + total ([0, s1, s2, ..., T]). Returns pre-activation
    [T, C] bf16. Async launch on the current stream via the fla_npu torch
    extension (torch_npu-managed opapi loading).
    """
    _ensure_conv_ops()
    x2d = x2d.contiguous()
    weight_kc = weight_kc.contiguous()
    C = x2d.shape[-1]
    K = weight_kc.shape[0]
    n_seq = len(cu) - 1
    bias = torch.zeros(C, dtype=x2d.dtype, device=x2d.device)
    # Zero initial history per sequence; tiling accepts [N, K, C] zeros (the
    # kernel only reads the last K-1 columns as pre-sequence context).
    states = torch.zeros(n_seq, K, C, dtype=x2d.dtype, device=x2d.device)
    return torch.ops.npu.npu_causal_conv1d(
        x2d,
        weight_kc,
        bias,
        states,
        query_start_loc=list(cu),
        activation_mode=activation_mode,
    )


def _conv_bwd_aclnn(x2d, y3d, weight_kc, dy3d, cu, activation=1):
    """AscendC causal conv backward (NTD layout). x2d [T, C] bf16, y3d/dy3d
    [1, T, C] bf16 (y = saved pre-activation when activation=1: the op
    applies silu_backward internally; y3d may be None with activation=0),
    weight_kc [K, C], cu as in _conv_fwd_aclnn. Returns (dx [T, C],
    dw [K, C]) -- dw in the conv-native layout; transpose(0, 1) for the
    [C, K] torch layout.
    """
    _ensure_conv_ops()
    x2d = x2d.contiguous()
    weight_kc = weight_kc.contiguous()
    dy3d = dy3d.contiguous()
    if y3d is not None:
        y3d = y3d.contiguous()
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

    Loads the fla_npu extension on first use; any failure (extension
    absent, tiling rejection) marks the fused path off for the process
    lifetime and the dispatcher falls back to unmasked/four-tap.
    """
    if not _fused_state["probed"]:
        _fused_state["probed"] = True
        try:
            x = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
            w = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
            y = _conv_fwd_aclnn(x, w, [0, 64])
            # Exercise the backward too (the raw-gradient, no-y route): the
            # op exists but the bwd tiling can in principle reject where fwd
            # passed; better to learn it here than mid-training.
            dx, dw = _conv_bwd_aclnn(x, None, w, y.unsqueeze(0), [0, 64], activation=0)
            torch.npu.synchronize()
            assert y.shape == (64, 128) and torch.isfinite(y.float()).all()
            assert dx.shape == x.shape and torch.isfinite(dx.float()).all()
            assert dw.shape == w.shape and torch.isfinite(dw.float()).all()
            _fused_state["ok"] = True
            print("[causal_conv1d] conv fused engaged: fla_npu AscendC causal_conv1d (torch.ops)", flush=True)
        except Exception as exc:
            print(
                f"[causal_conv1d] conv fused unavailable ({exc}); unmasked/tap fallback",
                flush=True,
            )
    return _fused_state["ok"]


class _CausalConv1dFused(torch.autograd.Function):
    """fla_npu AscendC causal_conv1d: one fused varlen op per direction.

    The AscendC kernel masks document boundaries natively via query_start_loc
    (per-sequence zero history), so the unmasked correction machinery disappears.
    silu runs inside the Function like the fla reference: backward hands the
    saved pre-activation to the bwd op with activation=1 and the op applies
    silu_backward internally.
    """

    @staticmethod
    def forward(ctx, x, weight, bounds, activation):
        op_weight = weight.transpose(-1, -2).contiguous()
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
        dx, dw = _conv_bwd_aclnn(
            op_x,
            pre.unsqueeze(0) if ctx.activation in ("silu", "swish") else None,
            op_weight,
            op_dy,
            ctx.bounds,
            activation=1 if ctx.activation in ("silu", "swish") else 0,
        )
        return dx.reshape(ctx.x_shape), dw.transpose(0, 1).contiguous(), None, None


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
