# Copyright (c) OpenMMLab. All rights reserved.
"""Deterministic DSA backward kernel via subclassing the cuDNN CuTe-DSL kernel.

Instead of vendoring the 2300-line ``dsa_bwd_sm90.py`` from the cudnn package, this
subclass overrides only ``scatter_dkv_atomic`` so each CTA writes its dKV contribution
into its own slice of a widened FP32 accumulator (plain single-writer atomics per
address) instead of racing all CTAs on one shared buffer. The driver
(``cudnn_dsa_det.py``) then reduces the slices in a fixed order. Mechanism follows FA2's
deterministic backward (split buffer + ordered reduction).

Fragility note: the override mirrors cudnn 1.28.0's method body; if the cudnn package
upgrades and changes that method's internals, ``flash_mla_cudnn.py``'s deterministic
dispatch must fall back (guarded by the import/capability check there).
"""

import cutlass
import cutlass.cute as cute
from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm90 import (
    FlashAttentionDSABackwardSm90,
)
from cudnn.deepseek_sparse_attention.utils.sm90.primitives import (
    atomic_add_fp32x4,
    make_acc_tensor_mn_view,
)
from cutlass import Int32, const_expr


class FlashAttentionDSABackwardSm90Det(FlashAttentionDSABackwardSm90):
    """cuDNN SM90 DSA backward with per-CTA dKV accumulator slices.

    The driver must set ``_det_slice_size`` (elements per slice, i.e.
    ``seqlen_k_rounded * head_dim_rounded``) and ``_det_n_splits`` on the instance before
    ``cute.compile``; they are read at trace time and baked in as constants. The widened
    accumulator keeps the stock 3D layout with ``n_splits * slice_size`` elements in the
    last dim, so the stock ``__call__``/transpose paths are unchanged.
    """

    _det_slice_size: int = 0
    _det_n_splits: int = 1

    @cute.jit
    def scatter_dkv_atomic(
        self,
        acc: cute.Tensor,
        chunk_idx: cutlass.Constexpr[int],
        mTopkIdxs_cur: cute.Tensor,
        mdKVaccum_cur: cute.Tensor,
        n_block: Int32,
        topK: Int32,
        thr_mma: cute.TiledMma,
        tidx: Int32,
    ):
        """Stock method + one change: target_ptr gains a per-CTA slice offset.

        Every address within a slice still has a single writer (the stock kernel's
        fragment partition is exclusive and duplicate topk entries accumulate in
        program order on the same thread), so the per-slice atomics are exact, and the
        cross-CTA nondeterminism disappears because CTAs no longer share slices.
        """
        tile_shape = (self.tile_n, self.hdim_chunk)
        cDKV = cute.make_identity_tensor(tile_shape if const_expr(not self.dKV_swapAB) else tile_shape[::-1])
        tScDKV_mn = make_acc_tensor_mn_view(thr_mma.partition_C(cDKV), transpose=self.dKV_swapAB)
        ROW = 0 if const_expr(not self.dKV_swapAB) else 1

        acc_mn = make_acc_tensor_mn_view(acc, transpose=self.dKV_swapAB)
        nrow = const_expr(cute.size(acc_mn.shape[0]))
        ncol = const_expr(cute.size(acc_mn.shape[1]))

        rank = tidx % 4
        # DET: this CTA's private slice base within the widened accumulator.
        split_offset = (cute.arch.block_idx()[0] % self._det_n_splits) * self._det_slice_size

        for r in cutlass.range_constexpr(nrow):
            local_row = tScDKV_mn[r, 0][ROW]
            global_topk_row = n_block * self.tile_n + local_row
            if global_topk_row < topK:
                global_kv_row = mTopkIdxs_cur[global_topk_row]
                should_accumulate_dkv = const_expr(True)
                if const_expr(not self.have_topk_length):
                    should_accumulate_dkv = global_kv_row >= 0
                if should_accumulate_dkv:
                    row_base = global_kv_row * self.tile_hdim + chunk_idx * self.hdim_chunk

                    # Each group of 4 MN-view cols maps to one N-tile.
                    # Interleaved: fake_pos = c4 * 16 + rank * 4
                    for c4 in cutlass.range_constexpr(ncol // 4):
                        fake_offset = const_expr(c4) * 16 + rank * 4
                        c = const_expr(c4 * 4)
                        target_ptr = mdKVaccum_cur.iterator + split_offset + row_base + fake_offset
                        atomic_add_fp32x4(
                            acc_mn[r, c + 0],
                            acc_mn[r, c + 1],
                            acc_mn[r, c + 2],
                            acc_mn[r, c + 3],
                            target_ptr,
                        )
