# Copyright (c) OpenMMLab. All rights reserved.
"""Fused GLM-5.2 DSA indexer top-k implemented with NVIDIA CuTe DSL."""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import pipeline
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext


class _CuteDSLIndexerTopK:
    """SM90 WGMMA score computation fused with exact radix top-k."""

    BLOCK_N = 128
    QUERY_BLOCK = 4
    THREADS = 512
    RADIX8_BINS = 257
    RADIX11_BINS = 2048
    ORDERED_NEG_INF = 0x007FFFFF

    def __init__(self, heads: int, index_dim: int, topk: int, kv_len: int):
        assert heads == 32
        assert index_dim == 128
        assert topk in (1024, 2048)
        self.heads = heads
        self.index_dim = index_dim
        self.topk = topk
        self.merge_reuses_k = True
        # topk=2048 retains the compact 8-bit radix scratch and fits 79
        # candidate tiles. The 11-bit topk=1024 specialization overlays its
        # larger histogram with K and fits 75 tiles.
        max_candidate_tiles = 79 if topk == 2048 else 75
        max_candidates = max_candidate_tiles * self.BLOCK_N
        max_batch = max_candidates - topk
        merge_count = 1 + max(
            0,
            (kv_len - max_candidates + max_batch - 1) // max_batch,
        )
        # The 8-bit topk=2048 radix path has a higher fixed compaction cost;
        # keep its four-merge specialization on the fixed-batch path.
        min_compaction_merges = 4 if topk == 1024 else 5
        self.use_candidate_compaction = merge_count >= min_compaction_merges
        # The six-merge topk=2048 band consistently regresses with the wider
        # candidate table, so it retains the int32 specialization.
        self.use_compact_relative_ids = (
            self.use_candidate_compaction and topk == 2048 and kv_len <= 65536 and merge_count != 6
        )
        # Int32 topk=2048 checks counters every other score tile. Reserving
        # three tiles keeps the append bound exact between those checks.
        self.use_sparse_candidate_checks = (
            self.use_candidate_compaction and topk == 2048 and not self.use_compact_relative_ids
        )
        self.candidate_index_dtype = cutlass.Uint16 if self.use_compact_relative_ids else cutlass.Int32
        self.store_initial_candidate_ids = self.use_candidate_compaction and topk == 2048
        self.use_packed_intermediate_output = topk == 1024 and merge_count > 1
        required_candidates = (kv_len + (merge_count - 1) * topk + merge_count - 1) // merge_count
        self.candidate_count = max(
            4 * topk,
            ((required_candidates + self.BLOCK_N - 1) // self.BLOCK_N) * self.BLOCK_N,
        )
        if self.use_candidate_compaction:
            # Score and absolute-ID tables jointly consume 32 B per candidate
            # across four queries before the optional relative-ID packing.
            # Capacities are tuned independently per radix path.
            if self.use_compact_relative_ids:
                compact_candidate_tiles = 41
            else:
                compact_candidate_tiles = 35 if topk == 1024 else 38
            self.candidate_count = compact_candidate_tiles * self.BLOCK_N
        shared_relative_ids = (
            topk == 1024 and merge_count > 1 and kv_len <= 65535 and not self.use_candidate_compaction
        )
        # Up to 48 tiles, an 8 KiB persistent table fits without raising the
        # shared-memory carveout. The measured overlay winners end at 56
        # tiles, plus the cheaper power-of-two 64-tile specialization.
        self.use_persistent_shared_ids = shared_relative_ids and self.candidate_count <= 48 * self.BLOCK_N
        self.use_overlay_shared_ids = (
            shared_relative_ids
            and not self.use_persistent_shared_ids
            and (self.candidate_count <= 56 * self.BLOCK_N or self.candidate_count == 64 * self.BLOCK_N)
        )
        self.use_shared_intermediate_output = self.use_persistent_shared_ids or self.use_overlay_shared_ids
        # Pairwise K-stage reuse trades registers for less WGMMA waiting. It
        # wins for topk=1024, the four-topk table, and online compaction.
        self.use_paired_k_stage = topk == 1024 or self.use_candidate_compaction or self.candidate_count == 4 * topk
        # Pair-local tile readiness only amortizes its named-barrier phases in
        # the long topk=2048 compaction path.
        self.use_pair_scoped_k_ready = topk == 2048 and self.use_candidate_compaction
        # Keeping score consumers in their producing warp group wins for the
        # paired topk=2048 paths; the fixed-batch middle band regresses.
        self.use_warpgroup_score_owners = topk == 2048 and self.use_paired_k_stage
        # Warp collection amortizes only in the packed relative-ID band;
        # its fixed ballot/prefix cost regresses every int32 specialization.
        self.use_warp_aggregated_append = self.use_compact_relative_ids
        # TMA removes per-warp copy instructions and synchronization. The
        # packed online-compaction path instead benefits from pair-local
        # cp.async overlap, so keep that specialization unchanged.
        self.use_tma_k_stage = not self.use_compact_relative_ids
        self.batch_size = self.candidate_count - topk
        self.retained_per_thread = self.QUERY_BLOCK * topk // self.THREADS

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        weights: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        output: cute.Tensor,
    ):
        matrix_m = self.QUERY_BLOCK * self.heads
        tile_shape = (matrix_m, self.BLOCK_N, self.index_dim)
        q_layout_staged = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR,
            tile_shape,
            cutlass.BFloat16,
            1,
        )
        k_layout_staged = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR,
            tile_shape,
            cutlass.BFloat16,
            1,
        )
        q_layout = cute.slice_(q_layout_staged, (None, None, 0))
        k_layout = cute.slice_(k_layout_staged, (None, None, 0))

        # Four warp groups cover a 128x128 score tile: two along the
        # query/head dimension and two along the key dimension.
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32,
            (2, 2, 1),
            (64, self.BLOCK_N // 2),
        )
        async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
            ),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )
        tma_atom_k, tma_tensor_k = async_copy, k
        if cutlass.const_expr(self.use_tma_k_stage):
            tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                k,
                k_layout,
                (self.BLOCK_N, self.index_dim),
            )

        self.kernel(
            q,
            k,
            weights,
            starts,
            ends,
            output,
            q_layout,
            k_layout,
            tiled_mma,
            async_copy,
            tma_atom_k,
            tma_tensor_k,
        ).launch(
            grid=(cute.ceil_div(q.shape[0], self.QUERY_BLOCK), 1, 1),
            block=(self.THREADS, 1, 1),
        )

    @cute.jit
    def _ordered_uint32(self, value: cutlass.Float32):
        """Map float32 bits to uint32 while preserving numerical order."""
        bits = value.bitcast(cutlass.Uint32)
        ordered = bits | cutlass.Uint32(0x80000000)
        if value < 0:
            ordered = (~bits) & cutlass.Uint32(0xFFFFFFFF)
        return ordered

    @cute.jit
    def _stage_k_rows(
        self,
        k: cute.Tensor,
        index_k: cute.Tensor,
        first_k: cutlass.Int32,
        last_k: cutlass.Int32,
        block_offset: cutlass.Int32,
        row_start: cutlass.Int32,
        row_count: cutlass.Constexpr,
        stage_tidx: cutlass.Int32,
        stage_threads: cutlass.Constexpr,
        async_copy: cute.CopyAtom,
    ):
        vector_width = 8
        vectors_per_row = self.index_dim // vector_width
        for chunk in range((row_count * vectors_per_row) // stage_threads):
            flat_index = stage_tidx + chunk * stage_threads
            key_in_block = row_start + flat_index // vectors_per_row
            vector_idx = flat_index % vectors_per_row
            key_idx = first_k + block_offset + key_in_block
            k_destination = cute.local_tile(
                index_k[key_in_block, None],
                (vector_width,),
                (vector_idx,),
            )
            if key_idx < last_k:
                k_source = cute.local_tile(
                    k[key_idx, None],
                    (vector_width,),
                    (vector_idx,),
                )
                cute.copy(async_copy, k_source, k_destination)
            else:
                k_destination.fill(0)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _stage_paired_k_rows(
        self,
        k: cute.Tensor,
        index_k: cute.Tensor,
        first_k: cutlass.Int32,
        last_k: cutlass.Int32,
        block_offset: cutlass.Int32,
        warp_group: cutlass.Int32,
        tidx: cutlass.Int32,
        async_copy: cute.CopyAtom,
    ):
        """Let each warp-group pair stage the K half that it consumes."""
        if warp_group < 2:
            self._stage_k_rows(
                k,
                index_k,
                first_k,
                last_k,
                block_offset,
                cutlass.Int32(0),
                self.BLOCK_N // 2,
                tidx,
                self.THREADS // 2,
                async_copy,
            )
        else:
            self._stage_k_rows(
                k,
                index_k,
                first_k,
                last_k,
                block_offset,
                cutlass.Int32(self.BLOCK_N // 2),
                self.BLOCK_N // 2,
                tidx - self.THREADS // 2,
                self.THREADS // 2,
                async_copy,
            )

    @cute.jit
    def _select_topk(
        self,
        candidate_keys: cute.Tensor,
        selected_slots: cute.Tensor,
        histogram: cute.Tensor,
        warp_offsets: cute.Tensor,
        threshold_bin: cute.Tensor,
        output_counts: cute.Tensor,
        active_count: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        remaining = cute.make_rmem_tensor(
            cute.make_layout(self.QUERY_BLOCK),
            cutlass.Int32,
        )
        prefix = cute.make_rmem_tensor(
            cute.make_layout(self.QUERY_BLOCK),
            cutlass.Uint32,
        )
        scan_values = cute.make_rmem_tensor(
            cute.make_layout(self.QUERY_BLOCK),
            cutlass.Int32,
        )
        for query in cutlass.range_constexpr(self.QUERY_BLOCK):
            remaining[query] = self.topk
            prefix[query] = 0

        if cutlass.const_expr(self.topk == 1024):
            own_totals = cute.make_rmem_tensor(
                cute.make_layout(self.QUERY_BLOCK),
                cutlass.Int32,
            )
            # A wider radix removes one full candidate traversal. Its 32 KiB
            # histogram aliases K only in this static specialization.
            for radix11_pass in cutlass.range_constexpr(3):
                radix_shift = 21 - radix11_pass * 11
                radix_width = cutlass.Int32(11)
                radix_mask = cutlass.Uint32(0x7FF)
                radix_bins = cutlass.Int32(self.RADIX11_BINS)
                bins_per_thread = cutlass.Int32(4)
                if radix11_pass == 2:
                    radix_shift = 0
                    radix_width = 10
                    radix_mask = cutlass.Uint32(0x3FF)
                    radix_bins = 1024
                    bins_per_thread = 2
                for chunk in range((self.QUERY_BLOCK * self.RADIX11_BINS + self.THREADS - 1) // self.THREADS):
                    index = tidx + chunk * self.THREADS
                    if index < self.QUERY_BLOCK * radix_bins:
                        histogram[
                            index // radix_bins,
                            index % radix_bins,
                        ] = 0
                if tidx < self.QUERY_BLOCK:
                    threshold_bin[tidx] = 0
                cute.arch.sync_threads()

                for chunk in range((self.candidate_count + self.THREADS - 1) // self.THREADS):
                    index = tidx + chunk * self.THREADS
                    if index < active_count:
                        for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                            ordered = candidate_keys[query, index]
                            match = radix11_pass == 0
                            if radix11_pass > 0:
                                match = ordered >> (32 - radix11_pass * 11) == prefix[query]
                            if match:
                                bin_id = cutlass.Int32((ordered >> radix_shift) & radix_mask)
                                cute.arch.atomic_add(
                                    (histogram.iterator + query * self.RADIX11_BINS + bin_id).llvm_ptr,
                                    cutlass.Int32(1),
                                )
                cute.arch.sync_threads()

                # Each thread owns four adjacent bins. A warp suffix-scan
                # followed by one 16-warp exchange finds the exact bin.
                radix11_lane_idx = tidx % 32
                radix11_warp_idx = tidx // 32
                bin_base = tidx * bins_per_thread
                for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                    own_totals[query] = histogram[query, bin_base] + histogram[query, bin_base + 1]
                    if radix11_pass < 2:
                        own_totals[query] += histogram[query, bin_base + 2] + histogram[query, bin_base + 3]
                    scan_values[query] = own_totals[query]
                for scan_step in cutlass.range_constexpr(5):
                    radix11_offset = 1 << scan_step
                    for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                        radix11_neighbor = cute.arch.shuffle_sync_down(
                            scan_values[query],
                            offset=radix11_offset,
                        )
                        if radix11_lane_idx < 32 - radix11_offset:
                            scan_values[query] += radix11_neighbor

                if radix11_lane_idx == 0:
                    for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                        warp_offsets[query, radix11_warp_idx] = scan_values[query]
                cute.arch.sync_threads()

                if radix11_warp_idx == 0:
                    for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                        radix11_warp_total = cutlass.Int32(0)
                        radix11_own_total = cutlass.Int32(0)
                        if radix11_lane_idx < 16:
                            radix11_own_total = warp_offsets[query, radix11_lane_idx]
                            radix11_warp_total = radix11_own_total
                        for scan_step in cutlass.range_constexpr(4):
                            radix11_offset = 1 << scan_step
                            radix11_neighbor = cute.arch.shuffle_sync_down(
                                radix11_warp_total,
                                offset=radix11_offset,
                            )
                            if radix11_lane_idx < 16 - radix11_offset:
                                radix11_warp_total += radix11_neighbor
                        if radix11_lane_idx < 16:
                            warp_offsets[query, radix11_lane_idx] = radix11_warp_total - radix11_own_total
                cute.arch.sync_threads()

                for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                    suffix = scan_values[query] - own_totals[query] + warp_offsets[query, radix11_warp_idx]
                    for local_bin in cutlass.range_constexpr(3, -1, -1):
                        if local_bin < bins_per_thread:
                            radix11_next_value = suffix
                            suffix += histogram[query, bin_base + local_bin]
                            histogram[query, bin_base + local_bin] = suffix
                            if suffix >= remaining[query] and radix11_next_value < remaining[query]:
                                threshold_bin[query] = bin_base + local_bin
                cute.arch.sync_threads()

                for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                    radix11_threshold = threshold_bin[query]
                    if radix11_threshold + 1 < radix_bins:
                        remaining[query] -= histogram[query, radix11_threshold + 1]
                    prefix[query] = (prefix[query] << radix_width) | cutlass.Uint32(radix11_threshold)
                cute.arch.sync_threads()
        else:
            # The compact byte radix keeps topk=2048 at four merge passes for
            # K=32K; the wider histogram would force a fifth merge.
            for radix_pass in cutlass.range_constexpr(4):
                for chunk in range((self.QUERY_BLOCK * self.RADIX8_BINS + self.THREADS - 1) // self.THREADS):
                    index = tidx + chunk * self.THREADS
                    if index < self.QUERY_BLOCK * self.RADIX8_BINS:
                        histogram[
                            index // self.RADIX8_BINS,
                            index % self.RADIX8_BINS,
                        ] = 0
                if tidx < self.QUERY_BLOCK:
                    threshold_bin[tidx] = 0
                cute.arch.sync_threads()

                for chunk in range((self.candidate_count + self.THREADS - 1) // self.THREADS):
                    index = tidx + chunk * self.THREADS
                    if index < active_count:
                        for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                            ordered = candidate_keys[query, index]
                            match = radix_pass == 0
                            if radix_pass > 0:
                                match = ordered >> (32 - radix_pass * 8) == prefix[query]
                            if match:
                                bin_id = cutlass.Int32((ordered >> (24 - radix_pass * 8)) & cutlass.Uint32(0xFF))
                                cute.arch.atomic_add(
                                    (histogram.iterator + query * self.RADIX8_BINS + bin_id).llvm_ptr,
                                    cutlass.Int32(1),
                                )
                cute.arch.sync_threads()

                if tidx < 256:
                    lane_idx = tidx % 32
                    warp_idx = tidx // 32
                    for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                        scan_values[query] = histogram[query, tidx]
                    for scan_step in cutlass.range_constexpr(5):
                        offset = 1 << scan_step
                        for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                            neighbor = cute.arch.shuffle_sync_down(
                                scan_values[query],
                                offset=offset,
                            )
                            if lane_idx < 32 - offset:
                                scan_values[query] += neighbor

                    if lane_idx == 0:
                        for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                            warp_offsets[query, warp_idx] = scan_values[query]
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)

                    if warp_idx == 0:
                        for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                            warp_total = cutlass.Int32(0)
                            own_total = cutlass.Int32(0)
                            if lane_idx < 8:
                                own_total = warp_offsets[query, lane_idx]
                                warp_total = own_total
                            for scan_step in cutlass.range_constexpr(3):
                                offset = 1 << scan_step
                                neighbor = cute.arch.shuffle_sync_down(
                                    warp_total,
                                    offset=offset,
                                )
                                if lane_idx < 8 - offset:
                                    warp_total += neighbor
                            if lane_idx < 8:
                                warp_offsets[query, lane_idx] = warp_total - own_total
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)

                    for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                        warp_offset = warp_offsets[query, warp_idx]
                        scan_values[query] += warp_offset
                        histogram[query, tidx] = scan_values[query]
                        next_value = cute.arch.shuffle_sync_down(
                            scan_values[query],
                            offset=1,
                        )
                        if lane_idx == 31:
                            next_value = warp_offset
                        if scan_values[query] >= remaining[query] and next_value < remaining[query]:
                            threshold_bin[query] = tidx
                cute.arch.sync_threads()

                for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                    threshold = threshold_bin[query]
                    remaining[query] -= histogram[query, threshold + 1]
                    prefix[query] = (prefix[query] << 8) | cutlass.Uint32(threshold)
                cute.arch.sync_threads()

        for chunk in range((self.QUERY_BLOCK * self.topk + self.THREADS - 1) // self.THREADS):
            index = tidx + chunk * self.THREADS
            if index < self.QUERY_BLOCK * self.topk:
                selected_slots[
                    index // self.topk,
                    index % self.topk,
                ] = cutlass.Uint16(0)
        if tidx < self.QUERY_BLOCK * 2:
            output_counts[tidx // 2, tidx % 2] = 0
        cute.arch.sync_threads()

        for chunk in range((self.candidate_count + self.THREADS - 1) // self.THREADS):
            index = tidx + chunk * self.THREADS
            if index < active_count:
                for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                    ordered = candidate_keys[query, index]
                    if ordered > prefix[query]:
                        output_pos = cute.arch.atomic_add(
                            (output_counts.iterator + query * 2).llvm_ptr,
                            cutlass.Int32(1),
                        )
                        if output_pos < self.topk - remaining[query]:
                            selected_slots[query, output_pos] = cutlass.Uint16(index)
                    elif ordered == prefix[query]:
                        output_pos = cute.arch.atomic_add(
                            (output_counts.iterator + query * 2 + 1).llvm_ptr,
                            cutlass.Int32(1),
                        )
                        if output_pos < remaining[query]:
                            selected_slots[
                                query,
                                self.topk - remaining[query] + output_pos,
                            ] = cutlass.Uint16(index)
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        weights: cute.Tensor,
        starts_input: cute.Tensor,
        ends_input: cute.Tensor,
        output: cute.Tensor,
        q_layout: cute.ComposedLayout,
        k_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        async_copy: cute.CopyAtom,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = tidx // 32
        if cutlass.const_expr(self.topk == 1024 and self.use_candidate_compaction):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tma_warp_idx = warp_idx
        if cutlass.const_expr(self.use_tma_k_stage and not (self.topk == 1024 and self.use_candidate_compaction)):
            tma_warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()
        query_offset = block_idx * self.QUERY_BLOCK
        matrix_m = self.QUERY_BLOCK * self.heads
        # The multi-merge topk=1024 specialization packs int32 key IDs into
        # the first half of each query's int64 output region. The gather
        # barrier makes the final in-place expansion to int64 safe.
        intermediate_output = cute.make_tensor(
            cute.recast_ptr(output.iterator, dtype=cutlass.Int32),
            cute.make_layout(output.shape[0] * self.topk * 2),
        )

        @cute.struct
        class QueryKeyStorage:
            query: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(q_layout),
                ],
                1024,
            ]
            key: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(k_layout),
                ],
                1024,
            ]

        @cute.struct
        class Radix8MergeStorage:
            selected_slots: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint16,
                    self.QUERY_BLOCK * self.topk,
                ],
                16,
            ]
            histogram: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    self.QUERY_BLOCK * self.RADIX8_BINS,
                ],
                16,
            ]
            warp_offsets: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    self.QUERY_BLOCK * 8,
                ],
                16,
            ]
            threshold_bin: cute.struct.MemRange[
                cutlass.Int32,
                self.QUERY_BLOCK,
            ]
            output_counts: cute.struct.MemRange[
                cutlass.Int32,
                self.QUERY_BLOCK * 2,
            ]

        @cute.struct
        class Radix11HistogramStorage:
            histogram: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    self.QUERY_BLOCK * self.RADIX11_BINS,
                ],
                16,
            ]

        allocator = cutlass.utils.SmemAllocator()
        # @cute.struct generates these members during DSL lowering, so they
        # are intentionally absent from the Python type stubs.
        query_key_raw = allocator.allocate(
            QueryKeyStorage.size_in_bytes(),  # type: ignore[attr-defined]
            byte_alignment=1024,
        )
        query_key_storage = QueryKeyStorage(query_key_raw)  # type: ignore[call-arg]
        index_q = query_key_storage.query.get_tensor(
            q_layout.outer,
            swizzle=q_layout.inner,
        )
        index_k = query_key_storage.key.get_tensor(
            k_layout.outer,
            swizzle=k_layout.inner,
        )
        if cutlass.const_expr(self.use_tma_k_stage):
            tma_barriers = allocator.allocate_tensor(
                cutlass.Int64,
                cute.make_layout(2),
                byte_alignment=8,
            )
        # WGMMA accumulator values are reduced in registers. Shared memory
        # holds only one 16-head partial per warp instead of the full
        # 128x128 float32 score tile.
        partial_scores = allocator.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (self.THREADS // 32, self.BLOCK_N // 2),
                stride=(self.BLOCK_N // 2, 1),
            ),
            byte_alignment=16,
        )
        # Radix selection only needs numerical order, so candidates stay in
        # this bijective uint32 representation across every merge.
        candidate_keys = allocator.allocate_tensor(
            cutlass.Uint32,
            cute.make_layout(
                (self.QUERY_BLOCK, self.candidate_count),
                stride=(self.candidate_count, 1),
            ),
            byte_alignment=16,
        )
        if cutlass.const_expr(self.use_candidate_compaction):
            candidate_indices = allocator.allocate_tensor(
                self.candidate_index_dtype,
                cute.make_layout(
                    (self.QUERY_BLOCK, self.candidate_count),
                    stride=(self.candidate_count, 1),
                ),
                byte_alignment=16,
            )
        else:
            # CuTe DSL type-checks statically dead compound conditions. This
            # zero-allocation alias gives that dead path a valid tensor type.
            candidate_indices = cute.make_tensor(
                cute.recast_ptr(
                    candidate_keys.iterator,
                    dtype=self.candidate_index_dtype,
                ),
                cute.make_layout(
                    (self.QUERY_BLOCK, self.candidate_count),
                    stride=(self.candidate_count, 1),
                ),
            )
        if cutlass.const_expr(self.topk == 2048):
            # The compact radix scratch fits entirely inside the staged K
            # range and preserves the original candidate capacity.
            assert Radix8MergeStorage.size_in_bytes() <= cute.cosize(k_layout) * 2  # type: ignore[attr-defined]
            merge_storage = Radix8MergeStorage(  # type: ignore[call-arg]
                cute.recast_ptr(
                    query_key_storage.key.data_ptr(),
                    dtype=cutlass.Uint8,
                )
            )
            selected_slots = merge_storage.selected_slots.get_tensor(
                cute.make_layout(
                    (self.QUERY_BLOCK, self.topk),
                    stride=(self.topk, 1),
                ),
            )
            histogram = merge_storage.histogram.get_tensor(
                cute.make_layout(
                    (self.QUERY_BLOCK, self.RADIX8_BINS),
                    stride=(self.RADIX8_BINS, 1),
                ),
            )
            warp_offsets = merge_storage.warp_offsets.get_tensor(
                cute.make_layout(
                    (self.QUERY_BLOCK, 8),
                    stride=(8, 1),
                ),
            )
            threshold_bin = merge_storage.threshold_bin.get_tensor(
                cute.make_layout(self.QUERY_BLOCK),
            )
            output_counts = merge_storage.output_counts.get_tensor(
                cute.make_layout(
                    (self.QUERY_BLOCK, 2),
                    stride=(2, 1),
                ),
            )
        else:
            # The 2,048 Int32 bins exactly fill the 32 KiB K stage. No K tile
            # is live during selection, so the two roles can safely alias.
            assert Radix11HistogramStorage.size_in_bytes() == cute.cosize(k_layout) * 2  # type: ignore[attr-defined]
            histogram_storage = Radix11HistogramStorage(  # type: ignore[call-arg]
                cute.recast_ptr(
                    query_key_storage.key.data_ptr(),
                    dtype=cutlass.Uint8,
                )
            )
            histogram = histogram_storage.histogram.get_tensor(
                cute.make_layout(
                    (self.QUERY_BLOCK, self.RADIX11_BINS),
                    stride=(self.RADIX11_BINS, 1),
                ),
            )
            selected_slots = allocator.allocate_tensor(
                cutlass.Uint16,
                cute.make_layout(
                    (self.QUERY_BLOCK, self.topk),
                    stride=(self.topk, 1),
                ),
                byte_alignment=16,
            )
            warp_offsets = allocator.allocate_tensor(
                cutlass.Int32,
                cute.make_layout(
                    (self.QUERY_BLOCK, self.THREADS // 32),
                    stride=(self.THREADS // 32, 1),
                ),
                byte_alignment=16,
            )
            threshold_bin = allocator.allocate_tensor(
                cutlass.Int32,
                cute.make_layout(self.QUERY_BLOCK),
            )
            output_counts = allocator.allocate_tensor(
                cutlass.Int32,
                cute.make_layout(
                    (self.QUERY_BLOCK, 2),
                    stride=(2, 1),
                ),
            )

        candidate_counts = cute.make_tensor(
            output_counts.iterator,
            cute.make_layout(self.QUERY_BLOCK, stride=2),
        )
        score_threshold = cute.make_tensor(
            candidate_keys.iterator,
            cute.make_layout(self.QUERY_BLOCK),
        )
        if cutlass.const_expr(self.use_candidate_compaction):
            if cutlass.const_expr(self.topk == 2048):
                # The compact radix path overlays output_counts with staged K,
                # so its online counters need independent persistent storage.
                candidate_counts = allocator.allocate_tensor(
                    cutlass.Int32,
                    cute.make_layout(self.QUERY_BLOCK),
                )
            score_threshold = allocator.allocate_tensor(
                cutlass.Uint32,
                cute.make_layout(self.QUERY_BLOCK),
            )
        candidate_count_iterator = output_counts.iterator
        candidate_count_stride = 2
        if cutlass.const_expr(self.topk == 2048):
            candidate_count_iterator = candidate_counts.iterator
            candidate_count_stride = 1

        if cutlass.const_expr(self.use_persistent_shared_ids):
            retained_id_scratch = allocator.allocate_tensor(
                cutlass.Uint16,
                cute.make_layout(
                    (self.QUERY_BLOCK, self.topk),
                    stride=(self.topk, 1),
                ),
                byte_alignment=16,
            )
        elif cutlass.const_expr(self.use_overlay_shared_ids):
            retained_id_scratch = cute.make_tensor(
                cute.recast_ptr(
                    query_key_storage.key.data_ptr(),
                    dtype=cutlass.Uint16,
                ),
                cute.make_layout(
                    (self.QUERY_BLOCK, self.topk),
                    stride=(self.topk, 1),
                ),
            )

        if cutlass.const_expr(self.use_tma_k_stage):
            tma_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=tma_barriers.iterator,
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.THREADS // 32,
                ),
                tx_count=cute.size_in_bytes(cutlass.BFloat16, k_layout),
            )
            tma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                1,
            )
            tma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                1,
            )

        # Dynamic indexing into per-thread range arrays spills them to local
        # memory, so the four CTA-wide ranges live in shared memory instead.
        starts = allocator.allocate_tensor(
            cutlass.Int32,
            cute.make_layout(self.QUERY_BLOCK),
        )
        ends = allocator.allocate_tensor(
            cutlass.Int32,
            cute.make_layout(self.QUERY_BLOCK),
        )

        range_query = cutlass.Int32(0)
        if tidx < self.QUERY_BLOCK:
            range_query = tidx
            if query_offset + range_query < q.shape[0]:
                starts[range_query] = starts_input[query_offset + range_query]
                ends[range_query] = ends_input[query_offset + range_query]
            else:
                starts[range_query] = 0
                ends[range_query] = 0

        for chunk in range((self.QUERY_BLOCK * self.topk + self.THREADS - 1) // self.THREADS):
            index = tidx + chunk * self.THREADS
            if index < self.QUERY_BLOCK * self.topk:
                candidate_keys[
                    index // self.topk,
                    index % self.topk,
                ] = cutlass.Uint32(self.ORDERED_NEG_INF)
        output_query = cutlass.Int32(0)
        for chunk in range((self.QUERY_BLOCK * self.topk + self.THREADS - 1) // self.THREADS):
            index = tidx + chunk * self.THREADS
            output_query = index // self.topk
            topk_idx = index % self.topk
            if index < self.QUERY_BLOCK * self.topk and query_offset + output_query < q.shape[0]:
                output[query_offset + output_query, 0, topk_idx] = -1
        if cutlass.const_expr(self.use_candidate_compaction):
            if tidx < self.QUERY_BLOCK:
                if cutlass.const_expr(self.topk == 2048):
                    candidate_counts[tidx] = 0
                else:
                    output_counts[tidx, 0] = 0
                score_threshold[tidx] = cutlass.Uint32(0xFFFFFFFF)
        cute.arch.sync_threads()

        first_k = cutlass.Int32(k.shape[0])
        last_k = cutlass.Int32(0)
        for query in cutlass.range_constexpr(self.QUERY_BLOCK):
            if query_offset + query < q.shape[0]:
                if starts[query] < first_k:
                    first_k = starts[query]
                if ends[query] > last_k:
                    last_k = ends[query]
        if cutlass.const_expr(self.use_tma_k_stage):
            offset_tma_k = cute.domain_offset(
                (first_k, cutlass.Int32(0)),
                tma_tensor_k,
            )
            tiled_tma_k = cute.local_tile(
                offset_tma_k,
                (self.BLOCK_N, self.index_dim),
                (None, None),
            )
            tma_smem_k, tma_gmem_k = cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(index_k, 0, 2),
                cute.group_modes(tiled_tma_k, 0, 2),
            )
        candidate_id_base = cutlass.Int32(0)
        if cutlass.const_expr(self.use_compact_relative_ids):
            candidate_id_base = first_k

        # Every thread materializes one score for one query in each K tile.
        # Hoisting its range out of the K loop avoids two dynamic array loads
        # for every score.
        score_query = tidx // self.BLOCK_N
        score_start = starts[score_query]
        score_end = ends[score_query]

        key_count = last_k - first_k
        first_batch_size = cutlass.Int32(self.batch_size)
        if key_count > self.topk:
            first_batch_size = self.candidate_count

        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        warp_group_layout = cute.make_layout(4, stride=128)
        thr_mma = tiled_mma.get_slice(warp_group_layout(warp_group_idx))
        q_fragment = tiled_mma.make_fragment_A(thr_mma.partition_A(index_q))
        k_fragment = tiled_mma.make_fragment_B(thr_mma.partition_B(index_k))
        accumulator = thr_mma.make_fragment_C(thr_mma.partition_shape_C((matrix_m, self.BLOCK_N)))

        retained_keys = cute.make_rmem_tensor(
            cute.make_layout(self.retained_per_thread),
            cutlass.Uint32,
        )
        retained_indices = cute.make_rmem_tensor(
            cute.make_layout(self.retained_per_thread),
            cutlass.Int32,
        )
        if cutlass.const_expr(self.use_overlay_shared_ids):
            retained_id_pairs = cute.make_rmem_tensor(
                cute.make_layout(self.retained_per_thread // 2),
                cutlass.Uint32,
            )

        score = cutlass.Float32(0.0)
        # Carry the destination tile across the K loop. Recomputing it from
        # block_offset used a dynamic modulo in every warp for every tile.
        candidate_tile_offset = cutlass.Int32(self.topk)
        if key_count > self.topk:
            candidate_tile_offset = 0
        processed = cutlass.Int32(0)
        batch_count = cutlass.Int32(0)
        active_count = cutlass.Int32(0)
        selected_key_idx = cutlass.Int32(-1)
        lane_idx = tidx % 32
        warp_group = warp_idx // 4
        warp_in_group = warp_idx % 4
        if cutlass.const_expr(self.use_warpgroup_score_owners):
            # Each warp group owns two queries for one 64-key half. Mapping
            # the final score to that owner avoids a CTA-wide barrier.
            score_thread = tidx % 128
            score_query = (warp_group % 2) * 2 + score_thread // 64
            score_key_in_group = score_thread % 64
            score_start = starts[score_query]
            score_end = ends[score_query]
        query_for_warp = (warp_group % 2) * 2 + warp_in_group // 2
        head_group = lane_idx // 4
        head_base = (warp_in_group % 2) * 16 + head_group
        weight_0 = cutlass.Float32(0.0)
        weight_1 = cutlass.Float32(0.0)
        if query_offset + query_for_warp < q.shape[0]:
            weight_0 = weights[
                query_offset + query_for_warp,
                head_base,
            ]
            weight_1 = weights[
                query_offset + query_for_warp,
                head_base + 8,
            ]

        # Q is invariant across all K blocks and remains resident in shared
        # memory for the complete score/top-k loop.
        vector_width = 8
        vectors_per_row = self.index_dim // vector_width
        for chunk in range((matrix_m * vectors_per_row) // self.THREADS):
            flat_index = tidx + chunk * self.THREADS
            query_head = flat_index // vectors_per_row
            vector_idx = flat_index % vectors_per_row
            query = query_head // self.heads
            q_destination = cute.local_tile(
                index_q[query_head, None],
                (vector_width,),
                (vector_idx,),
            )
            if query_offset + query < q.shape[0]:
                q_source = cute.local_tile(
                    q[
                        query_offset + query,
                        query_head % self.heads,
                        None,
                    ],
                    (vector_width,),
                    (vector_idx,),
                )
                cute.autovec_copy(q_source, q_destination)
            else:
                q_destination.fill(0)
        if cutlass.const_expr(self.use_tma_k_stage):
            cute.arch.sync_threads()

        block_count = cute.ceil_div(key_count, self.BLOCK_N)
        if block_count > 0:
            if cutlass.const_expr(self.use_tma_k_stage):
                if tma_warp_idx == 0:
                    tma_pipeline.producer_acquire(tma_producer_state)
                    tma_barrier = tma_pipeline.producer_get_barrier(tma_producer_state)
                    cute.copy(
                        tma_atom_k,
                        tma_gmem_k[None, 0, 0],
                        tma_smem_k,
                        tma_bar_ptr=tma_barrier,
                    )
                    tma_producer_state.advance()
            elif cutlass.const_expr(self.use_pair_scoped_k_ready):
                self._stage_paired_k_rows(
                    k,
                    index_k,
                    first_k,
                    last_k,
                    cutlass.Int32(0),
                    warp_group,
                    tidx,
                    async_copy,
                )
            else:
                self._stage_k_rows(
                    k,
                    index_k,
                    first_k,
                    last_k,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    self.BLOCK_N,
                    tidx,
                    self.THREADS,
                    async_copy,
                )

        for key_block in range(block_count):
            block_offset = key_block * self.BLOCK_N

            if cutlass.const_expr(self.use_tma_k_stage):
                tma_pipeline.consumer_wait(tma_consumer_state)
            else:
                cute.arch.cp_async_wait_group(0)
                if cutlass.const_expr(self.use_pair_scoped_k_ready):
                    cute.arch.barrier(
                        barrier_id=2 + warp_group // 2,
                        number_of_threads=self.THREADS // 2,
                    )
                else:
                    cute.arch.sync_threads()

            compact_merge_due = processed < 0
            if cutlass.const_expr(self.use_candidate_compaction):
                # A query appends at most BLOCK_N candidates per score tile.
                # Skip checks until the capacity reserve can be reached.
                if processed >= first_batch_size and (
                    cutlass.const_expr(self.topk == 1024)
                    or (
                        cutlass.const_expr(self.use_compact_relative_ids)
                        and candidate_tile_offset > self.candidate_count - 2 * self.BLOCK_N
                    )
                    or (
                        cutlass.const_expr(self.use_sparse_candidate_checks)
                        and candidate_tile_offset > self.candidate_count - 3 * self.BLOCK_N
                        and (key_block & 1) == 0
                    )
                ):
                    # Warp groups append independently. Join only on tiles
                    # that inspect counters so every thread takes the same
                    # exact-merge branch.
                    if cutlass.const_expr(self.use_warpgroup_score_owners):
                        cute.arch.sync_threads()
                    if cutlass.const_expr(self.topk == 2048):
                        compact_count = candidate_counts[0]
                    else:
                        compact_count = output_counts[0, 0]
                    for query in cutlass.range_constexpr(1, self.QUERY_BLOCK):
                        if cutlass.const_expr(self.topk == 2048):
                            query_compact_count = candidate_counts[query]
                        else:
                            query_compact_count = output_counts[query, 0]
                        if query_compact_count > compact_count:
                            compact_count = query_compact_count
                    if cutlass.const_expr(self.use_sparse_candidate_checks):
                        compact_merge_due = compact_count > self.batch_size - 3 * self.BLOCK_N
                    else:
                        compact_merge_due = compact_count > self.batch_size - 2 * self.BLOCK_N

            tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()
            for k_block in cutlass.range_constexpr(cute.size(q_fragment, mode=[2])):
                cute.gemm(
                    tiled_mma,
                    accumulator,
                    q_fragment[None, None, k_block],
                    k_fragment[None, None, k_block],
                    accumulator,
                )
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            if cutlass.const_expr(self.use_tma_k_stage):
                tma_pipeline.consumer_release(tma_consumer_state)
                tma_consumer_state.advance()

            prefetch_next = key_block + 1 < block_count
            if cutlass.const_expr(self.merge_reuses_k):
                next_processed = block_offset + self.BLOCK_N
                if next_processed > key_count:
                    next_processed = key_count
                if cutlass.const_expr(self.use_candidate_compaction):
                    merge_before_next_tile = (
                        next_processed == first_batch_size or next_processed == key_count or compact_merge_due
                    )
                else:
                    merge_before_next_tile = (
                        next_processed == first_batch_size
                        or next_processed == key_count
                        or (
                            next_processed > first_batch_size
                            and (next_processed - first_batch_size) % self.batch_size == 0
                        )
                    )
                prefetch_next = prefetch_next and not merge_before_next_tile

            if prefetch_next:
                if cutlass.const_expr(self.use_tma_k_stage):
                    if tma_warp_idx == 0:
                        tma_pipeline.producer_acquire(tma_producer_state)
                        tma_barrier = tma_pipeline.producer_get_barrier(tma_producer_state)
                        cute.copy(
                            tma_atom_k,
                            tma_gmem_k[None, key_block + 1, 0],
                            tma_smem_k,
                            tma_bar_ptr=tma_barrier,
                        )
                        tma_producer_state.advance()
                elif cutlass.const_expr(self.use_paired_k_stage):
                    # WG0/1 consume the first K half and WG2/3 the second.
                    # Each pair only overwrites the half it has finished.
                    if warp_group < 2:
                        cute.arch.barrier(
                            barrier_id=2,
                            number_of_threads=self.THREADS // 2,
                        )
                        self._stage_k_rows(
                            k,
                            index_k,
                            first_k,
                            last_k,
                            block_offset + self.BLOCK_N,
                            cutlass.Int32(0),
                            self.BLOCK_N // 2,
                            tidx,
                            self.THREADS // 2,
                            async_copy,
                        )
                    else:
                        cute.arch.barrier(
                            barrier_id=3,
                            number_of_threads=self.THREADS // 2,
                        )
                        self._stage_k_rows(
                            k,
                            index_k,
                            first_k,
                            last_k,
                            block_offset + self.BLOCK_N,
                            cutlass.Int32(self.BLOCK_N // 2),
                            self.BLOCK_N // 2,
                            tidx - self.THREADS // 2,
                            self.THREADS // 2,
                            async_copy,
                        )
                else:
                    # WGMMA completion is warp-group scoped. All warp groups
                    # must finish before the complete stage can be reused.
                    cute.arch.sync_threads()
                    self._stage_k_rows(
                        k,
                        index_k,
                        first_k,
                        last_k,
                        block_offset + self.BLOCK_N,
                        cutlass.Int32(0),
                        self.BLOCK_N,
                        tidx,
                        self.THREADS,
                        async_copy,
                    )

            # The WGMMA accumulator gives each thread two heads for 16
            # key pairs. Three shuffles reduce the eight lanes that jointly
            # own one 16-head partial.
            for key_group in cutlass.range_constexpr(8):
                for key_pair in cutlass.range_constexpr(2):
                    accumulator_idx = key_pair + key_group * 4
                    partial = (
                        cute.arch.fmax(
                            accumulator[accumulator_idx],
                            cutlass.Float32(0.0),
                        )
                        * weight_0
                        + cute.arch.fmax(
                            accumulator[accumulator_idx + 2],
                            cutlass.Float32(0.0),
                        )
                        * weight_1
                    )
                    partial += cute.arch.shuffle_sync_bfly(partial, offset=4)
                    partial += cute.arch.shuffle_sync_bfly(partial, offset=8)
                    partial += cute.arch.shuffle_sync_bfly(partial, offset=16)
                    if head_group == 0:
                        key_in_group = (lane_idx % 4) * 2 + key_pair + key_group * 8
                        partial_scores[warp_idx, key_in_group] = partial
            if cutlass.const_expr(self.use_warpgroup_score_owners):
                cute.arch.barrier(
                    barrier_id=4 + warp_group,
                    number_of_threads=self.THREADS // 4,
                )
            else:
                cute.arch.sync_threads()

            # Each score combines two shared 16-head halves. The paired
            # topk=2048 path keeps its consumer in the producing warp group.
            for output_slot in cutlass.range_constexpr(
                (self.QUERY_BLOCK * self.BLOCK_N + self.THREADS - 1) // self.THREADS
            ):
                if cutlass.const_expr(True):
                    if cutlass.const_expr(self.use_warpgroup_score_owners):
                        query = score_query
                        key_in_block = (warp_group // 2) * 64 + score_key_in_group
                        warp_base = warp_group * 4 + (query % 2) * 2
                        score = (
                            partial_scores[warp_base, score_key_in_group]
                            + partial_scores[warp_base + 1, score_key_in_group]
                        )
                    else:
                        score_flat_index = tidx + output_slot * self.THREADS
                        query = score_flat_index // self.BLOCK_N
                        key_in_block = score_flat_index % self.BLOCK_N
                        key_group = key_in_block // 64
                        key_in_group = key_in_block % 64
                        query_group = query // 2
                        score_warp_group = query_group + key_group * 2
                        warp_base = score_warp_group * 4 + (query % 2) * 2
                        score = partial_scores[warp_base, key_in_group] + partial_scores[warp_base + 1, key_in_group]
                    local_key = block_offset + key_in_block
                    key_idx = first_k + local_key
                    candidate_offset = candidate_tile_offset + key_in_block
                    if local_key < key_count:
                        ordered = cutlass.Uint32(self.ORDERED_NEG_INF)
                        if query_offset + query < q.shape[0] and key_idx >= score_start and key_idx < score_end:
                            ordered = self._ordered_uint32(score)
                        if cutlass.const_expr(self.use_candidate_compaction) and processed >= first_batch_size:
                            if (
                                cutlass.const_expr(self.use_warp_aggregated_append)
                                and block_offset + self.BLOCK_N <= key_count
                            ):
                                # Full warps reserve one contiguous range.
                                # atomic_add(0) keeps the leader path uniform
                                # and is faster than branching on an empty mask.
                                accepted = ordered > score_threshold[query]
                                accepted_mask = cute.arch.vote_ballot_sync(accepted)
                                accepted_count = cute.arch.popc(accepted_mask)
                                warp_candidate_base = cutlass.Int32(0)
                                if lane_idx == 0:
                                    warp_candidate_base = cute.arch.atomic_add(
                                        (candidate_count_iterator + query * candidate_count_stride).llvm_ptr,
                                        accepted_count,
                                    )
                                warp_candidate_base = cute.arch.shuffle_sync(
                                    warp_candidate_base,
                                    offset=0,
                                )
                                lane_rank = cute.arch.popc(accepted_mask & ((cutlass.Uint32(1) << lane_idx) - 1))
                                compact_slot = warp_candidate_base + lane_rank
                                if accepted:
                                    if compact_slot < self.batch_size:
                                        candidate_keys[
                                            query,
                                            self.topk + compact_slot,
                                        ] = ordered
                                        candidate_indices[
                                            query,
                                            self.topk + compact_slot,
                                        ] = self.candidate_index_dtype(key_idx - candidate_id_base)
                            else:
                                if ordered > score_threshold[query]:
                                    compact_slot = cute.arch.atomic_add(
                                        (candidate_count_iterator + query * candidate_count_stride).llvm_ptr,
                                        cutlass.Int32(1),
                                    )
                                    if compact_slot < self.batch_size:
                                        candidate_keys[
                                            query,
                                            self.topk + compact_slot,
                                        ] = ordered
                                        candidate_indices[
                                            query,
                                            self.topk + compact_slot,
                                        ] = self.candidate_index_dtype(key_idx - candidate_id_base)
                        else:
                            candidate_keys[query, candidate_offset] = ordered
                            if cutlass.const_expr(self.store_initial_candidate_ids):
                                candidate_indices[query, candidate_offset] = self.candidate_index_dtype(
                                    key_idx - candidate_id_base
                                )
            candidate_tile_offset += self.BLOCK_N
            if cutlass.const_expr(not self.merge_reuses_k):
                cute.arch.sync_threads()

            processed = block_offset + self.BLOCK_N
            if processed > key_count:
                processed = key_count
            if cutlass.const_expr(self.use_candidate_compaction):
                merge_candidates = processed == first_batch_size or processed == key_count or compact_merge_due
            else:
                merge_candidates = (
                    processed == first_batch_size
                    or processed == key_count
                    or (processed > first_batch_size and (processed - first_batch_size) % self.batch_size == 0)
                )
            if merge_candidates:
                # Non-merge tiles are synchronized by the next iteration's
                # cp.async wait barrier in the K-overlay specialization.
                if cutlass.const_expr(self.merge_reuses_k):
                    cute.arch.sync_threads()
                batch_count = processed
                active_count = processed
                if processed <= first_batch_size:
                    if key_count <= self.topk:
                        active_count = self.topk + processed
                elif cutlass.const_expr(self.use_candidate_compaction):
                    if cutlass.const_expr(self.topk == 2048):
                        batch_count = candidate_counts[0]
                    else:
                        batch_count = output_counts[0, 0]
                    for query in cutlass.range_constexpr(1, self.QUERY_BLOCK):
                        if cutlass.const_expr(self.topk == 2048):
                            query_batch_count = candidate_counts[query]
                        else:
                            query_batch_count = output_counts[query, 0]
                        if query_batch_count > batch_count:
                            batch_count = query_batch_count
                    active_count = self.topk + batch_count
                    for chunk in range((self.batch_size + self.THREADS - 1) // self.THREADS):
                        compact_slot = tidx + chunk * self.THREADS
                        if compact_slot < batch_count:
                            for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                                if cutlass.const_expr(self.topk == 2048):
                                    query_candidate_count = candidate_counts[query]
                                else:
                                    query_candidate_count = output_counts[query, 0]
                                if compact_slot >= query_candidate_count:
                                    candidate_keys[
                                        query,
                                        self.topk + compact_slot,
                                    ] = cutlass.Uint32(self.ORDERED_NEG_INF)
                                    candidate_indices[
                                        query,
                                        self.topk + compact_slot,
                                    ] = self.candidate_index_dtype(-1)
                    cute.arch.sync_threads()
                else:
                    batch_count = (processed - first_batch_size) % self.batch_size
                    if batch_count == 0:
                        batch_count = self.batch_size
                    active_count = self.topk + batch_count

                if cutlass.const_expr(self.use_overlay_shared_ids):
                    if processed > first_batch_size:
                        # Selection overwrites selected_slots. Carry two
                        # encoded IDs per register across it, then park them
                        # in the dead K/histogram stage for random gather.
                        for retained_pair in cutlass.range_constexpr(self.retained_per_thread // 2):
                            first_flat_index = tidx + retained_pair * 2 * self.THREADS
                            first_query = first_flat_index // self.topk
                            first_topk_idx = first_flat_index % self.topk
                            second_flat_index = first_flat_index + self.THREADS
                            second_query = second_flat_index // self.topk
                            second_topk_idx = second_flat_index % self.topk
                            retained_id_pairs[retained_pair] = cutlass.Uint32(
                                selected_slots[first_query, first_topk_idx]
                            ) | (cutlass.Uint32(selected_slots[second_query, second_topk_idx]) << 16)
                        cute.arch.sync_threads()

                self._select_topk(
                    candidate_keys,
                    selected_slots,
                    histogram,
                    warp_offsets,
                    threshold_bin,
                    output_counts,
                    active_count,
                    tidx,
                )

                if cutlass.const_expr(self.use_candidate_compaction):
                    if processed < key_count and tidx < self.QUERY_BLOCK:
                        score_threshold[tidx] = cutlass.Uint32(0xFFFFFFFF)
                        if cutlass.const_expr(self.topk == 2048):
                            candidate_counts[tidx] = 0
                        else:
                            output_counts[tidx, 0] = 0

                if cutlass.const_expr(self.use_overlay_shared_ids):
                    if processed > first_batch_size:
                        for retained_pair in cutlass.range_constexpr(self.retained_per_thread // 2):
                            for pair_offset in cutlass.range_constexpr(2):
                                flat_index = tidx + retained_pair * 2 * self.THREADS + pair_offset * self.THREADS
                                query = flat_index // self.topk
                                topk_idx = flat_index % self.topk
                                retained_id_scratch[query, topk_idx] = cutlass.Uint16(
                                    retained_id_pairs[retained_pair] >> (pair_offset * 16)
                                )
                        cute.arch.sync_threads()

                # Capture every selected value and key ID in registers before
                # reusing the front of the candidate buffer for the next
                # merge batch.
                for retained_slot in cutlass.range_constexpr(self.retained_per_thread):
                    flat_index = tidx + retained_slot * self.THREADS
                    query = flat_index // self.topk
                    topk_idx = flat_index % self.topk
                    slot = cutlass.Int32(selected_slots[query, topk_idx])
                    retained_keys[retained_slot] = candidate_keys[
                        query,
                        slot,
                    ]

                    selected_key_idx = cutlass.Int32(-1)
                    if cutlass.const_expr(self.use_candidate_compaction) and processed > first_batch_size:
                        selected_key_idx = candidate_id_base + cutlass.Int32(candidate_indices[query, slot])
                        if retained_keys[retained_slot] == cutlass.Uint32(self.ORDERED_NEG_INF):
                            selected_key_idx = -1
                    elif processed <= first_batch_size:
                        if key_count > self.topk:
                            selected_key_idx = first_k + slot
                        elif slot < self.topk:
                            if cutlass.const_expr(self.use_packed_intermediate_output):
                                selected_key_idx = cutlass.Int32(
                                    intermediate_output[(query_offset + query) * self.topk * 2 + slot]
                                )
                            else:
                                selected_key_idx = cutlass.Int32(
                                    output[
                                        query_offset + query,
                                        0,
                                        slot,
                                    ]
                                )
                        else:
                            selected_key_idx = first_k + slot - self.topk
                    elif slot < self.topk:
                        if cutlass.const_expr(self.use_shared_intermediate_output):
                            selected_key_idx = first_k + cutlass.Int32(retained_id_scratch[query, slot]) - 1
                        elif cutlass.const_expr(self.use_packed_intermediate_output):
                            selected_key_idx = cutlass.Int32(
                                intermediate_output[(query_offset + query) * self.topk * 2 + slot]
                            )
                        else:
                            selected_key_idx = cutlass.Int32(
                                output[
                                    query_offset + query,
                                    0,
                                    slot,
                                ]
                            )
                    else:
                        selected_key_idx = first_k + processed - batch_count + slot - self.topk
                    if selected_key_idx < starts[query] or selected_key_idx >= ends[query]:
                        selected_key_idx = -1
                    retained_indices[retained_slot] = selected_key_idx
                cute.arch.sync_threads()

                if cutlass.const_expr(self.use_candidate_compaction) and processed < key_count:
                    retained_per_query = self.topk // self.THREADS
                    for query in cutlass.range_constexpr(self.QUERY_BLOCK):
                        local_threshold = cutlass.Uint32(0xFFFFFFFF)
                        for query_slot in cutlass.range_constexpr(retained_per_query):
                            retained_slot = query * retained_per_query + query_slot
                            if retained_keys[retained_slot] < local_threshold:
                                local_threshold = retained_keys[retained_slot]
                        local_threshold = cute.arch.warp_redux_sync(
                            local_threshold,
                            kind="umin",
                        )
                        if lane_idx == 0:
                            cute.arch.atomic_min(
                                (score_threshold.iterator + query).llvm_ptr,
                                local_threshold,
                            )
                    cute.arch.sync_threads()

                # selected_slots is dead after the gather, so the aliased K
                # range can be restaged while retained results are scattered.
                if cutlass.const_expr(self.merge_reuses_k) and key_block + 1 < block_count:
                    if cutlass.const_expr(self.use_tma_k_stage):
                        if tma_warp_idx == 0:
                            tma_pipeline.producer_acquire(tma_producer_state)
                            tma_barrier = tma_pipeline.producer_get_barrier(tma_producer_state)
                            cute.copy(
                                tma_atom_k,
                                tma_gmem_k[None, key_block + 1, 0],
                                tma_smem_k,
                                tma_bar_ptr=tma_barrier,
                            )
                            tma_producer_state.advance()
                    elif cutlass.const_expr(self.use_pair_scoped_k_ready):
                        self._stage_paired_k_rows(
                            k,
                            index_k,
                            first_k,
                            last_k,
                            block_offset + self.BLOCK_N,
                            warp_group,
                            tidx,
                            async_copy,
                        )
                    else:
                        self._stage_k_rows(
                            k,
                            index_k,
                            first_k,
                            last_k,
                            block_offset + self.BLOCK_N,
                            cutlass.Int32(0),
                            self.BLOCK_N,
                            tidx,
                            self.THREADS,
                            async_copy,
                        )

                for retained_slot in cutlass.range_constexpr(self.retained_per_thread):
                    flat_index = tidx + retained_slot * self.THREADS
                    query = flat_index // self.topk
                    topk_idx = flat_index % self.topk
                    if processed < key_count:
                        candidate_keys[
                            query,
                            topk_idx,
                        ] = retained_keys[retained_slot]
                    if cutlass.const_expr(self.use_candidate_compaction):
                        if processed < key_count:
                            candidate_indices[
                                query,
                                topk_idx,
                            ] = self.candidate_index_dtype(retained_indices[retained_slot] - candidate_id_base)
                        elif query_offset + query < q.shape[0]:
                            output[
                                query_offset + query,
                                0,
                                topk_idx,
                            ] = retained_indices[retained_slot]
                    elif cutlass.const_expr(self.use_shared_intermediate_output):
                        if query_offset + query < q.shape[0] and processed < key_count:
                            encoded_key_idx = cutlass.Uint16(0)
                            if retained_indices[retained_slot] >= first_k:
                                encoded_key_idx = cutlass.Uint16(retained_indices[retained_slot] - first_k + 1)
                            if cutlass.const_expr(self.use_persistent_shared_ids):
                                retained_id_scratch[query, topk_idx] = encoded_key_idx
                            else:
                                selected_slots[query, topk_idx] = encoded_key_idx
                        elif query_offset + query < q.shape[0]:
                            output[
                                query_offset + query,
                                0,
                                topk_idx,
                            ] = retained_indices[retained_slot]
                    elif (
                        cutlass.const_expr(self.use_packed_intermediate_output)
                        and query_offset + query < q.shape[0]
                        and processed < key_count
                    ):
                        intermediate_output[(query_offset + query) * self.topk * 2 + topk_idx] = retained_indices[
                            retained_slot
                        ]
                    elif query_offset + query < q.shape[0]:
                        output[
                            query_offset + query,
                            0,
                            topk_idx,
                        ] = retained_indices[retained_slot]
                cute.arch.sync_threads()
                candidate_tile_offset = self.topk


@functools.cache
def _compile_kernel(
    q_shape: tuple[int, int, int],
    k_shape: tuple[int, int],
    topk: int,
    device_index: int,
):
    """Compile one static-shape TVM-FFI entry point and cache it."""
    with torch.cuda.device(device_index):
        cutlass.cuda.initialize_cuda_context()
        q = torch.empty(q_shape, device="cuda", dtype=torch.bfloat16)
        k = torch.empty(k_shape, device="cuda", dtype=torch.bfloat16)
        weights = torch.empty(q_shape[:2], device="cuda", dtype=torch.float32)
        starts = torch.empty(q_shape[0], device="cuda", dtype=torch.int32)
        ends = torch.empty_like(starts)
        output = torch.empty(
            (q_shape[0], 1, topk),
            device="cuda",
            dtype=torch.int64,
        )
        args = [
            from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True)
            for tensor in (q, k, weights, starts, ends, output)
        ]
        return cute.compile(
            _CuteDSLIndexerTopK(q_shape[1], q_shape[2], topk, k_shape[0]),
            *args,
            options="--enable-tvm-ffi",
        )


def indexer_topk_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    index_topk: int,
) -> torch.Tensor:
    """Run the fused CuTe DSL kernel on already-normalized inputs."""
    topk = min(index_topk, k.shape[0])
    if q.shape[1:] != (32, 128):
        raise RuntimeError("CuTe DSL DSA indexer requires q.shape[1:] == (32, 128).")
    if topk not in (1024, 2048):
        raise RuntimeError("CuTe DSL DSA indexer supports topk=1024 or topk=2048.")

    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    output = torch.empty(
        (q.shape[0], 1, topk),
        device=q.device,
        dtype=torch.int64,
    )
    kernel = _compile_kernel(
        tuple(q.shape),
        tuple(k.shape),
        topk,
        device_index,
    )
    kernel(q, k, weights, starts, ends, output)
    # The kernel keeps int64 storage because packed specializations reuse each
    # slot as two int32 scratch entries. SparseMLA's public index contract is
    # int32, so narrow only after the final IDs have been written.
    return output.to(torch.int32)


def ensure_cute_dsl_runtime_available() -> None:
    """Validate the architecture required by the SM90 WGMMA kernel."""
    if not torch.cuda.is_available():
        raise RuntimeError("CuTe DSL DSA indexer requires CUDA.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise RuntimeError(f"CuTe DSL DSA indexer currently requires an SM90 GPU (found sm_{major}{minor}).")


def cute_dsl_dsa_topk_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    """Return exact DSA top-k IDs without materializing global logits."""
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise RuntimeError("CuTe DSL DSA indexer requires bfloat16 q and k tensors.")
    if weights.dtype != torch.float32:
        raise RuntimeError("CuTe DSL DSA indexer requires float32 weights.")
    if not q.is_cuda or not k.is_cuda or not weights.is_cuda:
        raise RuntimeError("CuTe DSL DSA indexer requires CUDA tensors.")
    if q.shape[0] != 1 or k.shape[0] != 1 or weights.shape[0] != 1:
        raise RuntimeError("CuTe DSL DSA indexer requires batch size 1.")
    ensure_cute_dsl_runtime_available()

    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    weights = (weights.squeeze(0) * (index_head_dim**-0.5)).contiguous()
    starts, ends = seq_ctx.packed_causal_query_ranges(q.shape[0], q.device)
    return _cute_dsl_dsa_topk_indices_from_ranges(
        q,
        k,
        weights,
        starts,
        ends,
        index_topk,
    )


@torch.library.custom_op(
    "sparse_mla::cute_dsl_dsa_topk_indices",
    mutates_args=(),
    device_types="cuda",
)
def _cute_dsl_dsa_topk_indices_from_ranges(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    starts: Tensor,
    ends: Tensor,
    index_topk: int,
) -> Tensor:
    return indexer_topk_interface(
        q,
        k,
        weights,
        starts,
        ends,
        index_topk,
    )


@_cute_dsl_dsa_topk_indices_from_ranges.register_fake
def _(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    starts: Tensor,
    ends: Tensor,
    index_topk: int,
) -> Tensor:
    topk = min(index_topk, k.shape[0])
    return torch.empty(
        (q.shape[0], 1, topk),
        device=q.device,
        dtype=torch.int32,
    )
