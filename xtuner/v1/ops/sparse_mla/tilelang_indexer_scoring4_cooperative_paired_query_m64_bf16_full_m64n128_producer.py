# Copyright (c) OpenMMLab. All rights reserved.
"""Opt-in full-output BF16 scorer for installed DeepSelect.

Retains four M64N128 scoring WGs, a dedicated producer and two K stages.
All scheduled tiles publish directly to a full FP32 GPU matrix. Scores in
[0, end) outside each query's [start, end) range are initialized to -inf;
unscheduled suffixes after end are unspecified and DeepSelect ignores them.
DeepSelect has no begin support, so its companion wrapper requires starts=0.
"""

import tilelang
import torch
from tilelang import language as T
from tilelang.layout import make_wgmma_swizzled_layout
from torch import Tensor


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
}

_HELPER_SOURCE = r"""
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
__device__ __forceinline__ void xtuner_bf16_full_scorer_tma256_gemm(
    const void* k_stage, const void* q_shared, float* accum, int key_offset) {
  // BF16D128 stores two64-column planes under128-byte swizzling.
  // Each plane has128 bytes/row; a64-row quarter starts every8192 bytes.
  // The K stage plane is256*128=32768 bytes; the Q plane128*128=16384.
  // These BF16 K16 plane jumps differ from linear FP8 K32 increments.
  tl::GmmaDescriptor desc_a, desc_b;
  tl::initialize_wgmma_descriptor<1, 1, 64>(
      desc_a, reinterpret_cast<const char*>(k_stage) + key_offset * 128);
  tl::initialize_wgmma_descriptor<1, 1, 64>(
      desc_b, reinterpret_cast<const char*>(q_shared));
  tl::fence_proxy_async();
  tl::warpgroup_fence_operand(accum, 64);
  tl::warpgroup_arrive();
  #pragma unroll
  for (int ki = 0; ki < 8; ++ki) {
    tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
        tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(
        uint64_t(desc_a + (((ki >> 2) * 32768 + (ki & 3) * 32) >> 4)),
        uint64_t(desc_b + (((ki >> 2) * 16384 + (ki & 3) * 32) >> 4)),
        reinterpret_cast<uint32_t*>(accum), ki > 0 ? 1 : 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_wait<0>();
  tl::warpgroup_fence_operand(accum, 64);
}

__device__ __forceinline__ void xtuner_bf16_full_scorer_epilogue_load_weights(
    float* weights, const float* source, int q_start, int query_len) {
  const int head_pair = (int(threadIdx.x) & 3) * 2;
  #pragma unroll
  for (int qi = 0; qi < 4; ++qi) {
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 value = q_start + qi < query_len
          ? *reinterpret_cast<const float2*>(source + (q_start + qi) * 32 + pair * 8 + head_pair)
          : make_float2(0.0f, 0.0f);
      *reinterpret_cast<float2*>(weights + qi * 8 + pair * 2) = value;
    }
  }
}
__device__ __forceinline__ void xtuner_bf16_full_scorer_epilogue_store(
    float value0, float value1, float* scores,
    int tile_start, int key_offset, int start0, int end0, int start1, int end1,
    int start2, int end2, int start3, int end3, int last, int key_len,
    int q_start, int query_len, int score_stride) {
  const int lane = int(threadIdx.x) & 31;
  const int warp = (int(threadIdx.x) & 127) >> 5;
  const int qi = lane & 3;
  const int ni0 = warp * 16 + (lane >> 2);
  const int query_start = qi == 0 ? start0 : qi == 1 ? start1 : qi == 2 ? start2 : start3;
  const int query_end = qi == 0 ? end0 : qi == 1 ? end1 : qi == 2 ? end2 : end3;
  #pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int ni = ni0 + half * 8;
    const int key_id = tile_start + key_offset + ni;
    // Direct BF16 inputs have no Q/K quantization scales.
    float value = half ? value1 : value0;
    if (value == 0.0f) value = 0.0f;
    const bool valid = q_start + qi < query_len && key_id < last && key_id < key_len
        && key_id >= query_start && key_id < query_end;
    if (q_start + qi < query_len && key_id < key_len) {
      // A full128K*32K matrix exceeds int32 element addressing.
      const int64_t offset = int64_t(q_start + qi) * score_stride + key_id;
      scores[offset] = valid ? value : -__int_as_float(0x7f800000);
    }
  }
}
__device__ __forceinline__ void xtuner_bf16_full_scorer_epilogue_fma(
    const float* raw_dot, const float* weights, float* scores, int tile_start, int key_offset,
    int start0, int end0, int start1, int end1, int start2, int end2,
    int start3, int end3, int last, int key_len, int q_start,
    int query_len, int score_stride) {
  const int publish_qi = int(threadIdx.x) & 3;
  float value0 = 0.0f, value1 = 0.0f;
  // Native64 accumulators: each query segment has16 values, two key rows
  // times eight lane-local heads. Native weights have8 values/query/thread.
  #pragma unroll
  for (int qi = 0; qi < 4; ++qi) {
    const float* a = raw_dot + qi * 16;
    const float* w = weights + qi * 8;
    float sum0 = fmaxf(a[0], 0.0f) * w[0];
    float sum1 = fmaxf(a[1], 0.0f) * w[1];
    float sum2 = fmaxf(a[2], 0.0f) * w[0];
    float sum3 = fmaxf(a[3], 0.0f) * w[1];
    #pragma unroll
    for (int j = 1; j < 4; ++j) {
      // Explicit fmaf is intentional; the kernel retains --fmad=false.
      sum0 = fmaf(fmaxf(a[j * 4], 0.0f), w[j * 2], sum0);
      sum1 = fmaf(fmaxf(a[j * 4 + 1], 0.0f), w[j * 2 + 1], sum1);
      sum2 = fmaf(fmaxf(a[j * 4 + 2], 0.0f), w[j * 2], sum2);
      sum3 = fmaf(fmaxf(a[j * 4 + 3], 0.0f), w[j * 2 + 1], sum3);
    }
    float v0 = sum0 + sum1;
    float v1 = sum2 + sum3;
    // Keep every lane active and the current reduction's XOR2 then XOR1.
    v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
    v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
    v0 += __shfl_xor_sync(0xffffffffu, v0, 1);
    v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
    if (qi == publish_qi) {
      value0 = v0;
      value1 = v1;
    }
  }
  xtuner_bf16_full_scorer_epilogue_store(value0, value1, scores,
      tile_start, key_offset, start0, end0, start1, end1, start2, end2,
      start3, end3, last, key_len, q_start, query_len, score_stride);
}
__device__ __forceinline__ void xtuner_bf16_full_scorer_initialize_prefix(
    float* scores, int score_stride, int q_start, int query_len, int first) {
  if (first <= 0) return;
  for (int i = int(threadIdx.x); i < first * 4; i += 512) {
    const int qi = i / first;
    const int key = i - qi * first;
    if (q_start + qi < query_len)
      scores[int64_t(q_start + qi) * score_stride + key] = -__int_as_float(0x7f800000);
  }
}
"""


@tilelang.jit(pass_configs=_PASS_CONFIGS, compile_flags=["--maxrregcount=96", "--fmad=false"])
def _scorer_full_kernel(math_registers: int = 112, helper_source: str = _HELPER_SOURCE):
    if math_registers not in (112, 120):
        raise ValueError("Producer scorer math budget must be112 or120.")
    fma_reduce = True
    query_len = T.dynamic("query_len")
    key_len = T.dynamic("key_len")
    score_stride = T.dynamic("score_stride")
    heads, index_dim, block_q, block_n = 32, 128, 4, 256

    @T.macro
    def load_k(K, k0, k1, k_free, k_ready, first, tile):
        stage = tile % 2
        T.barrier_wait(k_free[stage], (tile // 2) % 2)
        if stage == 0:
            T.tma_copy(K[first + tile * block_n, 0], k0, barrier=k_ready[0])
        else:
            T.tma_copy(K[first + tile * block_n, 0], k1, barrier=k_ready[1])
        if T.get_thread_binding() == 512:
            T.barrier_arrive(k_ready[stage])

    @T.macro
    def compute(
        wg,
        Weights,
        Starts,
        Ends,
        Scores,
        k0,
        k1,
        q_all,
        k_free,
        k_ready,
        q_start,
        first,
        last,
        tiles,
    ):
        T.set_max_nreg(math_registers, 1)
        key_offset = wg * 64
        # All four queries are separate groups of32 heads in the N dimension.
        dot = T.alloc_fragment((64, block_q * heads), T.float32)
        dot_heads = T.reshape(dot, (64, block_q, heads))  # noqa: F841 - retain the validated TileLang IR
        weights = T.alloc_fragment((block_q, heads), T.float32)
        logits = T.alloc_fragment((64, block_q), T.float32)
        # Opaque register epilogue access cannot infer weight indexing. This
        # is the original native32-value/thread layout, with32 replicas.
        if fma_reduce:
            T.annotate_layout(
                {
                    dot: T.Fragment(
                        (64, block_q * heads),
                        forward_thread_fn=lambda ni, col: ni // 16 * 32 + ni % 8 * 4 + col % 8 // 2,
                        forward_index_fn=lambda ni, col: col // 8 * 4 + ni % 16 // 8 * 2 + col % 2,
                    ),
                    weights: T.Fragment(
                        (block_q, heads),
                        forward_thread_fn=lambda qi, head, rep: head % 8 // 2 + rep * 4,
                        forward_index_fn=lambda qi, head: qi * 8 + head // 8 * 2 + head % 2,
                        replicate=32,
                    ),
                }
            )
        T.annotate_layout(
            {
                logits: T.Fragment(
                    (64, block_q),
                    forward_thread_fn=lambda ni, qi, rep: ni // 16 * 32 + ni % 8 * 4 + rep,
                    forward_index_fn=lambda ni, qi: ni % 16 // 8 * block_q + qi,
                    replicate=4,
                )
            }
        )
        start0 = T.alloc_var(T.int32)
        end0 = T.alloc_var(T.int32)
        start1 = T.alloc_var(T.int32)
        end1 = T.alloc_var(T.int32)
        start2 = T.alloc_var(T.int32)
        end2 = T.alloc_var(T.int32)
        start3 = T.alloc_var(T.int32)
        end3 = T.alloc_var(T.int32)
        start0 = 0
        end0 = 0
        start1 = 0
        end1 = 0
        start2 = 0
        end2 = 0
        start3 = 0
        end3 = 0
        if q_start < query_len:
            start0 = Starts[q_start]
            end0 = Ends[q_start]
        if q_start + 1 < query_len:
            start1 = Starts[q_start + 1]
            end1 = Ends[q_start + 1]
        if q_start + 2 < query_len:
            start2 = Starts[q_start + 2]
            end2 = Ends[q_start + 2]
        if q_start + 3 < query_len:
            start3 = Starts[q_start + 3]
            end3 = Ends[q_start + 3]
        if fma_reduce:
            # The opaque native register layout is local to each128-thread
            # warp group. Load its32 cached values directly in every group.
            T.call_extern(
                "handle",
                "xtuner_bf16_full_scorer_epilogue_load_weights",
                T.access_ptr(weights, "w"),
                Weights.data,
                q_start,
                query_len,
            )
        else:
            for qi, head in T.Parallel(block_q, heads):
                weights[qi, head] = T.if_then_else(
                    q_start + qi < query_len,
                    Weights[q_start + qi, head],
                    T.float32(0),
                )
        for stage in T.Unroll(2):
            T.barrier_arrive(k_free[stage])
        for tile in T.serial(tiles):
            stage = tile % 2
            tile_start = first + tile * block_n
            T.barrier_wait(k_ready[stage], (tile // 2) % 2)
            if stage == 0:
                T.call_extern(
                    "handle",
                    "xtuner_bf16_full_scorer_tma256_gemm",
                    T.access_ptr(k0, "r"),
                    T.access_ptr(q_all, "r"),
                    T.access_ptr(dot, "rw"),
                    key_offset,
                )
            else:
                T.call_extern(
                    "handle",
                    "xtuner_bf16_full_scorer_tma256_gemm",
                    T.access_ptr(k1, "r"),
                    T.access_ptr(q_all, "r"),
                    T.access_ptr(dot, "rw"),
                    key_offset,
                )
            T.sync_threads(wg + 1, 128)
            T.call_extern(
                "handle",
                "xtuner_bf16_full_scorer_epilogue_fma",
                T.access_ptr(dot, "r"),
                T.access_ptr(weights, "r"),
                T.access_ptr(Scores, "w"),
                tile_start,
                key_offset,
                start0,
                end0,
                start1,
                end1,
                start2,
                end2,
                start3,
                end3,
                last,
                key_len,
                q_start,
                query_len,
                score_stride,
            )
            # Retain the baseline producer schema: release after publication.
            T.barrier_arrive(k_free[stage])

    @T.prim_func
    def bf16_scorer_full_m64n128_producer(
        Q: T.Tensor((query_len * heads, index_dim), T.bfloat16),
        K: T.Tensor((key_len, index_dim), T.bfloat16),
        Weights: T.Tensor((query_len, heads), T.float32),
        Starts: T.Tensor((query_len,), T.int32),
        Ends: T.Tensor((query_len,), T.int32),
        Scores: T.StridedTensor((query_len, key_len), (score_stride, 1), T.float32),
    ):
        with T.Kernel(T.ceildiv(query_len, block_q), threads=640) as bx:
            T.annotate_min_blocks_per_sm(1)
            T.import_source(helper_source)
            q_all = T.alloc_shared((block_q * heads, index_dim), T.bfloat16)
            k0 = T.alloc_shared((block_n, index_dim), T.bfloat16)
            k1 = T.alloc_shared((block_n, index_dim), T.bfloat16)
            T.annotate_layout(
                {
                    q_all: make_wgmma_swizzled_layout(q_all, continuity=128),
                    k0: make_wgmma_swizzled_layout(k0, continuity=128),
                    k1: make_wgmma_swizzled_layout(k1, continuity=128),
                }
            )
            k_ready = T.alloc_barrier([1, 1])
            k_free = T.alloc_barrier([512, 512])
            q_start = bx * block_q
            first = T.alloc_var(T.int32)
            last = T.alloc_var(T.int32)
            first = key_len
            last = 0
            for qi in T.serial(block_q):
                if q_start + qi < query_len:
                    first = T.min(first, Starts[q_start + qi])
                    last = T.max(last, Ends[q_start + qi])
            tx = T.get_thread_binding()
            if tx < 512:
                T.copy(Q[q_start * heads, 0], q_all)
                T.call_extern(
                    "handle",
                    "xtuner_bf16_full_scorer_initialize_prefix",
                    T.access_ptr(Scores, "w"),
                    score_stride,
                    q_start,
                    query_len,
                    first,
                )
            T.sync_threads()
            tiles = T.ceildiv(T.max(last - first, 0), block_n)
            tx = T.get_thread_binding()
            if tx < 128:
                compute(0, Weights, Starts, Ends, Scores, k0, k1, q_all, k_free, k_ready, q_start, first, last, tiles)
            elif tx < 256:
                compute(1, Weights, Starts, Ends, Scores, k0, k1, q_all, k_free, k_ready, q_start, first, last, tiles)
            elif tx < 384:
                compute(2, Weights, Starts, Ends, Scores, k0, k1, q_all, k_free, k_ready, q_start, first, last, tiles)
            elif tx < 512:
                compute(3, Weights, Starts, Ends, Scores, k0, k1, q_all, k_free, k_ready, q_start, first, last, tiles)
            else:
                # All128 producer threads deallocate before any elected
                # lane begins loading; math increases can then acquire regs.
                T.set_max_nreg(32, 0)
                for tile in T.serial(tiles):
                    load_k(K, k0, k1, k_free, k_ready, first, tile)
            T.sync_threads()

    return bf16_scorer_full_m64n128_producer


def allocate_scores(query_len: int, key_len: int, device) -> Tensor:
    """Allocate minimum-stride FP32 input accepted by installed DeepSelect."""
    import deep_select

    alignment_bytes = deep_select.get_stride_requirement()[0]
    if alignment_bytes % 4:
        raise ValueError("DeepSelect FP32 stride alignment must be a multiple of4 bytes.")
    alignment = alignment_bytes // 4
    stride = (key_len + alignment - 1) // alignment * alignment
    return torch.empty((query_len, stride), device=device, dtype=torch.float32)[:, :key_len]


def scorer_full_interface(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    starts: Tensor,
    ends: Tensor,
    *,
    out: Tensor | None = None,
    math_registers: int = 112,
) -> Tensor:
    """Write all valid scores to FP32[Q,K] with a DeepSelect-aligned stride.

    Q[Q,32,128]/K[K,128] are BF16; weights[Q,32] are pre-scaled FP32.
    Starts/ends are int32, with0<=start<=end<=K. Every [start,end) value is
    written; preceding prefix holes are -inf. Unscheduled suffixes after
    end are unspecified. This matrix remains on GPU; allocation can be
    excluded by passing a preallocated output from allocate_scores().
    """
    import deep_select

    if q.ndim != 3 or q.shape[1:] != (32, 128) or k.ndim != 2 or k.shape[1:] != (128,) or not k.shape[0]:
        raise ValueError("Full BF16 scorer requires32 heads, dimension128 and at least one key.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or weights.dtype != torch.float32:
        raise ValueError("Full BF16 scorer requires BF16 Q/K and FP32 weights.")
    if starts.dtype != torch.int32 or ends.dtype != torch.int32:
        raise ValueError("Full BF16 scorer requires int32 ranges.")
    if weights.shape != q.shape[:2] or starts.shape != q.shape[:1] or ends.shape != q.shape[:1]:
        raise ValueError("Full BF16 weights/ranges must match query rows.")
    if not q.is_cuda or any(t.device != q.device for t in (k, weights, starts, ends)):
        raise ValueError("Full BF16 scorer requires one CUDA device.")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("Full BF16 scorer requires SM90.")
    if out is None:
        out = allocate_scores(q.shape[0], k.shape[0], q.device)
    if (
        out.device != q.device
        or out.dtype != torch.float32
        or out.shape != (q.shape[0], k.shape[0])
        or out.stride(1) != 1
        or out.stride(0) < k.shape[0]
        or out.stride(0) * 4 % deep_select.get_stride_requirement()[0]
    ):
        raise ValueError("Full BF16 output must have DeepSelect-aligned FP32[Q,K] storage.")
    if q.shape[0]:
        _scorer_full_kernel(math_registers=math_registers)(
            q.contiguous().view(-1, 128),
            k.contiguous(),
            weights.contiguous(),
            starts.contiguous(),
            ends.contiguous(),
            out,
        )
    return out


def deepselect_topk(
    scores: Tensor,
    starts: Tensor,
    ends: Tensor,
    topk: int = 512,
    *,
    output_idx: Tensor | None = None,
) -> Tensor:
    """Select valid zero-start prefixes; pad short rows with ID -1.

    The installed selector has no begin support. Nonzero starts are
    rejected because masking cannot reproduce its short-range padding.
    Provide output_idx[Q,topk] int32 to exclude output allocation.
    """
    import deep_select

    if starts.dtype != torch.int32 or starts.shape != scores.shape[:1] or bool((starts != 0).any()):
        raise ValueError("Installed DeepSelect requires zero starts and int32 ranges.")
    if ends.dtype != torch.int32 or ends.shape != scores.shape[:1]:
        raise ValueError("DeepSelect ends must be int32[Q].")
    if output_idx is not None and (
        output_idx.dtype != torch.int32
        or output_idx.device != scores.device
        or output_idx.shape != (scores.shape[0], topk)
        or not output_idx.is_contiguous()
    ):
        raise ValueError("DeepSelect output_idx must be contiguous int32[Q,topk].")
    _, indices = deep_select.topk(
        scores,
        topk,
        end=ends.contiguous(),
        indices_type=torch.int32,
        sorted=False,
        sorted_index=False,
        return_value=False,
        output_idx=output_idx,
        idx_oob_fill_value=-1,
    )
    return indices
