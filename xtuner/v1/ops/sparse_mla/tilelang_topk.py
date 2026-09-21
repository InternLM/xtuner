# Copyright (c) OpenMMLab. All rights reserved.

import torch
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}
_RADIX = 1 << 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(x, T.uint32)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.cast((0xFFFFFFFF), T.uint32),
        bits_uint | T.cast((0x80000000), T.uint32),
    )
    return bits_uint


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_impl(input, index, starts, ends, threads=1024, in_dtype=T.float32, out_dtype=T.int32):
    topk = T.const("topk")
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = _RADIX
    histogram_size = RADIX * 2 if threads == RADIX else RADIX + 1
    # The FP32 high-byte distribution is concentrated around the normal
    # range.  For the 16K training shape the boundary bucket can exceed
    # 4K, so leave enough shared-memory slots for the full candidate set.
    SMEM_INPUT_SIZE = 8192

    input: T.Tensor[(batch, seq_len), in_dtype]
    index: T.Tensor[(batch, topk), out_dtype]
    starts: T.Tensor[(batch), out_dtype]
    ends: T.Tensor[(batch), out_dtype]

    with T.Kernel(batch, threads=threads) as (bx):
        tx = T.get_thread_binding()

        s_threshold_bin_id = T.alloc_shared([1], T.int32)
        # Pad the 257 logical entries when using 256 threads so T.fill has a
        # bijective two-element-per-thread layout on the ROCm launch.
        s_histogram = T.alloc_shared([histogram_size], T.int32)
        s_num_input = T.alloc_shared([2], T.int32)
        s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

        l_threshold_bin_id = T.alloc_var(T.int32)
        l_new_topk = T.alloc_var(T.int32)
        l_num_input = T.alloc_var(T.int32)
        l_bin_id32 = T.alloc_var(T.int32)
        l_val = T.alloc_var(T.int32)
        l_start_pos = T.alloc_var(T.int32)
        l_start_idx = T.alloc_var(T.int32)
        l_end_idx = T.alloc_var(T.int32)
        l_out_pos = T.alloc_var(T.int32)
        l_dense_scan = T.alloc_var(T.int32)
        l_key_prefix = T.alloc_var(T.uint32)
        l_input_idx = T.alloc_var(T.int32)
        l_key = T.alloc_var(T.uint32)

        pos = T.alloc_var(T.int32)

        l_new_topk = topk
        l_start_idx = starts[bx]
        l_end_idx = ends[bx]

        # Initialize padding slots in the same launch as selection.  The
        # causal prefix of early rows can contain fewer than ``topk`` values;
        # keeping this write in the selector avoids a separate aten.fill
        # kernel on the training path.
        for i in T.serial(T.ceildiv(topk, threads)):
            output_pos = i * threads + tx
            if output_pos < topk:
                index[bx, output_pos] = -1
        T.sync_threads()

        # stage 1: use 8bit to do quick topk
        T.fill(s_histogram, 0)
        T.fill(s_num_input[0], 0)

        # Default to bin 0 in case no threshold crossing is found, e.g. when
        # the valid input range has no more than topk elements.
        T.fill(s_threshold_bin_id, 0)

        T.sync_threads()
        for s in T.serial(T.ceildiv(seq_len, threads * 4)):
            input_base = s * threads * 4 + tx * 4
            for j in T.serial(4):
                input_idx = input_base + j
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    # Use the most-significant byte of the FP32 key directly.
                    # A float16 cast here can move values across a radix-bin
                    # boundary; that makes the threshold bucket incomplete
                    # and can select a different index than torch.topk.
                    inval_bin = (convert_to_uint32(input[bx, input_idx]) >> 24) & 0xFF
                    T.atomic_add(s_histogram[inval_bin], 1)
        T.sync_threads()

        # cumsum
        if tx < RADIX:
            for i in T.serial(8):
                offset = 1 << i
                T.sync_threads(3, RADIX)
                if tx < RADIX - offset:
                    l_val = s_histogram[tx] + s_histogram[tx + offset]
                T.sync_threads(3, RADIX)
                if tx < RADIX - offset:
                    s_histogram[tx] = l_val

            # find threshold bin id
            T.sync_threads(3, RADIX)
            # s_histogram[tx] is the suffix count for bins >= tx. Use >=/< to
            # also catch the exact-boundary case where that count equals topk.
            if s_histogram[tx] >= l_new_topk and s_histogram[tx + 1] < l_new_topk:
                s_threshold_bin_id[0] = tx
        T.sync_threads()
        l_threshold_bin_id = s_threshold_bin_id[0]
        l_dense_scan = T.cast(
            s_histogram[l_threshold_bin_id] - s_histogram[l_threshold_bin_id + 1] > SMEM_INPUT_SIZE,
            T.int32,
        )
        l_key_prefix = T.cast(l_threshold_bin_id, T.uint32)
        l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
        T.sync_threads()

        # collect all elements with exponent ≥ threshold
        for s in T.serial(T.ceildiv(seq_len, threads * 4)):
            T.sync_threads()
            input_base = s * threads * 4 + tx * 4
            for j in T.serial(4):
                input_idx = input_base + j
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    bin_id = (convert_to_uint32(input[bx, input_idx]) >> 24) & 0xFF
                    l_bin_id32 = T.cast(bin_id, T.int32)
                    if l_bin_id32 > l_threshold_bin_id:
                        # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                        index[bx, pos] = input_idx

                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0 and l_dense_scan == 0:
                        # pos = s_num_input[0]
                        pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                        s_input_idx[0, pos] = input_idx

        # stage 2: tail pass
        # The first byte was consumed by stage 1.  Select the remaining three
        # FP32 bytes from most to least significant.
        for round in T.serial(3):
            if l_new_topk <= 0:
                break

            r_idx = round % 2
            l_start_pos = topk - l_new_topk

            T.sync_threads()
            T.fill(s_histogram, 0)
            if tx == 0:
                s_num_input[r_idx ^ 1] = 0
            T.sync_threads()

            l_num_input = s_num_input[r_idx]
            if l_dense_scan != 0:
                # A concentrated FP32 distribution can overflow the bounded
                # candidate buffer. Refine the same prefix by rescanning the
                # dense row instead; the ordinary compact path is unchanged.
                l_num_input = seq_len
            for s in T.serial(T.ceildiv(l_num_input, threads)):
                if s * threads + tx < l_num_input:
                    if l_dense_scan != 0:
                        l_input_idx = s * threads + tx
                    else:
                        l_input_idx = s_input_idx[r_idx, s * threads + tx]
                    if l_input_idx >= l_start_idx and l_input_idx < l_end_idx:
                        l_key = convert_to_uint32(input[bx, l_input_idx])
                        if l_dense_scan == 0 or (l_key >> (24 - round * 8)) == l_key_prefix:
                            l_bin_id32 = T.cast((l_key >> (16 - round * 8)) & 0xFF, T.int32)
                            T.atomic_add(s_histogram[l_bin_id32], 1)
            T.sync_threads()
            # cumsum
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                # find threshold bin id
                T.sync_threads(3, RADIX)
                # s_histogram[tx] is the suffix count for bins >= tx. Use >=/< to
                # also catch the exact-boundary case where that count equals topk.
                if s_histogram[tx] >= l_new_topk and s_histogram[tx + 1] < l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()

            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            for s in T.serial(T.ceildiv(l_num_input, threads)):
                T.sync_threads()
                if s * threads + tx < l_num_input:
                    if l_dense_scan != 0:
                        l_input_idx = s * threads + tx
                    else:
                        l_input_idx = s_input_idx[r_idx, s * threads + tx]
                    if l_input_idx >= l_start_idx and l_input_idx < l_end_idx:
                        l_key = convert_to_uint32(input[bx, l_input_idx])
                        if l_dense_scan == 0 or (l_key >> (24 - round * 8)) == l_key_prefix:
                            l_bin_id32 = T.cast((l_key >> (16 - round * 8)) & 0xFF, T.int32)
                            if l_bin_id32 > l_threshold_bin_id:
                                pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                index[bx, pos] = l_input_idx
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                if round == 2:
                                    l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                    if l_out_pos < topk:
                                        index[bx, l_out_pos] = l_input_idx
                                elif l_dense_scan == 0:
                                    pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                    s_input_idx[r_idx ^ 1, pos] = l_input_idx
            l_key_prefix = (l_key_prefix << 8) | T.cast(l_threshold_bin_id, T.uint32)


def tl_topk(input: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, topk: int) -> torch.Tensor:
    batch, _ = input.shape
    indexes = torch.empty((batch, topk), dtype=torch.int32, device=input.device)
    # CUDA supports named barriers for the 256-thread radix subgroup. HIP
    # currently lowers that synchronization to a full workgroup barrier, so
    # launch exactly the radix participants there to keep every barrier
    # convergent without changing the tuned CUDA configuration.
    threads = _RADIX if torch.version.hip is not None else 1024
    tl_topk_impl(input, indexes, starts, ends, threads=threads)
    # Radix selection is unordered. Sort only the selected values to match
    # torch.topk's descending order without sorting the full logits row.
    valid = indexes >= 0
    selected_scores = input.gather(1, indexes.clamp_min(0).long()).masked_fill(~valid, -torch.inf)
    order = selected_scores.argsort(dim=-1, descending=True)
    return indexes.gather(1, order)
