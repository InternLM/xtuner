# mypy: ignore-errors
# ruff: noqa
# fmt: off

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from statistics import quantiles
from typing import Optional, Dict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import os

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel_paral(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16
    base_t = i_t * LARGE_BLOCK_T
    tl.device_print("i_t:", i_t)
    tl.device_print("base_t:", base_t)

    b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)  # (N_BLOCKS, 16, 16)
    for blkid in range(0, N_BLOCKS):
        row_start_o = base_t + blkid * 16
        col_start_o = row_start_o % BT
        p_A_subrec16 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (row_start_o, col_start_o), (16, 16), (1, 0)
        )
        b_A_subrec16 = tl.load(p_A_subrec16, boundary_check=(0, 1)).to(
            tl.float32
        )  # (16, 16)
        b_A = tl.insert_slice(
            ful=b_A,
            sub=b_A_subrec16[None, :, :],  # (1, 16, 16)
            offsets=[blkid, 0, 0],
            sizes=[1, 16, 16],
            strides=[1, 1, 1],
        )

    # load multi 16x16 into UB
    local_ori_A = tl.trans(b_A, (1, 0, 2))
    local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))  # (16, N_BLOCKS*16)

    tmp = tl.arange(0, 16).to(tl.float32)
    rows = tmp[:, None]
    cols = tmp[None, :]
    is_lower = (rows > cols).to(b_A.dtype)
    b_A = -b_A * is_lower

    o_i = tl.arange(0, 16)
    for i in range(1, 16):

        nblks_vec16 = -tl.extract_slice(
            local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1)
        )
        b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

        dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
        dot_product = tl.sum(dot_tmp, 0)
        b_a = b_a + dot_product  # (N_BLOCKS, 16)

        row_mask = o_i == i  # (16,), True at position i
        update_mask = row_mask[None, :, None]  # (1, 16, 1)
        b_a_expanded = b_a[:, None, :]  # (N_BLOCKS, 1, 16)
        b_A = tl.where(update_mask, b_a_expanded, b_A)  # shape keeps (N_BLOCKS, 16, 16)

    on_diagonal = rows == cols
    b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

    b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))
    p_Ai = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (base_t, 0), (N_BLOCKS * 16, 16), (1, 0)
    )
    tl.store(
        p_Ai,
        b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel_paral_v3(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    base_t = i_t * LARGE_BLOCK_T

    NTASKS: tl.constexpr = 2
    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16 // NTASKS

    for taskid in range(0, NTASKS):
        base_t += taskid * (LARGE_BLOCK_T // NTASKS)

        b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)  # (N_BLOCKS, 16, 16)
        for blkid in range(0, N_BLOCKS):
            row_start_o = base_t + blkid * 16
            col_start_o = row_start_o % BT
            # using ptr with mask instead of tl.load(block_ptr)
            offs_rows_in_block = tl.arange(0, 16)
            offs_cols_in_block = tl.arange(0, 16)
            # strides (H*BT, 1)
            ptr_A_subrec16 = (
                A
                + row_start_o * H * BT
                + col_start_o
                + offs_rows_in_block[:, None] * H * BT
                + offs_cols_in_block[None, :]
            )
            global_rows = row_start_o + offs_rows_in_block[:, None]
            global_cols = col_start_o + offs_cols_in_block[None, :]
            load_mask = (global_rows < T) & (global_cols < BT)
            b_A_subrec16 = tl.load(ptr_A_subrec16, mask=load_mask, other=0.0).to(
                tl.float32
            )
            b_A = tl.insert_slice(
                ful=b_A,
                sub=b_A_subrec16[None, :, :],  # (1, 16, 16)
                offsets=[blkid, 0, 0],
                sizes=[1, 16, 16],
                strides=[1, 1, 1],
            )

        # load multi 16x16
        local_ori_A = tl.trans(b_A, (1, 0, 2))
        local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))  # (16, N_BLOCKS*16)

        # change mask into matrix elementwise action
        tmp = tl.arange(0, 16).to(tl.float32)
        rows = tmp[:, None]
        cols = tmp[None, :]
        is_lower = (rows > cols).to(b_A.dtype)
        b_A = -b_A * is_lower

        for i in range(1, 16):

            nblks_vec16 = -tl.extract_slice(
                local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1)
            )
            b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

            dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
            dot_product = tl.sum(dot_tmp, 0)
            b_a = b_a + dot_product  # (N_BLOCKS, 16)

            b_a_new_expanded = b_a[:, None, :]  # (N_BLOCKS, 1, 16)
            b_A = tl.insert_slice(
                ful=b_A,
                sub=b_a_new_expanded,
                offsets=[0, i, 0],
                sizes=[N_BLOCKS, 1, 16],
                strides=[1, 1, 1],
            )

        on_diagonal = rows == cols
        b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

        b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))
        p_Ai = tl.make_block_ptr(
            Ad, (T, 16), (H * 16, 1), (base_t, 0), (N_BLOCKS * 16, 16), (1, 0)
        )
        # using ptr with mask instead of tl.load(block_ptr)
        offs_rows_to_store = tl.arange(0, N_BLOCKS * 16)
        offs_cols_to_store = tl.arange(0, 16)
        # strides (H*16, 1)
        p_Ai = (
            Ad
            + base_t * H * 16
            + 0
            + offs_rows_to_store[:, None] * H * 16
            + offs_cols_to_store[None, :]
        )
        global_store_rows = base_t + offs_rows_to_store[:, None]
        store_mask = global_store_rows < T
        tl.store(
            p_Ai,
            b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=store_mask,
        )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tt, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_tt * 2).to(tl.int32), tl.load(
            chunk_indices + i_tt * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * BT
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 32

    p_A_21 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * 32 + 16, 0 + i_t % (BT // 32) * 32), (16, 16), (1, 0)
    )
    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def merge_32x32_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * BT
    Ad += (bos * H + i_h) * 32
    Ai += (bos * H + i_h) * 64

    p_A_21 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * 64 + 32, 0 + i_t % (BT // 64) * 64), (32, 32), (1, 0)
    )

    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 32), (H * 32, 1), (i_t * 64, 0), (32, 32), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 32), (H * 32, 1), (i_t * 64 + 32, 0), (32, 32), (1, 0)
    )

    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 0), (32, 32), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 32), (32, 32), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (32, 32), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def merge_64x64_to_128x128_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 128
    Ad += (bos * H + i_h) * 64
    Ai += (bos * H + i_h) * 128

    p_A_21 = tl.make_block_ptr(
        A, (T, 128), (H * 128, 1), (i_t * 128 + 64, 0), (64, 64), (1, 0)
    )

    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 64), (H * 64, 1), (i_t * 128, 0), (64, 64), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 64), (H * 64, 1), (i_t * 128 + 64, 0), (64, 64), (1, 0)
    )

    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 128), (H * 128, 1), (i_t * 128, 0), (64, 64), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 128), (H * 128, 1), (i_t * 128 + 64, 64), (64, 64), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 128), (H * 128, 1), (i_t * 128 + 64, 0), (64, 64), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel_reorder_all_masked(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t_val = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        i_t = i_t_val
    else:
        bos, eos = i_b * T, i_b * T + T

    # Base pointers (already offset by batch and head)
    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64

    # ------------------ Load Ai_22 (Ad block at row i_t*64+16, col 0, 16x16) ------------------
    offs_m = i_t * 64 + 16 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_22 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # ------------------ Load A_21 (A block at row i_t*64+16, col 0, 16x16) ------------------
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)  # A has 64 cols
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_21 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

    tmp = tl.dot(Ai_22, A_21, input_precision="ieee")

    # ------------------ Load Ai_11 (Ad block at row i_t*64, col 0, 16x16) ------------------
    offs_m = i_t * 64 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_11 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_21 = -tl.dot(tmp, Ai_11, input_precision="ieee")

    # ------------------ Load Ai_44 (Ad block at row i_t*64+48, col 0, 16x16) ------------------
    offs_m = i_t * 64 + 48 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_44 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # ------------------ Load A_43 (A block at row i_t*64+48, col 32, 16x16) ------------------
    offs_n = 32 + tl.arange(0, 16)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_43 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

    tmp = tl.dot(Ai_44, A_43, input_precision="ieee")

    # ------------------ Load Ai_33 (Ad block at row i_t*64+32, col 0, 16x16) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_33 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_43 = -tl.dot(tmp, Ai_33, input_precision="ieee")

    # ------------------ Build Ai_22_32 (32x32) ------------------
    Ai_22_32 = tl.zeros((32, 32), tl.float32)
    Ai_22_32 = tl.insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16), (1, 1))
    Ai_22_32 = tl.insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16), (1, 1))
    Ai_22_32 = tl.insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16), (1, 1))

    # ------------------ Load A_21_32 (A block at row i_t*64+32, col 0, 32x32) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_21_32 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

    tmp = tl.dot(Ai_22_32, A_21_32, input_precision="ieee")

    # ------------------ Build Ai_11_32 (32x32) ------------------
    Ai_11_32 = tl.zeros((32, 32), tl.float32)
    Ai_11_32 = tl.insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16), (1, 1))
    Ai_11_32 = tl.insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16), (1, 1))
    Ai_11_32 = tl.insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16), (1, 1))

    Ai_21_32 = -tl.dot(tmp, Ai_11_32, input_precision="ieee")

    # ------------------ Store Ai_11_32 to (i_t*64, 0) ------------------
    offs_m = i_t * 64 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_11_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # ------------------ Store Ai_22_32 to (i_t*64+32, 32) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = 32 + tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_22_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # ------------------ Store Ai_21_32 to (i_t*64+32, 0) ------------------
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_21_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # ------------------ Zero out the upper-right 32x32 block (rows 0~31, cols 32~63) ------------------
    # offs_m = i_t * 64 + tl.arange(0, 32)
    # offs_n = 32 + tl.arange(0, 32)
    # mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < BT)  # BT=64
    # ptr_Ai = Ai + offs_m[:, None] * (H * BT) + offs_n[None, :]
    # zero_block = tl.zeros((32, 32), dtype=ptr_Ai.dtype.element_ty)
    # tl.store(ptr_Ai, zero_block, mask=mask_store)


def solve_tril_npu(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices_out: Dict[str, Optional[torch.Tensor]] = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Compute the inverse of the lower triangular matrix A should be strictly
    lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, K]
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor.
            Default: None.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64, 128]

    B, T, H, BT = A.shape
    Ad = torch.empty(
        B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype
    )

    LARGE_BLOCK_T = 608 * 2
    # assert A.shape[1]%LARGE_BLOCK_T == 0 # or last N_BLOCKS have not enough block which leads to tl.arange failed
    # LARGE_BLOCK_T = BT
    chunk_indices = (chunk_indices_out[str(LARGE_BLOCK_T)] if cu_seqlens is not None else None)
    # chunk_indices = (
    #     prepare_chunk_indices(cu_seqlens, LARGE_BLOCK_T)
    #     if cu_seqlens is not None
    #     else None
    # )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, LARGE_BLOCK_T)
    solve_tril_16x16_kernel_paral_v3[NT, B * H](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        LARGE_BLOCK_T=LARGE_BLOCK_T,
        # num_warps=1,
        # num_stages=4,
    )

    if BT == 16:
        return Ad

    Ai = torch.zeros(
        B, T, H, 32, device=A.device, dtype=torch.float if BT != 32 else output_dtype
    )
    merge_fn = (
        merge_16x16_to_32x32_inverse_kernel
        if BT == 32
        else merge_16x16_to_64x64_inverse_kernel_reorder_all_masked
    )
    # chunk_indices = (
    #     prepare_chunk_indices(cu_seqlens, 32 if BT == 32 else 64) if cu_seqlens is not None else None
    # )
    # NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 32 if BT == 32 else 64)
    # breakpoint()
    
    # merge_fn[NT, B * H](
    #     A=A,
    #     Ad=Ad,
    #     Ai=Ai,
    #     cu_seqlens=cu_seqlens,
    #     chunk_indices=chunk_indices,
    #     T=T,
    #     H=H,
    #     BT=32 if BT == 32 else 64,
    #     # BT=BT,
    #     # num_warps=4,
    #     # num_stages=3,
    # )

    chunk_indices = (chunk_indices_out[str(32)] if cu_seqlens is not None else None)
    # chunk_indices = (
    #     # prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    #     prepare_chunk_indices(cu_seqlens, 32) if cu_seqlens is not None else None
    # )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 32)
    # print(A.shape)
    merge_16x16_to_32x32_inverse_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    if BT == 32:
        return Ai

    Ad = Ai
    Ai = torch.zeros(
        B, T, H, 64, device=A.device, dtype=torch.float if BT != 64 else output_dtype
    )
    chunk_indices = (chunk_indices_out[str(64)] if cu_seqlens is not None else None)
    # chunk_indices = (
    #     # prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    #     prepare_chunk_indices(cu_seqlens, 64) if cu_seqlens is not None else None
    # )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 64)
    merge_32x32_to_64x64_inverse_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    if BT == 64:
        return Ai
    
    assert BT == 128
    Aii = torch.zeros_like(A, device=A.device, dtype=output_dtype)
    chunk_indices = (chunk_indices_out[str(128)] if cu_seqlens is not None else None)
    # chunk_indices = (
    #     prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    # )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    merge_64x64_to_128x128_inverse_kernel[NT, B * H](
        A=A,
        Ad=Ai,
        Ai=Aii,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        # num_warps=4,
        # num_stages=3,
    )

    return Aii



if __name__ == "__main__":

    # 参数配置
    # B, H, T, K, V, BT = 1, 32, 65536, 128, 128, 64
    # B, H, T, K, V, BT = 1, 32, 209, 128, 128, 64
    B, H, T, K, V, BT = 1, 32, 65536, 128, 128, 128
    # B, H, T, K, V, BT = 1, 1, 209, 1, 1, 128
    # B, H, T, K, V, BT = 1, 32, 209, 128, 1, 64
    # B, H, T, K, V, BT = 1, 1, 128, 128, 1, 64
    device = "npu:0"
    
    # 设置随机种子以保证可复现性
    torch.manual_seed(42)
    
    # 统一生成随机数据
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    # v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    A = torch.randn(B, T, H, BT, device=device, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    data = [0,   112,   209,   240,   281,   489,   523,   566,   689,   721,
          785,   837,   985,  1071,  1121,  1186,  1255,  1328,  1449,  1592,
         1733,  1766,  1830,  1870,  2054,  2181,  2219,  2332,  2560,  2690,
         2855,  2955,  3012,  3044,  3835,  3910,  3964,  4005,  4050,  4079,
         4110,  4240,  4308,  4410,  4552,  4684,  4763,  4805,  4862,  5006,
         5112,  5133,  5293,  5443,  5512,  5606,  5728,  5794,  5966,  6035,
         6170,  6272,  6470,  6598,  6691,  6747,  6819,  6860,  6884,  7129,
         7164,  7218,  7320,  7411,  7500,  7561,  7626,  7679,  7769,  7809,
         7893,  7951,  8000,  8092,  8209,  8305,  8343,  8392,  8451,  8509,
         8761,  8929,  9014,  9130,  9182,  9211,  9245,  9278,  9331,  9476,
         9536,  9575,  9627,  9785,  9899,  9982, 10080, 10119, 10192, 10270,
        10305, 10384, 10455, 10595, 10683, 10721, 10767, 11032, 11148, 11297,
        11388, 11533, 11588, 11643, 11723, 11857, 11915, 11959, 12229, 12282,
        12310, 12382, 12487, 12543, 12638, 12678, 12730, 12812, 12951, 13000,
        13126, 13165, 13236, 13306, 13436, 13514, 13541, 13677, 13729, 13845,
        13872, 13962, 14079, 14175, 14205, 14325, 14406, 14434, 14555, 14596,
        14662, 14728, 14883, 14916, 14999, 15092, 15226, 15257, 15453, 15550,
        15771, 15896, 15965, 16009, 16066, 16136, 16254, 16488, 16649, 16724,
        16846, 16970, 17142, 17185, 17304, 17340, 17514, 17577, 17689, 17867,
        17909, 17948, 17977, 18050, 18153, 18218, 18311, 18365, 18419, 18443,
        18473, 18724, 18865, 19017, 19061, 19226, 19265, 19349, 19454, 19494,
        19518, 19578, 19642, 19702, 19848, 19884, 20027, 20074, 20273, 20443,
        20501, 20583, 20729, 20796, 20818, 20866, 20911, 21157, 21196, 21259,
        21312, 21471, 21647, 21750, 21870, 21920, 21952, 21982, 22027, 22141,
        22183, 22287, 22423, 22470, 22664, 22764, 22859, 23027, 23142, 23241,
        23330, 23377, 23441, 23574, 23603, 23727, 23859, 23956, 24001, 24107,
        24302, 24398, 24476, 24545, 24614, 24659, 24854, 24987, 25309, 25352,
        25548, 25627, 25744, 25837, 25869, 25913, 26004, 26044, 26076, 26100,
        26227, 26291, 26344, 26515, 26547, 26593, 26625, 26698, 26726, 26847,
        26964, 27136, 27327, 27357, 27440, 27540, 27571, 27672, 27718, 27872,
        27913, 28010, 28068, 28106, 28270, 28318, 28363, 28508, 28603, 28686,
        28722, 28754, 28793, 28835, 28867, 28928, 28975, 29064, 29107, 29194,
        29231, 29265, 29449, 29519, 29560, 29608, 29652, 29758, 29794, 29851,
        29887, 29981, 30058, 30118, 30306, 30445, 30580, 30621, 30645, 30691,
        31252, 31332, 31362, 31501, 31566, 31641, 31675, 31748, 31805, 31927,
        32000, 32081, 32143, 32235, 32359, 32405, 32505, 32683, 32718, 32803,
        32902, 32942, 33006, 33109, 33227, 33316, 33370, 33475, 33505, 33554,
        33627, 33703, 33749, 33926, 33962, 34392, 34552, 34585, 34616, 34654,
        34723, 34887, 34926, 34964, 35015, 35169, 35251, 35307, 35485, 35604,
        35642, 35819, 35930, 35959, 36102, 36256, 36348, 36550, 36585, 36621,
        36696, 36834, 36912, 36957, 36983, 37037, 37152, 37298, 37327, 37364,
        37474, 37522, 37550, 37669, 37718, 37778, 37807, 37909, 37979, 38086,
        38179, 38291, 38385, 38417, 38545, 38661, 38828, 38883, 39006, 39051,
        39083, 39126, 39202, 39236, 39331, 39414, 39541, 39566, 39661, 39715,
        39759, 39796, 39840, 39881, 39911, 39963, 40083, 40124, 40258, 40338,
        40378, 40427, 40487, 40680, 40783, 40852, 40897, 40947, 40987, 41069,
        41099, 41131, 41214, 41247, 41375, 41414, 41449, 41543, 41600, 41685,
        41721, 41883, 41991, 42086, 42233, 42265, 42302, 42356, 42445, 42504,
        42534, 42653, 42693, 42778, 42828, 42937, 43030, 43209, 43351, 43530,
        43573, 43603, 43651, 43731, 43777, 43886, 43925, 44068, 44112, 44163,
        44204, 44301, 44476, 44616, 44689, 44716, 44847, 44917, 45020, 45151,
        45216, 45309, 45370, 45524, 45628, 45669, 45813, 45846, 45931, 45963,
        46111, 46386, 46484, 46554, 46599, 46631, 46704, 46882, 47012, 47087,
        47158, 47262, 47359, 47458, 47497, 47526, 47807, 47959, 48162, 48267,
        48298, 48348, 48389, 48482, 48600, 48714, 48806, 48856, 48893, 49008,
        49070, 49158, 49302, 49364, 49572, 49618, 49653, 49750, 49782, 49852,
        49883, 49910, 49944, 50014, 50142, 50220, 50301, 50338, 50448, 50503,
        50699, 50808, 50947, 51093, 51178, 51221, 51292, 51314, 51363, 51428,
        51497, 51639, 51790, 51874, 51968, 52068, 52105, 52221, 52270, 52340,
        52453, 52527, 52623, 52679, 52801, 52844, 52893, 52977, 53074, 53183,
        53222, 53258, 53348, 53417, 53541, 53579, 53713, 53813, 53843, 53881,
        53930, 54070, 54106, 54249, 54317, 54351, 54392, 54483, 54513, 54577,
        54980, 55029, 55182, 55253, 55284, 55342, 55446, 55680, 55717, 55747,
        55772, 55811, 56027, 56098, 56135, 56229, 56270, 56332, 56370, 56405,
        56499, 56730, 56919, 57037, 57081, 57189, 57220, 57332, 57372, 57409,
        57459, 57520, 57599, 57698, 57770, 57948, 58053, 58184, 58300, 58414,
        58491, 58577, 58671, 58715, 58793, 58838, 58866, 59001, 59041, 59096,
        59220, 59273, 59323, 59418, 59467, 59532, 59638, 59762, 59832, 59959,
        60107, 60164, 60316, 60421, 60553, 60621, 60823, 60875, 60934, 61021,
        61058, 61109, 61141, 61234, 61322, 61479, 61505, 61634, 61765, 61805,
        61855, 62014, 62064, 62148, 62247, 62303, 62392, 62530, 62575, 62615,
        62760, 62884, 62984, 63025, 63085, 63138, 63167, 63355, 63388, 63495,
        63604, 63681, 63767, 64013, 64051, 64189, 64321, 64459, 64501, 64620,
        64682, 64712, 64798, 64838, 64869, 65066, 65222, 65357, 65535, 65536]
    # data = data[:3]
    # data = data[::15] + [data[-1]]
    # data = [0, 64, 128]
    cu_seqlens = torch.tensor(data, dtype=torch.long, device=device)
    # 调用测试函数
    # g = test_kkt(k, g, beta, cu_seqlens, BT)
    # A = solve_tril(A,cu_seqlens)

    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    A_list = []
    ref_list = []
    for t in seqlens:
        k = F.normalize(torch.randn((B, H, t, K), dtype=torch.float32, device=device), dim=-1)
        padding_size = (BT - t % BT) % BT
        k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
        k_padded = k_padded.reshape(B, H, -1, BT, K)
        A_part = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)

        ref_part = torch.inverse(A_part + torch.eye(A_part.shape[-1], device=A_part.device)[None, None, None, ...])
        ref_part = ref_part.reshape(B, H, -1, BT)[:, :, :t, :]
        ref_list.append(ref_part)

        A_part = A_part.reshape(B, H, -1, BT)[:, :, :t, :]
        A_list.append(A_part)

    A = torch.cat(A_list, dim=2).transpose(1, 2).contiguous()
    ref = torch.cat(ref_list, dim=2).transpose(1, 2).contiguous()

    # breakpoint()
    tt_A = solve_tril_npu(A,cu_seqlens)
    print(triton.testing.do_bench(lambda: solve_tril_npu(A,cu_seqlens)))
    # tt_V_A = solve_tril(A, cuseqlens)
    # breakpoint()
    torch.testing.assert_close(tt_A, ref, atol=0.001, rtol=0.001)
