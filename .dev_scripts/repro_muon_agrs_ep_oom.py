#!/usr/bin/env python3
"""Reproduce the Muon AGRS EP/FSDP memory amplification.

The production failure can be reduced to two facts:

* a ``fused_w2`` parameter is sharded on the same tensor dimension by EP and
  FSDP, so one rank owns ``global_rows / (ep * fsdp)`` rows;
* ``_agrs_orthogonalize`` receives the DTensor's global size instead of the
  EP-local global size and pads each local shard to ``global_rows / fsdp``.

The ``--distributed`` mode calls the production AGRS function on a tiny
two-dimensional mesh.  The arithmetic report uses the GLM-5.2 dimensions and
does not allocate the 72 GiB tensors.

The arithmetic/layout check runs on CPU.  The distributed check must use CUDA:
Gloo rejects the flattened ``reduce_scatter_tensor`` output shape used by the
production NCCL path.

Examples (single-process arithmetic/layout check):

    python .dev_scripts/repro_muon_agrs_ep_oom.py

Example (distributed CUDA check, four local ranks):

    torchrun --standalone --nproc_per_node=4 \
        .dev_scripts/repro_muon_agrs_ep_oom.py --distributed --device cuda
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from xtuner.v1.optim.muon import _agrs_orthogonalize, zeropower_via_newtonschulz5


def gibibytes(num_bytes: int) -> float:
    return num_bytes / 2**30


def report_glm_shape() -> None:
    experts = 256
    rows = experts * 6_144  # GLM-5.2 fused_w2
    cols = 2_048
    ep = 8
    fsdp = 64  # 512 world size / EP 8
    param_count = 76  # 75 decoder MoE layers + one shared MTP layer
    remainder = param_count % fsdp
    local_rows = rows // (ep * fsdp)
    ep_local_global_rows = rows // ep
    buggy_padded_rows = math.ceil(rows / fsdp)
    correct_padded_rows = math.ceil(ep_local_global_rows / fsdp)
    element_size = 2  # bf16

    buggy_buffer = fsdp * remainder * buggy_padded_rows * cols * element_size
    correct_buffer = fsdp * remainder * correct_padded_rows * cols * element_size

    print("GLM-5.2 AGRS shape arithmetic")
    print(f"  parameters={param_count}, fsdp={fsdp}, remainder R={remainder}, ep={ep}")
    print(
        f"  DTensor global shape=({rows}, {cols}), "
        f"EP-local global rows={ep_local_global_rows}, effective local rows={local_rows}"
    )
    print(f"  buggy padded shard rows={buggy_padded_rows}, correct={correct_padded_rows}")
    print(f"  buggy ag_output shape=({fsdp * remainder}, {buggy_padded_rows}, {cols})")
    print(f"  buggy full_params shape=({remainder}, {fsdp * buggy_padded_rows}, {cols})")
    print(f"  buggy ag_output bytes={gibibytes(buggy_buffer):.1f} GiB")
    print(f"  buggy full_params copy bytes={gibibytes(buggy_buffer):.1f} GiB")
    print(f"  corrected each buffer bytes={gibibytes(correct_buffer):.1f} GiB")

    # ``permute(...).flatten(...)`` is a copy when R > 1.  Keep this small,
    # but use the same layout as the production code.
    w, r, padded, n = 4, 3, 5, 3
    gathered = torch.empty((w * r, padded, n), dtype=torch.bfloat16)
    reshaped = gathered.view(w, r, padded, n)
    permuted = reshaped.permute(1, 0, 2, 3)
    flattened = permuted.flatten(1, 2)
    print(
        "  flatten copy: "
        f"permuted_contiguous={permuted.is_contiguous()}, "
        f"same_storage={permuted.data_ptr() == flattened.data_ptr()}, "
        f"copy_bytes={flattened.untyped_storage().nbytes()}"
    )


def run_distributed(device_name: str) -> None:
    if device_name != "cuda":
        raise ValueError("The distributed check uses NCCL; pass --device cuda")

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"run this check with exactly 4 ranks, got {world_size}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    mesh_device = "cuda"

    if rank == 0:
        report_glm_shape()

    # This is the smallest EP+FSDP mesh: fsdp=2, ep=2.
    mesh = init_device_mesh(
        mesh_device,
        (2, 2),
        mesh_dim_names=("default.fsdp", "default.ep"),
    )
    fsdp_pg = mesh.get_group(0)

    fsdp_size, ep = mesh.size(0), mesh.size(1)
    global_rows, cols = 32, 8
    local_rows = global_rows // (fsdp_size * ep)
    local = torch.arange(local_rows * cols, dtype=torch.bfloat16, device=device).reshape(local_rows, cols)
    local.add_(rank * 1000)
    param = DTensor.from_local(
        local,
        mesh,
        (Shard(0), Shard(0)),
        run_check=False,
    )

    # R=1 is enough to exercise the real collective path.  The production
    # R=12 copy is covered by the arithmetic/layout check above.
    U = [local.clone()]

    def drive(global_size: int):
        ns_input_shapes: list[tuple[int, ...]] = []

        def tracked_newton_schulz(x, epsilon, num_experts):
            ns_input_shapes.append(tuple(x.shape))
            return zeropower_via_newtonschulz5(x, epsilon=epsilon, num_experts=num_experts)

        generator = _agrs_orthogonalize(
            U=[u.clone() for u in U],
            shard_dim=0,
            global_shard_dim_size=global_size,
            process_group=fsdp_pg,
            newton_schulz_func=tracked_newton_schulz,
            flatten=False,
            epsilon=torch.tensor(1e-8, device=device),
            # With this 2-way EP toy, each FSDP group owns two complete experts.
            num_experts=2,
        )
        while True:
            try:
                next(generator)
            except StopIteration as stop:
                return stop.value, ns_input_shapes

    buggy_global_size = param.size(0)
    corrected_global_size = buggy_global_size // ep
    buggy, buggy_ns_shapes = drive(buggy_global_size)
    corrected, corrected_ns_shapes = drive(corrected_global_size)
    if rank == 0:
        print("distributed AGRS mechanism")
        print(f"  DTensor shape={tuple(param.shape)}, local shape={tuple(param.to_local().shape)}")
        print(f"  placements={param.placements}")
        print(f"  fsdp group size={fsdp_pg.size()}, EP size={ep}")
        print(f"  buggy global_size={buggy_global_size}: output local shape={tuple(buggy[0].shape)}")
        print(f"  corrected global_size={corrected_global_size}: output local shape={tuple(corrected[0].shape)}")
        print(f"  buggy Newton-Schulz input shape={buggy_ns_shapes[0]}")
        print(f"  corrected Newton-Schulz input shape={corrected_ns_shapes[0]}")
        print(
            "  toy_output_difference="
            f"{torch.max(torch.abs(buggy[0] - corrected[0])).item():.3e} "
            "(the allocation shapes, not this toy value, demonstrate the bug)"
        )

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.distributed:
        if args.device != "cuda":
            parser.error("--distributed requires --device cuda (the production AGRS collective uses NCCL)")
        run_distributed(args.device)
    else:
        report_glm_shape()


if __name__ == "__main__":
    main()
