"""Single-GPU repro for the 512K SP16 backward slice in the failed run."""

import sys

import torch


tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 512 * 1024
shard = tokens // 16
channels = 512
projection_width = int(sys.argv[2]) if len(sys.argv) > 2 else 8192


@torch.compile
def forward(base: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
    # The last SP shard consumes a 512-channel view of an 8192-wide projection.
    return (base[-shard:] * projected[:, -channels:]).sum()


base = torch.zeros((tokens, channels), device="cuda", dtype=torch.bfloat16, requires_grad=True)
projected = torch.ones((shard, projection_width), device="cuda", dtype=torch.bfloat16)
print(f"torch={torch.__version__}, tokens={tokens}, shard={shard}, projected_stride={projected.stride()}", flush=True)
forward(base, projected).backward()
print(f"backward succeeded, grad_shape={tuple(base.grad.shape)}", flush=True)
