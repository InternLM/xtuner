"""Check FSDP2 checkpoint composition and backward-prefetch bookkeeping.

Run on two GPUs, for example::

    torchrun --standalone --nproc-per-node=2 \
        .dev_scripts/repro_fsdp_checkpoint_prefetch.py --mode reentrant
    torchrun --standalone --nproc-per-node=2 \
        .dev_scripts/repro_fsdp_checkpoint_prefetch.py --mode non-reentrant

The model intentionally has no MoE, compile, explicit forward prefetch, or large
activations. It isolates the default FSDP2 backward-prefetch bookkeeping from
the Qwen3.5-VL async-RL case; an outer checkpoint wrapper should keep the
logical post-forward order equal to the number of layers in both modes.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from xtuner.v1.model.utils import apply_gradient_checkpointing


class Block(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.grad_modes: list[bool] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.grad_modes.append(torch.is_grad_enabled())
        return hidden_states + self.down(torch.nn.functional.silu(self.up(hidden_states)))


class Model(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block(hidden_size) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


@dataclass
class Observation:
    max_pending_groups: int = 0
    max_pending_bytes: int = 0
    max_post_forward_order: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reentrant", "non-reentrant"), required=True)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    torch.manual_seed(0)
    model = Model(args.num_layers, args.hidden_size).cuda()
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    use_reentrant = args.mode == "reentrant"

    for index, layer in enumerate(model.layers):
        layer = apply_gradient_checkpointing(layer, use_reentrant=use_reentrant)
        model.layers[index] = layer
        fully_shard(layer, mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)

    observation = Observation()

    def observe_fsdp_state(_module: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
        # Synchronize only in this diagnostic script so async all-gathers have a
        # stable lifetime at the point where we account for their storage.
        torch.cuda.synchronize()
        param_groups = [fully_shard.state(layer)._fsdp_param_group for layer in model.layers]
        pending = [group._all_gather_result for group in param_groups]
        pending = [result for result in pending if result is not None]
        pending_bytes = sum(result.all_gather_output.nbytes for result in pending)
        comm_ctx = param_groups[0].comm_ctx
        observation.max_pending_groups = max(observation.max_pending_groups, len(pending))
        observation.max_pending_bytes = max(observation.max_pending_bytes, pending_bytes)
        observation.max_post_forward_order = max(
            observation.max_post_forward_order,
            len(comm_ctx.post_forward_order),
        )

    # Register after fully_shard so this observer runs after FSDP's post-forward
    # hook. It measures real FSDP state without replacing any production method.
    handles = [layer.register_forward_hook(observe_fsdp_state) for layer in model.layers]
    inputs = torch.randn(2, 8, args.hidden_size, device="cuda", requires_grad=True)
    model(inputs).float().square().mean().backward()
    torch.cuda.synchronize()

    if dist.get_rank() == 0:
        grad_modes = [layer.grad_modes for layer in model.layers]
        print(f"mode={args.mode}")
        print(f"grad_modes={grad_modes}")
        print(f"max_post_forward_order={observation.max_post_forward_order}")
        print(f"max_pending_groups={observation.max_pending_groups}")
        print(f"max_pending_bytes={observation.max_pending_bytes}")

    for handle in handles:
        handle.remove()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
