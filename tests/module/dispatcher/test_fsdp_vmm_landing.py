import unittest

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.module.dispatcher.fsdp_vmm_landing import (
    accumulate_fsdp_unsharded_expert_gradients,
    fsdp_current_unsharded_expert_parameters,
    install_fsdp_vmm_landing,
    uninstall_fsdp_vmm_landing,
)


class _ExpertLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fused_w1w3 = nn.Linear(8, 16, bias=False)
        self.fused_w2 = nn.Linear(8, 8, bias=False)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 CUDA devices")
class TestFSDPVMMDirectLanding(DeterministicDDPTestCase):
    def test_installation_is_atomic_and_preserves_the_unsharded_contract(self) -> None:
        self.create_pg("cuda")
        root = nn.ModuleList([_ExpertLayer(), _ExpertLayer()]).cuda()
        mesh = init_device_mesh("cuda", (2,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        for layer in root:
            fully_shard(layer.fused_w1w3, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
            fully_shard(layer.fused_w2, mesh=mesh, mp_policy=policy, reshard_after_forward=False)

        targets = []
        for layer_idx, layer in enumerate(root):
            projections = (layer.fused_w1w3, layer.fused_w2)
            landings = tuple(
                torch.empty(projection.weight.shape, dtype=torch.bfloat16, device="cuda") for projection in projections
            )
            targets.append((f"layers.{layer_idx}.experts", projections, landings))

        # A failure in the last target must leave the earlier, valid targets
        # untouched.  A corrected retry through the public API proves that no
        # binding attributes or method overrides leaked from validation.
        invalid_targets = [*targets]
        last_fqn, last_projections, last_landings = invalid_targets[-1]
        invalid_targets[-1] = (
            last_fqn,
            last_projections,
            (last_landings[0], torch.empty(last_landings[1].shape, dtype=torch.float32, device="cuda")),
        )
        with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
            install_fsdp_vmm_landing(fsdp_root=root, targets=invalid_targets)

        fsdp_params = install_fsdp_vmm_landing(fsdp_root=root, targets=targets)
        assert len(fsdp_params) == 4
        for layer in root:
            layer.fused_w1w3.unshard()
            layer.fused_w2.unshard()
            parameters = fsdp_current_unsharded_expert_parameters((layer.fused_w1w3, layer.fused_w2))
            gradients = tuple(
                torch.ones_like(parameter.to_local() if isinstance(parameter, DTensor) else parameter)
                for parameter in parameters
            )
            accumulate_fsdp_unsharded_expert_gradients(parameters, gradients)
            for parameter in parameters:
                gradient = parameter.grad
                assert gradient is not None
                gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
                assert gradient.dtype is torch.bfloat16
                torch.testing.assert_close(gradient, torch.ones_like(gradient), rtol=0, atol=0)

        uninstall_fsdp_vmm_landing(fsdp_params)
        with self.assertRaisesRegex(RuntimeError, "is not installed"):
            fsdp_current_unsharded_expert_parameters((root[0].fused_w1w3, root[0].fused_w2))

    @property
    def world_size(self) -> int:
        return 2
