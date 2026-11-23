"""UltraEP/FSDP invariant coverage independent of the C-5 post-backward gate.

Covers the plan invariants whose guarantees come from existing seams:

- INV-3: ``UltraEPLayerRuntime.start_grad_reduce`` refuses to stage master
  gradients that autograd has not produced yet.
- INV-6: the master-weight view saved by ``UltraEPGroupedGemm`` stays valid
  until backward, under both ``reshard_after_forward`` settings.
- INV-10: the UltraEP path runs uncompiled (no dynamo wrapper on the MoE
  decoder layers, no compiled autograd around the GEMM backward).
- INV-11: ``validate_before_fsdp`` rejects non-BF16 param/reduce dtypes, CPU
  offload, missing ``requires_grad``, an open DP>1 unshard window, and
  activation recompute.

The ordering and state-machine invariants (INV-1/2/4/5/12) arrive with the
C-5 gate and extend this file.
"""

import importlib.util
import os
import unittest

import pytest
import torch
import torch.distributed as dist
from torch import nn

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.module.ultraep import UltraEPConfig
from xtuner.v1.module.ultraep.runtime import UltraEPLayerRuntime, UltraEPModelRuntime
from xtuner.v1.ops.moe.cuda.group_gemm import UltraEPGroupedGemm

from .test_moe_train_engine_tpep import (
    _build_tiny_moe_cfg,
    _make_engine_input,
    _run_train_step_items_without_clip,
)


@pytest.mark.gpu
class TestUltraEPConfigGuards(DeterministicDDPTestCase):
    """Guard checks that need no model and no native UltraEP manager."""

    @property
    def world_size(self) -> int:
        return 2

    def _make_model_runtime(self) -> UltraEPModelRuntime:
        return UltraEPModelRuntime(
            group=dist.group.WORLD,
            num_model_layers=2,
            num_logical_experts=16,
            hidden_size=64,
            expert_intermediate_size=32,
            num_redundant_experts_per_rank=1,
            max_microbatches=1,
        )

    @staticmethod
    def _fsdp_config(**overrides) -> FSDPConfig:
        values = dict(ep_size=2, recompute_ratio=0, reshard_after_forward=True, torch_compile=False)
        values.update(overrides)
        return FSDPConfig(**values)

    def test_validate_before_fsdp_guards(self) -> None:
        """INV-11 and its sibling seams reject unsupported FSDP settings."""
        self.create_pg("cuda")
        try:
            runtime = self._make_model_runtime()
            # INV-11: the BF16 handoff writes replica contributions straight
            # into the parameter .grad; any other dtype moves the gradient
            # into FSDP's fp32 accumulation path and the write-back is lost.
            with self.assertRaisesRegex(ValueError, "param_dtype=reduce_dtype"):
                runtime.validate_before_fsdp(self._fsdp_config(reduce_dtype=torch.float32))
            with self.assertRaisesRegex(ValueError, "param_dtype=reduce_dtype"):
                runtime.validate_before_fsdp(self._fsdp_config(param_dtype=torch.float32))
            # CPU offload moves the storages the seam points at.
            with self.assertRaisesRegex(ValueError, "CPU offload"):
                runtime.validate_before_fsdp(self._fsdp_config(cpu_offload=True))
            # Recompute replays forward outside the UltraEP call protocol.
            with self.assertRaisesRegex(ValueError, "recompute"):
                runtime.validate_before_fsdp(self._fsdp_config(recompute_ratio=0.5))
            with self.assertRaisesRegex(ValueError, "requires_grad"):
                runtime.validate_before_fsdp(self._fsdp_config(requires_grad=False))
            # DP>1 without reshard leaves the FSDP unshard window open.
            with self.assertRaisesRegex(ValueError, "reshard_after_forward"):
                runtime.validate_before_fsdp(self._fsdp_config(ep_size=1, reshard_after_forward=False))
            # The supported configuration passes.
            runtime.validate_before_fsdp(self._fsdp_config())
        finally:
            dist.destroy_process_group()

    def test_start_grad_reduce_requires_master_grads(self) -> None:
        """INV-3: grad-reduce launch requires autograd-produced master grads."""
        self.create_pg("cuda")
        try:
            runtime = self._make_model_runtime()
            fused_w1w3 = nn.Linear(4, 4)
            fused_w2 = nn.Linear(4, 4)
            layer = UltraEPLayerRuntime(
                layer_id=0,
                model_runtime=runtime,
                fused_w1w3=fused_w1w3,
                fused_w2=fused_w2,
            )
            with self.assertRaisesRegex(RuntimeError, "master gradients are unavailable"):
                layer.start_grad_reduce(0)
            # A missing fc2 grad alone must also fail: partial staging would
            # silently drop one projection's replica contribution.
            fused_w1w3.weight.grad = torch.zeros_like(fused_w1w3.weight)
            with self.assertRaisesRegex(RuntimeError, "master gradients are unavailable"):
                layer.start_grad_reduce(0)
        finally:
            dist.destroy_process_group()


@pytest.mark.gpu
@unittest.skipUnless(
    os.getenv("XTUNER_TEST_ULTRA_EP") == "1"
    and torch.cuda.device_count() >= 8
    and importlib.util.find_spec("ultra_ep") is not None
    and importlib.util.find_spec("deep_ep") is not None,
    "Set XTUNER_TEST_ULTRA_EP=1 with eight GPUs and the patched native UltraEP extra",
)
class TestUltraEPForwardInvariants(DeterministicDDPTestCase):
    """Backward-time invariants that need a real UltraEP engine."""

    @property
    def world_size(self) -> int:
        return 8

    def test_master_weight_view_survives_backward_reshard_ep4(self) -> None:
        self._check_saved_master_weight(ep_size=4, reshard_after_forward=True)

    def test_master_weight_view_survives_backward_no_reshard_ep8(self) -> None:
        self._check_saved_master_weight(ep_size=8, reshard_after_forward=False)

    def _build_engine(self, ep_size: int, reshard_after_forward: bool) -> TrainEngine:
        config = _build_tiny_moe_cfg(ep_size=ep_size)
        config.n_routed_experts = 16
        config.gate_bias = True
        config.dispatcher = "deepep"
        config.mesh_prefix = "ultraep_inv_test"
        config.ultraep_cfg = UltraEPConfig(num_redundant_experts_per_rank=1)
        return TrainEngine(
            model_cfg=config,
            optim_cfg=AdamWConfig(lr=1e-3, eps=1e-4),
            fsdp_cfg=FSDPConfig(
                ep_size=ep_size,
                recompute_ratio=0,
                reshard_after_forward=reshard_after_forward,
                torch_compile=False,
                mesh_prefix=config.mesh_prefix,
            ),
            intra_layer_micro_batch=1,
        )

    def _check_saved_master_weight(self, ep_size: int, reshard_after_forward: bool) -> None:
        os.environ["XTUNER_GROUP_GEMM"] = "triton"
        self.create_pg("cuda")
        engine = self._build_engine(ep_size, reshard_after_forward)
        errors: list[str] = []
        hook_calls = [0]
        original_backward = UltraEPGroupedGemm.backward

        def checked_backward(ctx, grad_output):
            hook_calls[0] += 1
            _x, master_weight, _tokens_per_expert = ctx.saved_tensors
            if master_weight.untyped_storage().nbytes() == 0 or master_weight.numel() == 0:
                # INV-6: DGrad reads this saved view; a freed storage means
                # FSDP reshard invalidated it between forward and backward.
                errors.append(f"rank {dist.get_rank()}: saved master_weight storage was freed before backward")
            if torch.compiler.is_compiling():
                # INV-10: this path is only validated uncompiled.
                errors.append(f"rank {dist.get_rank()}: UltraEPGroupedGemm.backward ran under torch.compile")
            return original_backward(ctx, grad_output)

        UltraEPGroupedGemm.backward = checked_backward
        try:
            engine.init_model_weights()
            # INV-10 premise: the MoE decoder layers stay uncompiled so the
            # UltraEP autograd boundaries keep their ordering guarantees.
            for layer in engine.model.layers.values():
                if hasattr(layer, "_orig_mod"):
                    errors.append(f"rank {dist.get_rank()}: MoE decoder layer is dynamo-wrapped")
            device = torch.device("cuda", dist.get_rank())
            batches = [_make_engine_input(device, seed_offset=8 * dist.get_rank() + index) for index in range(2)]
            _run_train_step_items_without_clip(engine, CELossConfig(), batches)
            if hook_calls[0] == 0:
                errors.append(f"rank {dist.get_rank()}: UltraEPGroupedGemm.backward never ran")
            self._assert_all_ranks_match(errors)
        finally:
            UltraEPGroupedGemm.backward = original_backward
            engine.close()
            # See test_moe_train_engine_ultraep.py: finalize the process-wide
            # native runtime collectively before module teardown.
            manager = engine.model._ep_runtime._manager
            if manager is not None:
                manager.runtime.runtime.destroy()
            dist.destroy_process_group()

    def _assert_all_ranks_match(self, errors: list[str]) -> None:
        # Fail every worker together: a rank-local assertion must not leave
        # its peers entering the next native collective during teardown.
        gathered: list[list[str] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, errors)
        for rank, failures in enumerate(gathered):
            if failures:
                raise AssertionError(failures[0])
