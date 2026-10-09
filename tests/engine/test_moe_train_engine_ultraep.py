"""Native UltraEP/FSDP regression, enabled with XTUNER_TEST_ULTRA_EP=1.

Requires eight CUDA GPUs, DeepEP, and UltraEP with both patches documented in
patch/ultraep/README.md. Each case uses fresh worker processes because UltraEP
owns a process-wide NVSHMEM runtime. No XTuner module is mocked.
"""

import importlib.util
import os
import unittest

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.module.ultraep import UltraEPConfig

from .test_moe_train_engine_tpep import (
    BF16_ATOL,
    BF16_GEMM_ATOL,
    BF16_RTOL,
    _build_tiny_moe_cfg,
    _make_engine_input,
    _run_train_step_items_without_clip,
)


@pytest.mark.gpu
@unittest.skipUnless(
    os.getenv("XTUNER_TEST_ULTRA_EP") == "1"
    and torch.cuda.device_count() >= 8
    and importlib.util.find_spec("ultra_ep") is not None
    and importlib.util.find_spec("deep_ep") is not None,
    "Set XTUNER_TEST_ULTRA_EP=1 with eight GPUs and the patched native UltraEP extra",
)
class TestMoETrainEngineUltraEP(DeterministicDDPTestCase):
    def test_ep8_matches_deepep(self) -> None:
        self._check_training(ep_size=8, microbatches=1)

    def test_ep4_dp2_multi_microbatch_matches_deepep(self) -> None:
        self._check_training(ep_size=4, microbatches=2)

    @property
    def world_size(self) -> int:
        return 8

    def _build_engine(self, ep_size: int, microbatches: int, *, ultraep: bool) -> TrainEngine:
        config = _build_tiny_moe_cfg(ep_size=ep_size)
        config.n_routed_experts = 16
        config.gate_bias = True
        config.dispatcher = "deepep"
        config.mesh_prefix = "ultraep_test" if ultraep else "deepep_reference"
        config.ultraep_cfg = UltraEPConfig(num_redundant_experts_per_rank=1) if ultraep else None
        return TrainEngine(
            model_cfg=config,
            optim_cfg=AdamWConfig(lr=1e-3, eps=1e-4),
            fsdp_cfg=FSDPConfig(
                ep_size=ep_size,
                recompute_ratio=0,
                reshard_after_forward=True,
                torch_compile=False,
                mesh_prefix=config.mesh_prefix,
            ),
            intra_layer_micro_batch=microbatches,
        )

    def _check_training(self, ep_size: int, microbatches: int) -> None:
        # Keep backend selection local to the worker process. The parent pytest
        # environment and other tests retain their own grouped-GEMM selection.
        os.environ["XTUNER_GROUP_GEMM"] = "triton"
        self.create_pg("cuda")
        reference = self._build_engine(ep_size, microbatches, ultraep=False)
        candidate = self._build_engine(ep_size, microbatches, ultraep=True)
        try:
            self._compare_training(reference, candidate, microbatches)
            print(f"rank {self.rank}: UltraEP/FSDP numerical checks passed", flush=True)
        finally:
            candidate.close()
            reference.close()
            # Model.close deliberately retains the process-wide native cache.
            # These isolated workers will never reuse it. Finalize the native
            # dependency collectively before Python tears down modules in an
            # unspecified order (its C++ destructor contains an NVSHMEM barrier).
            manager = candidate.model._ep_runtime._manager
            if manager is not None:
                manager.runtime.runtime.destroy()
            dist.destroy_process_group()

    def _compare_training(self, reference: TrainEngine, candidate: TrainEngine, microbatches: int) -> None:
        reference.init_model_weights()
        candidate.init_model_weights()
        with torch.no_grad():
            # Concentrate tokens on two experts with a clear routing margin.
            # Exact ties would make top-k discontinuous under BF16 roundoff
            # after the first optimizer update, confounding backend parity.
            for layer in reference.model.layers.values():
                layer.gate.weight.zero_()
                bias = layer.gate.bias
                values = torch.full(bias.shape, -1.0, device=bias.device, dtype=bias.dtype)
                values[0], values[1] = 1, 0.5
                bias.copy_(distribute_tensor(values, bias.device_mesh, bias.placements))
            reference_parameters = dict(reference.model.named_parameters())
            for name, parameter in candidate.model.named_parameters():
                # Equal sharding layouts, independent DeviceMesh objects.
                self._local(parameter).copy_(self._local(reference_parameters[name]))

        device = torch.device("cuda", dist.get_rank())
        loss_config = CELossConfig()
        for step in range(2):
            # Two backward calls per step also exercise gradient accumulation
            # and reuse of native microbatch slots before the optimizer update.
            batches = [
                _make_engine_input(device, seed_offset=100 * step + 8 * dist.get_rank() + index)
                for index in range(2 * microbatches)
            ]
            # DeepEP keeps a process-wide buffer sized on first dispatch.
            # Initialize it with UltraEP's larger physical-expert count.
            loss = _run_train_step_items_without_clip(candidate, loss_config, batches)
            norm = candidate.clip_grad_norm(do_clip=False)
            ref_loss = _run_train_step_items_without_clip(reference, loss_config, batches)
            ref_norm = reference.clip_grad_norm(do_clip=False)
            errors: list[str] = []
            try:
                torch.testing.assert_close(torch.tensor(loss), torch.tensor(ref_loss), atol=BF16_ATOL, rtol=BF16_RTOL)
                torch.testing.assert_close(norm, ref_norm, atol=BF16_ATOL, rtol=BF16_RTOL)
            except AssertionError as error:
                errors.append(f"step {step} loss/norm: {error}")

            ref_params = dict(reference.model.named_parameters())
            for name, parameter in candidate.model.named_parameters():
                expected = ref_params[name]
                if parameter.grad is None or expected.grad is None:
                    errors.append(f"step {step} {name}: missing gradient")
                    continue
                actual_grad = self._local(parameter.grad)
                expected_grad = self._local(expected.grad)
                try:
                    assert torch.isfinite(actual_grad).all(), name
                    torch.testing.assert_close(actual_grad, expected_grad, atol=BF16_GEMM_ATOL, rtol=BF16_RTOL)
                    # Keep small tensors constrained too: elementwise absolute
                    # tolerance alone could hide a missing replica gradient.
                    torch.testing.assert_close(
                        actual_grad.float().norm(), expected_grad.float().norm(), atol=BF16_ATOL, rtol=BF16_RTOL
                    )
                except AssertionError as error:
                    errors.append(f"step {step} {name} gradient: {error}")
            self._assert_all_ranks_match(errors)

            reference.step_optimizer(ref_norm)
            candidate.step_optimizer(norm)
            # Use the existing BF16 GEMM parity tolerance: cancellation in
            # gradients and Adam's normalization affect near-zero parameters.
            for name, parameter in candidate.model.named_parameters():
                try:
                    torch.testing.assert_close(
                        self._local(parameter), self._local(ref_params[name]), atol=BF16_GEMM_ATOL, rtol=BF16_RTOL
                    )
                except AssertionError as error:
                    errors.append(f"step {step} {name} parameter: {error}")
            self._assert_all_ranks_match(errors)

            # This is a per-step FSDP/gradient parity regression. Start the
            # next step from the same updated weights so BF16/Adam rounding
            # drift does not confound the next fresh unshard/weight-sync.
            # Check both updates above before aligning any parameters.
            with torch.no_grad():
                for name, parameter in candidate.model.named_parameters():
                    self._local(parameter).copy_(self._local(ref_params[name]))

        dist.barrier()

    @staticmethod
    def _local(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def _assert_all_ranks_match(self, errors: list[str]) -> None:
        # Fail every worker together: a rank-local assertion must not leave
        # its peers entering the next native collective during teardown.
        gathered: list[list[str] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, errors)
        for rank, failures in enumerate(gathered):
            if failures:
                # The worker forwards its exception through a multiprocessing
                # pipe; keep it short enough to avoid blocking that handoff.
                lines = failures[0].splitlines()
                detail = next((line for line in lines if line.startswith("Greatest absolute")), "")
                raise AssertionError(f"rank {rank}: {lines[0]} {detail}")
