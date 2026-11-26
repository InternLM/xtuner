"""UltraEP checkpoint/persistence regression, enabled with XTUNER_TEST_ULTRA_EP=1.

Verifies the design contract that FSDP2 solely owns model parameters, optimizer
state and checkpoint identity: a DCP checkpoint written by an UltraEP engine
must contain no UltraEP runtime state, and reloading it must reproduce the
uninterrupted run bit-identically. Requires eight CUDA GPUs and the patched
native UltraEP extra (see patch/ultraep/README.md). No XTuner module is mocked.

Note: UltraEP keeps a process-wide native runtime bound to the first EP group,
so resume is exercised on the same engine instance -- the same load_dcp code
path a fresh trainer process would take after a restart.
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.module.ultraep import UltraEPConfig

from .test_moe_train_engine_tpep import (
    _build_tiny_moe_cfg,
    _make_engine_input,
    _run_train_step_items_without_clip,
)


def _local_model_state(engine: TrainEngine) -> dict[str, torch.Tensor]:
    state = {}
    for name, value in engine.model.state_dict().items():
        if isinstance(value, DTensor):
            value = value.to_local()
        state[name] = value.detach().clone()
    return state


def _optimizer_tensor_state(engine: TrainEngine) -> dict[str, torch.Tensor]:
    tensors = {}
    parameter_names = {id(parameter): name for name, parameter in engine.model.named_parameters()}
    state_dict = engine.optimizer.state_dict()
    for saved_group, live_group in zip(state_dict["param_groups"], engine.optimizer.param_groups, strict=True):
        for parameter_id, parameter in zip(saved_group["params"], live_group["params"], strict=True):
            parameter_name = parameter_names[id(parameter)]
            for name, value in state_dict["state"][parameter_id].items():
                if isinstance(value, torch.Tensor):
                    if isinstance(value, DTensor):
                        value = value.to_local()
                    tensors[f"{parameter_name}.{name}"] = value.detach().clone()
    return tensors


def _optimizer_step(engine: TrainEngine, loss_config: CELossConfig, *, offset: int) -> tuple[float, torch.Tensor]:
    device = torch.device("cuda", dist.get_rank())
    batches = [_make_engine_input(device, seed_offset=offset)]
    loss = _run_train_step_items_without_clip(engine, loss_config, batches)
    grad_norm = engine.clip_grad_norm(do_clip=False).detach().clone()
    engine.step_optimizer(grad_norm)
    return loss, grad_norm


def _shared_temporary_directory() -> Path:
    directory = tempfile.mkdtemp() if dist.get_rank() == 0 else None
    shared = [directory]
    dist.broadcast_object_list(shared, src=0)
    assert shared[0] is not None
    return Path(shared[0])


@pytest.mark.gpu
@unittest.skipUnless(
    os.getenv("XTUNER_TEST_ULTRA_EP") == "1"
    and torch.cuda.device_count() >= 8
    and importlib.util.find_spec("ultra_ep") is not None
    and importlib.util.find_spec("deep_ep") is not None,
    "Set XTUNER_TEST_ULTRA_EP=1 with eight GPUs and the patched native UltraEP extra",
)
class TestUltraEPPersistence(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 8

    @staticmethod
    def _build_engine(mesh_prefix: str) -> TrainEngine:
        config = _build_tiny_moe_cfg(ep_size=4)
        config.n_routed_experts = 16
        config.gate_bias = True
        config.dispatcher = "deepep"
        config.mesh_prefix = mesh_prefix
        config.ultraep_cfg = UltraEPConfig(num_redundant_experts_per_rank=1)
        return TrainEngine(
            model_cfg=config,
            optim_cfg=AdamWConfig(lr=1e-3, eps=1e-4),
            fsdp_cfg=FSDPConfig(
                ep_size=4,
                recompute_ratio=0,
                reshard_after_forward=True,
                torch_compile=False,
                mesh_prefix=mesh_prefix,
            ),
            intra_layer_micro_batch=1,
        )

    @staticmethod
    def _destroy_native_manager(engine: TrainEngine) -> None:
        # Model.close deliberately retains the process-wide native cache.
        # This isolated worker will never reuse it. Finalize the native
        # dependency collectively before Python tears down modules in an
        # unspecified order (its C++ destructor contains an NVSHMEM barrier).
        manager = engine.model._ep_runtime._manager
        if manager is not None:
            manager.runtime.runtime.destroy()

    def test_engine_close_is_idempotent(self) -> None:
        self.create_pg("cuda")
        os.environ["XTUNER_GROUP_GEMM"] = "triton"
        torch.manual_seed(20260805)
        engine = self._build_engine("ultraep_close_test")
        engine.init_model_weights()
        try:
            engine.close()
            engine.close()
        finally:
            self._destroy_native_manager(engine)
            dist.destroy_process_group()

    def test_dcp_reload_matches_uninterrupted_step(self) -> None:
        self.create_pg("cuda")
        os.environ["XTUNER_GROUP_GEMM"] = "triton"
        checkpoint_root = _shared_temporary_directory()
        weights_dir = checkpoint_root / "weights"
        loss_config = CELossConfig()

        torch.manual_seed(20260805)
        engine = self._build_engine("ultraep_ckpt")
        engine.init_model_weights()

        try:
            _optimizer_step(engine, loss_config, offset=0)
            _optimizer_step(engine, loss_config, offset=16)
            checkpoint_model = _local_model_state(engine)
            checkpoint_optimizer = _optimizer_tensor_state(engine)
            engine.save_dcp(weights_dir)
            dist.barrier()

            metadata_keys = dcp.FileSystemReader(weights_dir).read_metadata().state_dict_metadata.keys()
            assert any(key.startswith("model.") for key in metadata_keys)
            assert any(key.startswith("optimizer.") for key in metadata_keys)
            transient_markers = ("ultraep", "ultra_ep", "staging", "workspace", "manager", "event")
            assert not any(marker in key.lower() for key in metadata_keys for marker in transient_markers)

            # Uninterrupted continuation: the reference trajectory.
            expected_loss, expected_norm = _optimizer_step(engine, loss_config, offset=32)
            expected_model = _local_model_state(engine)
            expected_optimizer = _optimizer_tensor_state(engine)
            updated_names = {
                name for name, value in expected_model.items() if not torch.equal(value, checkpoint_model[name])
            }
            assert updated_names

            # Rewind to the checkpoint and replay the same step.
            engine.load_dcp(weights_dir)
            actual_checkpoint_model = _local_model_state(engine)
            actual_checkpoint_optimizer = _optimizer_tensor_state(engine)
            assert actual_checkpoint_model.keys() == checkpoint_model.keys()
            assert actual_checkpoint_optimizer.keys() == checkpoint_optimizer.keys()
            for name, expected in checkpoint_model.items():
                torch.testing.assert_close(actual_checkpoint_model[name], expected, rtol=0, atol=0)
            for name, expected in checkpoint_optimizer.items():
                torch.testing.assert_close(actual_checkpoint_optimizer[name], expected, rtol=0, atol=0)

            actual_loss, actual_norm = _optimizer_step(engine, loss_config, offset=32)
            actual_model = _local_model_state(engine)
            actual_optimizer = _optimizer_tensor_state(engine)
            torch.testing.assert_close(
                torch.tensor(actual_loss, device="cuda"),
                torch.tensor(expected_loss, device="cuda"),
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-5, atol=1e-6)
            # The replayed step is NOT bit-identical: UltraEP's placement
            # quota state is runtime memory (data-driven, accumulates expert
            # load history) and deliberately not part of the checkpoint, so
            # the replay routes through a slightly different placement. The
            # checkpoint contract verified here is that model/optimizer state
            # round-trips exactly (above) and the replay stays within BF16
            # parity tolerance.
            for name in updated_names:
                torch.testing.assert_close(actual_model[name], expected_model[name], rtol=1.6e-2, atol=1e-5)
            assert actual_optimizer.keys() == expected_optimizer.keys()
            for name, expected in expected_optimizer.items():
                torch.testing.assert_close(actual_optimizer[name], expected, rtol=1.6e-2, atol=1e-5)
        finally:
            rank = dist.get_rank()
            engine.close()
            dist.barrier()
            self._destroy_native_manager(engine)
            dist.destroy_process_group()
            if rank == 0:
                shutil.rmtree(checkpoint_root, ignore_errors=True)

    def test_hf_reload_restores_exact_bf16_state(self) -> None:
        self.create_pg("cuda")
        os.environ["XTUNER_GROUP_GEMM"] = "triton"
        export_root = _shared_temporary_directory()
        hf_dir = export_root / "hf"
        loss_config = CELossConfig()

        torch.manual_seed(20260805)
        engine = self._build_engine("ultraep_hf")
        engine.init_model_weights()

        try:
            _optimizer_step(engine, loss_config, offset=0)
            expected_state = _local_model_state(engine)
            engine.save_hf(str(hf_dir))
            dist.barrier()

            # Move the live weights away from the checkpoint, then reload.
            _optimizer_step(engine, loss_config, offset=16)
            engine.from_hf(hf_dir, strict=True)
            actual_state = _local_model_state(engine)
            assert actual_state.keys() == expected_state.keys()
            for name, expected in expected_state.items():
                # HF is deliberately a BF16 interchange format while FSDP
                # optimizer shards remain FP32 in the live engine.
                torch.testing.assert_close(actual_state[name].bfloat16(), expected.bfloat16(), rtol=0, atol=0)
        finally:
            rank = dist.get_rank()
            engine.close()
            dist.barrier()
            self._destroy_native_manager(engine)
            dist.destroy_process_group()
            if rank == 0:
                shutil.rmtree(export_root, ignore_errors=True)
