import gc
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import GreedyRouterConfig


def _tiny_moonep_config(
    *,
    return_router_results: bool = False,
    router_async_offload: bool = False,
) -> Qwen3MoEConfig:
    return Qwen3MoEConfig(
        vocab_size=256,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        hidden_size=512,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=1024,
        ep_size=4,
        dispatcher="moonep",
        router_compute_dtype="float32",
        moonep_staging_reference=False,
        balancing_loss_cfg=None,
        return_router_results=return_router_results,
        router_async_offload=router_async_offload,
        compile_cfg=False,
        attention=MHAConfig(
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=64,
            qk_norm=True,
            attn_impl="flex_attention",
        ),
        router=GreedyRouterConfig(
            scoring_func="softmax",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        ),
    )


def _training_item(engine: TrainEngine, *, offset: int = 0) -> ModelItem:
    input_ids = (torch.arange(2, 18, device="cuda") + offset).view(1, -1) % 256
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
    loss_ctx = engine.model.build_loss_ctx_batch(
        [{"seq_ctx": seq_ctx, "shifted_labels": (input_ids + 1) % 256}],
        sp_mesh=None,
    )[0]
    return ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)


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


def _optimizer_step(engine: TrainEngine, *, offset: int) -> tuple[float, torch.Tensor]:
    step = engine.train_step([_training_item(engine, offset=offset)])
    grad_norm = engine.clip_grad_norm(do_clip=False).detach().clone()
    engine.step_optimizer(grad_norm)
    return step["total_loss"], grad_norm


@torch.no_grad()
def _probe_logits(engine: TrainEngine, *, offset: int = 48) -> torch.Tensor:
    item = _training_item(engine, offset=offset)
    engine.model.eval()
    output = engine.model(seq_ctx=item["seq_ctx"], loss_ctx=None)
    assert output.logits is not None
    return output.logits.detach().clone()


def _shared_temporary_directory() -> Path:
    directory = tempfile.mkdtemp() if dist.get_rank() == 0 else None
    shared = [directory]
    dist.broadcast_object_list(shared, src=0)
    assert shared[0] is not None
    return Path(shared[0])


@unittest.skipUnless(torch.cuda.device_count() >= 8, "requires 8 CUDA devices")
class TestMoonEPPersistence(DeterministicDDPTestCase):
    @staticmethod
    def _build_engine(optim_cfg=None, *, model_cfg=None) -> TrainEngine:
        return TrainEngine(
            model_cfg=model_cfg or _tiny_moonep_config(),
            optim_cfg=optim_cfg or AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(ep_size=4, recompute_ratio=0.0),
        )

    def test_engine_close_is_idempotent_and_rejects_further_forward(self) -> None:
        self.create_pg("cuda")
        torch.manual_seed(20260805)
        engine = self._build_engine()
        engine.init_model_weights()
        item = _training_item(engine)

        try:
            engine.model.eval()
            with torch.no_grad():
                output = engine.model(seq_ctx=item["seq_ctx"], loss_ctx=None)
            assert output.logits is not None

            engine.close()
            engine.close()

            with self.assertRaisesRegex(RuntimeError, "closed"):
                engine.train_step([item])
            with self.assertRaisesRegex(RuntimeError, "closed"):
                engine.model(seq_ctx=item["seq_ctx"], loss_ctx=None)
        finally:
            # 红测阶段 close 尚不存在，仍需显式释放 collective 资源。
            if not getattr(engine, "_closed", False):
                torch.cuda.synchronize()
                engine.model.close_ep_runtime()
            dist.barrier()

    def test_rank_divergent_destructor_only_warns(self) -> None:
        self.create_pg("cuda")
        torch.manual_seed(20260805)
        engine = self._build_engine()
        engine.init_model_weights()

        # Rank 0 drops the engine first. If __del__ enters a Buffer barrier,
        # CUDA synchronize, or VMM teardown, this world barrier cannot finish.
        if dist.get_rank() == 0:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                del engine
                gc.collect()
            assert any("TrainEngine.close" in str(item.message) for item in caught)
        dist.barrier()

        if dist.get_rank() != 0:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                del engine
                gc.collect()
            assert any("TrainEngine.close" in str(item.message) for item in caught)
        dist.barrier()

    def test_sync_dcp_cold_resume_matches_uninterrupted_step(self) -> None:
        self.create_pg("cuda")
        checkpoint_root = _shared_temporary_directory()
        weights_dir = checkpoint_root / "weights"
        torch.manual_seed(20260805)
        reference = self._build_engine()
        reference.init_model_weights()

        try:
            _optimizer_step(reference, offset=0)
            _optimizer_step(reference, offset=16)
            checkpoint_model = _local_model_state(reference)
            checkpoint_optimizer = _optimizer_tensor_state(reference)
            reference.save_dcp(weights_dir)
            dist.barrier()

            metadata_keys = dcp.FileSystemReader(weights_dir).read_metadata().state_dict_metadata.keys()
            assert any(key.startswith("model.") for key in metadata_keys)
            assert any(key.startswith("optimizer.") for key in metadata_keys)
            transient_markers = ("moonep", "workspace", "landing", "invocation", "gradient_slot", "event")
            assert not any(marker in key.lower() for key in metadata_keys for marker in transient_markers)

            expected_loss, expected_norm = _optimizer_step(reference, offset=32)
            expected_model = _local_model_state(reference)
            expected_optimizer = _optimizer_tensor_state(reference)
            updated_names = {
                name for name, value in expected_model.items() if not torch.equal(value, checkpoint_model[name])
            }
            assert updated_names
            reference.close()

            # Load occurs before this fresh runtime's first forward/AllGather.
            torch.manual_seed(17)
            resumed = self._build_engine()
            resumed.init_model_weights()
            resumed.load_dcp(weights_dir)

            actual_checkpoint_model = _local_model_state(resumed)
            actual_checkpoint_optimizer = _optimizer_tensor_state(resumed)
            assert actual_checkpoint_model.keys() == checkpoint_model.keys()
            assert actual_checkpoint_optimizer.keys() == checkpoint_optimizer.keys()
            for name, expected in checkpoint_model.items():
                torch.testing.assert_close(actual_checkpoint_model[name], expected, rtol=0, atol=0)
            for name, expected in checkpoint_optimizer.items():
                torch.testing.assert_close(actual_checkpoint_optimizer[name], expected, rtol=0, atol=0)

            actual_loss, actual_norm = _optimizer_step(resumed, offset=32)
            actual_model = _local_model_state(resumed)
            actual_optimizer = _optimizer_tensor_state(resumed)
            torch.testing.assert_close(
                torch.tensor(actual_loss, device="cuda"),
                torch.tensor(expected_loss, device="cuda"),
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-5, atol=1e-6)
            for name in updated_names:
                torch.testing.assert_close(actual_model[name], expected_model[name], rtol=0, atol=0)
            assert actual_optimizer.keys() == expected_optimizer.keys()
            for name, expected in expected_optimizer.items():
                torch.testing.assert_close(actual_optimizer[name], expected, rtol=0, atol=0)
            resumed.close()
        finally:
            if not reference._closed:
                reference.close()
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(checkpoint_root)

    def test_sync_hf_export_loads_into_fresh_moonep_runtime(self) -> None:
        self.create_pg("cuda")
        export_root = _shared_temporary_directory()
        hf_dir = export_root / "hf"
        torch.manual_seed(20260805)
        reference = self._build_engine()
        reference.init_model_weights()
        restored = None

        try:
            _optimizer_step(reference, offset=0)
            _optimizer_step(reference, offset=16)
            expected_state = _local_model_state(reference)
            expected_logits = _probe_logits(reference)
            reference.save_hf(str(hf_dir))
            reference.close()

            restored = self._build_engine()
            restored.from_hf(hf_dir, strict=True)
            actual_state = _local_model_state(restored)
            assert actual_state.keys() == expected_state.keys()
            for name, expected in expected_state.items():
                # HF is deliberately a BF16 interchange format while FSDP
                # optimizer shards remain FP32 in the live engine.
                torch.testing.assert_close(actual_state[name].bfloat16(), expected.bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(_probe_logits(restored), expected_logits, rtol=0, atol=0)
            restored.close()
        finally:
            if not reference._closed:
                reference.close()
            if restored is not None and not restored._closed:
                restored.close()
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(export_root)

    def test_activation_offload_preserves_moonep_training(self) -> None:
        self.create_pg("cuda")
        results = []
        for enabled in (False, True):
            torch.manual_seed(20260805)
            engine = self._build_engine()
            engine.init_model_weights()
            try:
                with patch.dict(os.environ, {"XTUNER_ACTIVATION_OFFLOAD": str(int(enabled))}):
                    loss, grad_norm = _optimizer_step(engine, offset=0)
                results.append((loss, grad_norm, _local_model_state(engine)))
            finally:
                engine.close()

        expected_loss, expected_norm, expected_state = results[0]
        actual_loss, actual_norm, actual_state = results[1]
        torch.testing.assert_close(
            torch.tensor(actual_loss, device="cuda"),
            torch.tensor(expected_loss, device="cuda"),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)
        for name, expected in expected_state.items():
            torch.testing.assert_close(actual_state[name], expected, rtol=0, atol=0)
        dist.barrier()

    def test_router_async_offload_only_changes_detached_logging_outputs(self) -> None:
        self.create_pg("cuda")
        results = []
        for enabled in (False, True):
            torch.manual_seed(20260805)
            config = _tiny_moonep_config(return_router_results=True, router_async_offload=enabled)
            engine = self._build_engine(model_cfg=config)
            engine.init_model_weights()
            try:
                item = _training_item(engine)
                with torch.no_grad():
                    output = engine.model(seq_ctx=item["seq_ctx"], loss_ctx=None)
                assert output.router_logits
                assert output.router_weights
                logging_tensors = [*output.router_logits.values(), *output.router_weights.values()]
                expected_device = "cpu" if enabled else "cuda"
                assert all(tensor.device.type == expected_device for tensor in logging_tensors)
                assert all(not tensor.requires_grad for tensor in logging_tensors)

                loss, grad_norm = _optimizer_step(engine, offset=16)
                results.append((loss, grad_norm, _local_model_state(engine)))
            finally:
                engine.close()

        expected_loss, expected_norm, expected_state = results[0]
        actual_loss, actual_norm, actual_state = results[1]
        torch.testing.assert_close(
            torch.tensor(actual_loss, device="cuda"),
            torch.tensor(expected_loss, device="cuda"),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)
        for name, expected in expected_state.items():
            torch.testing.assert_close(actual_state[name], expected, rtol=0, atol=0)
        dist.barrier()

    def test_async_hf_export_is_immutable_and_close_waits_for_writer(self) -> None:
        self.create_pg("cuda")
        export_root = _shared_temporary_directory()
        hf_dir = export_root / "hf"
        torch.manual_seed(20260805)
        reference = self._build_engine()
        reference.init_model_weights()
        restored = None

        try:
            _optimizer_step(reference, offset=0)
            _optimizer_step(reference, offset=16)
            expected_state = _local_model_state(reference)
            expected_logits = _probe_logits(reference)

            save_future = reference.async_save_hf(str(hf_dir))
            _optimizer_step(reference, offset=32)
            reference.close()
            assert save_future.done()
            assert hf_dir.is_dir()

            restored = self._build_engine()
            restored.from_hf(hf_dir, strict=True)
            actual_state = _local_model_state(restored)
            for name, expected in expected_state.items():
                torch.testing.assert_close(actual_state[name].bfloat16(), expected.bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(_probe_logits(restored), expected_logits, rtol=0, atol=0)
            restored.close()
        finally:
            if not reference._closed:
                if "save_future" in locals():
                    save_future.result()
                reference.close()
            if restored is not None and not restored._closed:
                restored.close()
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(export_root)

    @property
    def world_size(self) -> int:
        return 8
