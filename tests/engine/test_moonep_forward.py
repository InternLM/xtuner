import unittest

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.glm52 import Glm52MoEConfig
from xtuner.v1.model.moe.moe import MoEConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import DSAMLAConfig, MHAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router import GreedyRouterConfig, NoAuxRouterConfig
from xtuner.v1.utils.test_utils import init_data_mesh


def _tiny_config(
    family: str,
    dispatcher: str,
    *,
    compile: bool,
    router_compute_dtype: str = "float32",
    staging_reference: bool | None = None,
    mtp_config: MTPConfig | None = None,
    n_shared_experts: int = 1,
    with_shared_expert_gate: bool = False,
) -> MoEConfig:
    common = dict(
        vocab_size=256,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        # With EP4/E8 each home chunk must satisfy CUDA's 2 MiB VMM
        # granularity for both fused projections.
        hidden_size=512,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        n_routed_experts=8,
        n_shared_experts=n_shared_experts,
        with_shared_expert_gate=with_shared_expert_gate,
        num_experts_per_tok=2,
        moe_intermediate_size=1024,
        ep_size=4,
        dispatcher=dispatcher,
        router_compute_dtype=router_compute_dtype,
        moonep_staging_reference=False if staging_reference is None else staging_reference,
        balancing_loss_cfg=None,
        mtp_config=mtp_config,
        compile_cfg=(
            {"xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.forward": {"fullgraph": True}}
            if compile
            else False
        ),
    )
    if family == "qwen":
        return Qwen3MoEConfig(
            **common,
            bos_token_id=2,
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
    if family == "glm52":
        return Glm52MoEConfig(
            **common,
            hf_eos_token_id=[1],
            attention=DSAMLAConfig(
                num_attention_heads=2,
                head_dim=4,
                kv_lora_rank=4,
                q_lora_rank=8,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=4,
                index_topk=4,
                index_head_dim=4,
                index_n_heads=2,
                indexer_types=["full", "shared", "shared"],
                sparse_mla_backend="torch",
            ),
            hf_head_dim=4,
            qk_head_dim=8,
            router=NoAuxRouterConfig(
                n_group=1,
                topk_group=1,
                scoring_func="sigmoid",
                norm_topk_prob=True,
                router_scaling_factor=2.5,
            ),
            mlp_layer_types=["dense", "sparse", "sparse"],
            num_nextn_predict_layers=None,
        )
    raise AssertionError(f"unknown tiny model family: {family}")


@unittest.skipUnless(torch.cuda.device_count() >= 8, "requires 8 CUDA devices")
class TestMoonEPStagingForward(DeterministicDDPTestCase):
    @staticmethod
    def _training_item() -> ModelItem:
        input_ids = torch.arange(2, 18, device="cuda").view(1, -1)
        labels = (input_ids + 1) % 256
        loss_cfg = CELossConfig()
        loss_ctx = loss_cfg.build(data={"shifted_labels": labels})
        assert loss_ctx is not None
        loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
        return ModelItem(
            seq_ctx=SequenceContext.from_input_ids((input_ids,), device="cuda"),
            loss_ctx={"lm": loss_ctx},
        )

    @staticmethod
    def _model_training_item(
        engine: TrainEngine,
        *,
        offset: int = 0,
        sequence_length: int = 16,
        sp_mesh=None,
    ) -> ModelItem:
        input_ids = (torch.arange(2, 2 + sequence_length, device="cuda") + offset).view(1, -1) % 256
        full_seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
        loss_ctx = engine.model.build_loss_ctx_batch(
            [{"seq_ctx": full_seq_ctx, "shifted_labels": (input_ids + 1) % 256}],
            sp_mesh=sp_mesh,
        )[0]
        seq_ctx = full_seq_ctx if sp_mesh is None else full_seq_ctx.split(sp_mesh)
        return ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)

    def _forward(self, family: str, dispatcher: str) -> torch.Tensor:
        torch.manual_seed(20260805)
        engine = TrainEngine(
            model_cfg=_tiny_config(family, dispatcher, compile=True),
            optim_cfg=AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(ep_size=4, recompute_ratio=0.0, torch_compile=True),
            intra_layer_micro_batch=1,
        )
        engine.init_model_weights()
        if dispatcher == "moonep":
            assert engine.model.config.intra_layer_micro_batch == 1
        input_ids = torch.arange(2, 18, device="cuda").view(1, -1)

        try:
            engine.model.eval()
            with torch.no_grad():
                first = engine.model(
                    seq_ctx=SequenceContext.from_input_ids((input_ids,), device="cuda"),
                    loss_ctx=None,
                ).logits

            assert first is not None
            assert torch.isfinite(first).all()
            repeats = 3 if dispatcher == "moonep" else int(dispatcher != "deepep")
            for _ in range(repeats):
                with torch.no_grad():
                    repeated = engine.model(
                        seq_ctx=SequenceContext.from_input_ids((input_ids,), device="cuda"),
                        loss_ctx=None,
                    ).logits
                torch.testing.assert_close(first, repeated, rtol=0, atol=0)
            return first.clone()
        finally:
            # Resource teardown may unmap VMM landings, so the test must first
            # complete queued output copies. This is lifecycle-only, not a hot-path sync.
            torch.cuda.synchronize()
            if dispatcher == "moonep":
                engine.model.destroy_moonep()
            del engine
            # DeepEP owns a process-scoped C++ Buffer. Forcing cyclic GC here
            # can destruct it on only a subset of ranks; leave that resource
            # to the distributed process teardown.
            torch.cuda.empty_cache()
            dist.barrier()

    def _assert_matches_reference(self, family: str, reference: str) -> None:
        self.create_pg("cuda")
        expected = self._forward(family, reference)
        moonep = self._forward(family, "moonep")
        torch.testing.assert_close(moonep, expected, rtol=1e-2, atol=1e-2)

    @staticmethod
    def _selected_training_tensors(engine: TrainEngine, *, gradients: bool) -> dict[str, torch.Tensor]:
        selected = {}
        for name, parameter in engine.model.named_parameters():
            if not any(
                marker in name for marker in (".experts.", ".shared_experts.", ".shared_expert_gate.", ".gate.")
            ):
                continue
            value = parameter.grad if gradients else parameter
            assert value is not None
            if isinstance(value, DTensor):
                value = value.to_local()
            selected[name] = value.detach().clone()
        return selected

    def _train_two_steps(
        self,
        dispatcher: str,
        *,
        staging_reference: bool | None = None,
    ) -> tuple[list[float], list[torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        torch.manual_seed(20260805)
        engine = TrainEngine(
            model_cfg=_tiny_config(
                "qwen",
                dispatcher,
                compile=True,
                staging_reference=staging_reference,
            ),
            optim_cfg=AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(ep_size=4, recompute_ratio=0.0, torch_compile=True),
        )
        engine.init_model_weights()
        losses = []
        grad_norms = []
        gradients = []
        try:
            for _ in range(2):
                step = engine.train_step([self._training_item()])
                losses.append(step["total_loss"])
                grad_norms.append(engine.clip_grad_norm(do_clip=False).detach().clone())
                gradients.append(self._selected_training_tensors(engine, gradients=True))
                engine.step_optimizer(grad_norms[-1])
            parameters = self._selected_training_tensors(engine, gradients=False)
            return losses, grad_norms, gradients, parameters
        finally:
            torch.cuda.synchronize()
            if dispatcher == "moonep":
                engine.model.destroy_moonep()
            del engine
            torch.cuda.empty_cache()
            dist.barrier()

    @staticmethod
    def _assert_training_runs_close(
        actual: tuple[list[float], list[torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        expected: tuple[list[float], list[torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    ) -> None:
        actual_losses, actual_norms, actual_gradients, actual_parameters = actual
        expected_losses, expected_norms, expected_gradients, expected_parameters = expected
        torch.testing.assert_close(
            torch.tensor(actual_losses, device="cuda"),
            torch.tensor(expected_losses, device="cuda"),
            rtol=1e-2,
            atol=1e-3,
        )
        for actual_norm, expected_norm in zip(actual_norms, expected_norms):
            torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-2, atol=1e-3)
        for actual_step, expected_step in zip(actual_gradients, expected_gradients):
            assert actual_step.keys() == expected_step.keys()
            for name in actual_step:
                torch.testing.assert_close(actual_step[name], expected_step[name], rtol=1e-2, atol=1e-3)
        assert actual_parameters.keys() == expected_parameters.keys()
        for name in actual_parameters:
            torch.testing.assert_close(actual_parameters[name], expected_parameters[name], rtol=1e-2, atol=1e-3)

    def test_qwen_fixed_length_fused_expert_forward_matches_deepep(self) -> None:
        self._assert_matches_reference("qwen", "deepep")

    def test_glm52_fixed_length_fused_expert_forward_matches_all2all(self) -> None:
        self._assert_matches_reference("glm52", "all2all")

    def test_qwen_backward_updates_routed_expert_fsdp_shards(self) -> None:
        self.create_pg("cuda")
        torch.manual_seed(20260805)
        engine = TrainEngine(
            model_cfg=_tiny_config("qwen", "moonep", compile=True, router_compute_dtype="native"),
            optim_cfg=AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(ep_size=4, recompute_ratio=0.0, torch_compile=True),
        )
        engine.init_model_weights()
        routed_parameter = next(
            parameter for name, parameter in engine.model.named_parameters() if ".experts." in name
        )
        routed_parameter.grad = torch.full_like(routed_parameter, 15)
        engine.model.scale_and_reduce_grad()
        torch.testing.assert_close(
            routed_parameter.grad.to_local(),
            torch.full_like(routed_parameter.grad.to_local(), 15 / 4),
            rtol=0,
            atol=0,
        )
        engine.optimizer.zero_grad()
        before = {
            name: parameter.to_local().detach().clone()
            for name, parameter in engine.model.named_parameters()
            if ".experts." in name
        }

        try:
            step = engine.train_step([self._training_item()])
            assert torch.isfinite(torch.tensor(step["total_loss"], device="cuda"))
            routed = {name: parameter for name, parameter in engine.model.named_parameters() if ".experts." in name}
            assert routed
            assert all(parameter.grad is not None for parameter in routed.values())
            assert all(torch.isfinite(parameter.grad.to_local()).all() for parameter in routed.values())

            grad_norm = engine.clip_grad_norm(do_clip=False)
            engine.step_optimizer(grad_norm)
            assert any(not torch.equal(before[name], parameter.to_local()) for name, parameter in routed.items())
        finally:
            torch.cuda.synchronize()
            engine.model.destroy_moonep()
            del engine
            torch.cuda.empty_cache()
            dist.barrier()

    def test_qwen_two_step_training_matches_deepep(self) -> None:
        self.create_pg("cuda")
        expected = self._train_two_steps("deepep")
        actual = self._train_two_steps("moonep")
        repeated = self._train_two_steps("moonep")
        self._assert_training_runs_close(actual, expected)
        self._assert_training_runs_close(repeated, actual)

    def test_qwen_direct_landing_matches_staging_training(self) -> None:
        self.create_pg("cuda")
        staging = self._train_two_steps("moonep", staging_reference=True)
        direct = self._train_two_steps("moonep", staging_reference=False)
        self._assert_training_runs_close(direct, staging)

    def test_qwen_direct_hot_path_has_no_full_weight_copy_or_host_sync(self) -> None:
        self.create_pg("cuda")

        def profile_mode(
            staging_reference: bool,
            *,
            mtp_micro2_sp4: bool = False,
        ) -> tuple[int, int, list[str], int]:
            torch.manual_seed(20260805)
            engine = TrainEngine(
                model_cfg=_tiny_config(
                    "qwen",
                    "moonep",
                    compile=True,
                    staging_reference=staging_reference,
                    mtp_config=(MTPConfig(num_layers=2, share_weights=True) if mtp_micro2_sp4 else None),
                ),
                optim_cfg=AdamWConfig(foreach=False),
                fsdp_cfg=FSDPConfig(
                    ep_size=4,
                    recompute_ratio=1.0 if mtp_micro2_sp4 else 0.0,
                    torch_compile=True,
                    mtp_checkpoint_use_reentrant=True,
                ),
                intra_layer_micro_batch=2 if mtp_micro2_sp4 else 1,
            )
            engine.init_model_weights()
            if mtp_micro2_sp4:
                sp_mesh = init_data_mesh("cuda", sp_size=4)["sp"]
                train_items = [
                    self._model_training_item(
                        engine,
                        offset=micro_batch_idx * 32,
                        sequence_length=32,
                        sp_mesh=sp_mesh,
                    )
                    for micro_batch_idx in range(2)
                ]
            else:
                train_items = [self._training_item()]
            try:
                torch.cuda.reset_peak_memory_stats()
                # Compile/autotune before profiling so their setup-only CUDA
                # synchronization cannot be confused with the steady hot path.
                engine.train_step(train_items)
                grad_norm = engine.clip_grad_norm(do_clip=False)
                engine.step_optimizer(grad_norm)

                with torch.profiler.profile(
                    activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
                    record_shapes=True,
                ) as profiler:
                    engine.train_step(train_items)

                full_home_shapes = {(2, 2048, 512), (2, 512, 1024)}
                full_local_dw_shapes = {(4, 2048, 512), (4, 512, 1024)}
                full_weight_copies = 0
                full_dw_materializations = 0
                host_syncs: list[str] = []
                for event in profiler.events():
                    tensor_shapes = {
                        tuple(shape)
                        for shape in event.input_shapes
                        if isinstance(shape, list) and all(isinstance(dim, int) for dim in shape)
                    }
                    if event.name == "aten::copy_" and tensor_shapes & full_home_shapes:
                        full_weight_copies += 1
                    if event.name in {"aten::clone", "aten::copy_", "aten::zeros_like"} and (
                        tensor_shapes & full_local_dw_shapes
                    ):
                        full_dw_materializations += 1

                    parent = event.cpu_parent
                    inside_gate = False
                    while parent is not None:
                        if parent.name.startswith("MoonEP::"):
                            inside_gate = True
                            break
                        parent = parent.cpu_parent
                    if inside_gate and event.name in {
                        "cudaDeviceSynchronize",
                        "cudaEventSynchronize",
                        "cudaStreamSynchronize",
                    }:
                        ancestry = []
                        parent = event.cpu_parent
                        while parent is not None:
                            ancestry.append(parent.name)
                            parent = parent.cpu_parent
                        host_syncs.append(f"{event.name} <- {' <- '.join(ancestry)}")
                return (
                    full_weight_copies,
                    full_dw_materializations,
                    host_syncs,
                    torch.cuda.max_memory_allocated(),
                )
            finally:
                torch.cuda.synchronize()
                engine.model.destroy_moonep()
                del engine
                torch.cuda.empty_cache()
                dist.barrier()

        staging_copies, _, _, _ = profile_mode(True)
        direct_copies, direct_dw_materializations, direct_host_syncs, _ = profile_mode(False)
        assert staging_copies > 0  # Calibrates the shape-based copy detector.
        assert direct_copies == 0
        assert direct_dw_materializations == 0
        assert direct_host_syncs == [], direct_host_syncs
        combo_copies, combo_dw_materializations, combo_host_syncs, combo_peak_bytes = profile_mode(
            False,
            mtp_micro2_sp4=True,
        )
        assert combo_copies == 0
        assert combo_dw_materializations == 0
        assert combo_host_syncs == [], combo_host_syncs
        # The fixed tiny fallback measured 0.185 GiB/rank on H200; leave ample
        # headroom while still catching an accidental full-model materialization.
        assert combo_peak_bytes < 2**30

    def _train_mtp_micro2(
        self,
        dispatcher: str,
        *,
        share_weights: bool,
    ) -> tuple[list[tuple[float, float]], list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        torch.manual_seed(20260805)
        engine = TrainEngine(
            model_cfg=_tiny_config(
                "qwen",
                dispatcher,
                compile=True,
                mtp_config=MTPConfig(num_layers=2, share_weights=share_weights),
            ),
            optim_cfg=AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(
                ep_size=4,
                recompute_ratio=1.0,
                torch_compile=True,
                mtp_checkpoint_use_reentrant=True,
            ),
            intra_layer_micro_batch=2,
        )
        engine.init_model_weights()
        losses = []
        grad_norms = []
        gradients = []
        try:
            if dispatcher == "moonep":
                # 两次 forward-only 调用必须各自完成并释放 main/MTP plan；随后
                # 同一个 runtime 直接进入正常 reentrant training/replay。
                engine.model.eval()
                forward_only_losses = []
                with torch.no_grad():
                    for _ in range(2):
                        item = self._model_training_item(engine)
                        output = engine.model(seq_ctx=item["seq_ctx"], loss_ctx=item["loss_ctx"])
                        assert output.loss is not None and output.mtp_loss is not None
                        forward_only_losses.append(torch.stack((output.loss, output.mtp_loss)))
                torch.testing.assert_close(forward_only_losses[1], forward_only_losses[0], rtol=1e-2, atol=3e-3)
                engine.model.train()
            for step_idx in range(3):
                step = engine.train_step(
                    [
                        self._model_training_item(engine, offset=step_idx * 32),
                        self._model_training_item(engine, offset=step_idx * 32 + 16),
                    ]
                )
                losses.append((step["total_loss"], step["logs_info"]["reduced_mtp_loss"]))
                grad_norms.append(engine.clip_grad_norm(do_clip=False).detach().clone())
                if step_idx == 0:
                    gradients.append(self._selected_training_tensors(engine, gradients=True))
                engine.step_optimizer(grad_norms[-1])
            return losses, grad_norms, gradients
        finally:
            torch.cuda.synchronize()
            if dispatcher == "moonep":
                engine.model.destroy_moonep()
            del engine
            torch.cuda.empty_cache()
            dist.barrier()

    def _assert_mtp_micro2_matches_deepep(self, *, share_weights: bool) -> None:
        self.create_pg("cuda")
        expected = self._train_mtp_micro2("deepep", share_weights=share_weights)
        actual = self._train_mtp_micro2("moonep", share_weights=share_weights)
        try:
            expected_losses, expected_norms, expected_gradients = expected
            actual_losses, actual_norms, actual_gradients = actual
            torch.testing.assert_close(
                torch.tensor(actual_losses, device="cuda"),
                torch.tensor(expected_losses, device="cuda"),
                rtol=1e-2,
                atol=1e-3,
            )
            for actual_norm, expected_norm in zip(actual_norms, expected_norms):
                torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-2, atol=1e-3)
            # DeepEP/MoonEP 的 BF16 前向舍入会通过后续 router 放大，不适合在
            # micro2 多层图上逐元素判梯度；slot 累加由 identical-items 测试精确覆盖。
            for actual_step, expected_step in zip(actual_gradients, expected_gradients):
                assert actual_step.keys() == expected_step.keys()
                for gradients in (actual_step, expected_step):
                    assert all(torch.isfinite(tensor).all() for tensor in gradients.values())
                    assert any(torch.count_nonzero(tensor) > 0 for tensor in gradients.values())
        finally:
            # All ranks must leave numerical assertions together before the
            # distributed test harness destroys the world process group.
            torch.cuda.synchronize()
            dist.barrier()

    def test_qwen_unshared_mtp_reentrant_micro2_matches_deepep(self) -> None:
        self._assert_mtp_micro2_matches_deepep(share_weights=False)

    def test_qwen_shared_mtp_reentrant_micro2_matches_deepep(self) -> None:
        self._assert_mtp_micro2_matches_deepep(share_weights=True)

    def test_qwen_rejects_domino_width_above_the_gradient_ring_capacity(self) -> None:
        self.create_pg("cuda")
        torch.manual_seed(20260805)
        engine = TrainEngine(
            model_cfg=_tiny_config("qwen", "moonep", compile=True),
            optim_cfg=AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(ep_size=4, recompute_ratio=0.0, torch_compile=True),
            intra_layer_micro_batch=2,
        )
        engine.init_model_weights()
        items = [self._model_training_item(engine, offset=idx * 16) for idx in range(3)]
        try:
            with torch.no_grad(), self.assertRaisesRegex(ValueError, "width 3 exceeds configured capacity 2"):
                engine.model(
                    seq_ctx=[item["seq_ctx"] for item in items],
                    loss_ctx=[item["loss_ctx"] for item in items],
                )
        finally:
            torch.cuda.synchronize()
            engine.model.destroy_moonep()
            del engine
            torch.cuda.empty_cache()
            dist.barrier()

    def _train_microbatches_without_mtp(
        self,
        dispatcher: str,
        *,
        recompute_ratio: float,
        offsets: tuple[int, ...] = (0, 16),
        n_shared_experts: int = 1,
        with_shared_expert_gate: bool = False,
        routed_only: bool = True,
    ) -> tuple[float, torch.Tensor, dict[str, torch.Tensor]]:
        torch.manual_seed(20260805)
        engine = TrainEngine(
            model_cfg=_tiny_config(
                "qwen",
                dispatcher,
                compile=True,
                n_shared_experts=n_shared_experts,
                with_shared_expert_gate=with_shared_expert_gate,
            ),
            optim_cfg=AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(
                ep_size=4,
                recompute_ratio=recompute_ratio,
                torch_compile=True,
                mtp_checkpoint_use_reentrant=True,
            ),
            intra_layer_micro_batch=len(offsets),
        )
        engine.init_model_weights()
        try:
            step = engine.train_step([self._model_training_item(engine, offset=offset) for offset in offsets])
            grad_norm = engine.clip_grad_norm(do_clip=False).detach().clone()
            gradients = {
                name: tensor
                for name, tensor in self._selected_training_tensors(engine, gradients=True).items()
                if not routed_only or ".experts." in name
            }
            return step["total_loss"], grad_norm, gradients
        finally:
            torch.cuda.synchronize()
            if dispatcher == "moonep":
                engine.model.destroy_moonep()
            del engine
            torch.cuda.empty_cache()
            dist.barrier()

    def test_qwen_reentrant_micro2_routed_gradients_match_deepep(self) -> None:
        self.create_pg("cuda")
        expected_loss, expected_norm, expected_gradients = self._train_microbatches_without_mtp(
            "deepep", recompute_ratio=1.0
        )
        actual_loss, actual_norm, actual_gradients = self._train_microbatches_without_mtp(
            "moonep", recompute_ratio=1.0
        )
        torch.testing.assert_close(
            torch.tensor(actual_loss, device="cuda"),
            torch.tensor(expected_loss, device="cuda"),
            rtol=1e-2,
            atol=1e-3,
        )
        torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-2, atol=1e-3)
        assert actual_gradients.keys() == expected_gradients.keys()
        for gradients in (actual_gradients, expected_gradients):
            assert all(torch.isfinite(tensor).all() for tensor in gradients.values())
            assert any(torch.count_nonzero(tensor) > 0 for tensor in gradients.values())

    def test_qwen_micro2_identical_items_accumulate_like_micro1(self) -> None:
        self.create_pg("cuda")
        expected_loss, expected_norm, expected_gradients = self._train_microbatches_without_mtp(
            "moonep", recompute_ratio=0.0, offsets=(0,)
        )
        actual_loss, actual_norm, actual_gradients = self._train_microbatches_without_mtp(
            "moonep", recompute_ratio=0.0, offsets=(0, 0)
        )
        torch.testing.assert_close(
            torch.tensor(actual_loss, device="cuda"),
            torch.tensor(2 * expected_loss, device="cuda"),
            rtol=1e-2,
            atol=1e-3,
        )
        torch.testing.assert_close(actual_norm, 2 * expected_norm, rtol=1e-2, atol=1e-3)
        assert actual_gradients.keys() == expected_gradients.keys()
        for name in actual_gradients:
            max_error = (actual_gradients[name].float() - 2 * expected_gradients[name].float()).abs().max()
            dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
            assert max_error <= 1e-3, f"{name}: max_abs={max_error.item()}"

    def test_qwen_shared_expert_variants_train(self) -> None:
        self.create_pg("cuda")
        _, no_shared_norm, no_shared_gradients = self._train_microbatches_without_mtp(
            "moonep",
            recompute_ratio=0.0,
            offsets=(0,),
            n_shared_experts=0,
            routed_only=False,
        )
        _, gated_norm, gated_gradients = self._train_microbatches_without_mtp(
            "moonep",
            recompute_ratio=0.0,
            offsets=(0,),
            n_shared_experts=1,
            with_shared_expert_gate=True,
            routed_only=False,
        )
        assert torch.isfinite(no_shared_norm) and torch.isfinite(gated_norm)
        assert not any("shared_expert" in name for name in no_shared_gradients)
        assert any("shared_experts" in name for name in gated_gradients)
        assert any("shared_expert_gate" in name for name in gated_gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gated_gradients.values())

    def test_qwen_shared_expert_gradient_uses_fp32_ep_mean(self) -> None:
        self.create_pg("cuda")
        torch.manual_seed(20260805)
        engine = TrainEngine(
            model_cfg=_tiny_config(
                "qwen",
                "moonep",
                compile=True,
                n_shared_experts=1,
                with_shared_expert_gate=True,
            ),
            optim_cfg=AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(ep_size=4, recompute_ratio=0.0, torch_compile=True),
        )
        engine.init_model_weights()
        shared_gate = next(
            parameter for name, parameter in engine.model.named_parameters() if ".shared_expert_gate." in name
        )
        assert isinstance(shared_gate, DTensor)
        shared_gate.grad = torch.full_like(shared_gate, dist.get_rank() + 1)

        try:
            engine.model.scale_and_reduce_grad()
            local_grad = shared_gate.grad.to_local()
            # Model mesh is [FSDP2, EP4]. Shared parameters are sharded on the
            # first dimension and replicated on each contiguous EP4 row.
            expected_mean = (dist.get_rank() // 4) * 4 + 2.5
            assert local_grad.dtype is torch.float32
            torch.testing.assert_close(
                local_grad,
                torch.full_like(local_grad, expected_mean),
                rtol=0,
                atol=0,
            )
        finally:
            torch.cuda.synchronize()
            engine.model.destroy_moonep()
            del engine
            torch.cuda.empty_cache()
            dist.barrier()

    def _train_sp_once(self, dispatcher: str, *, sp_size: int) -> tuple[float, torch.Tensor]:
        torch.manual_seed(20260805)
        engine = TrainEngine(
            model_cfg=_tiny_config("qwen", dispatcher, compile=True),
            optim_cfg=AdamWConfig(foreach=False),
            fsdp_cfg=FSDPConfig(ep_size=4, recompute_ratio=0.0, torch_compile=True),
        )
        engine.init_model_weights()
        sp_mesh = init_data_mesh("cuda", sp_size=sp_size)["sp"]
        item = self._model_training_item(
            engine,
            sequence_length=32,
            sp_mesh=sp_mesh,
        )

        try:
            step = engine.train_step([item])
            grad_norm = engine.clip_grad_norm(do_clip=False).detach().clone()
            routed_gradients = {
                name: gradient
                for name, gradient in self._selected_training_tensors(engine, gradients=True).items()
                if ".experts." in name
            }
            assert routed_gradients
            assert all(torch.isfinite(gradient).all() for gradient in routed_gradients.values())
            assert any(torch.count_nonzero(gradient) > 0 for gradient in routed_gradients.values())
            engine.step_optimizer(grad_norm)
            return step["total_loss"], grad_norm
        finally:
            torch.cuda.synchronize()
            if dispatcher == "moonep":
                engine.model.destroy_moonep()
            del engine
            torch.cuda.empty_cache()
            dist.barrier()

    def _assert_sp_matches_deepep(self, sp_size: int) -> None:
        self.create_pg("cuda")
        expected_loss, expected_norm = self._train_sp_once("deepep", sp_size=sp_size)
        actual_loss, actual_norm = self._train_sp_once("moonep", sp_size=sp_size)
        torch.testing.assert_close(
            torch.tensor(actual_loss, device="cuda"),
            torch.tensor(expected_loss, device="cuda"),
            rtol=1e-2,
            atol=1e-3,
        )
        torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-2, atol=1e-3)

    def test_qwen_sp2_ep4_matches_deepep(self) -> None:
        self._assert_sp_matches_deepep(2)

    def test_qwen_sp4_ep4_matches_deepep(self) -> None:
        self._assert_sp_matches_deepep(4)

    def test_qwen_sp8_ep4_matches_deepep(self) -> None:
        self._assert_sp_matches_deepep(8)

    @property
    def world_size(self) -> int:
        return 8
