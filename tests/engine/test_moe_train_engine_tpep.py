"""Validate that EP+TP training produces the same forward loss and backward
gradients as a pure single-GPU (EP=1, TP=1) run.

Test topology: world_size = EP * TP * DP = 2 * 2 * 1 = 4 GPUs.

Strategy
--------
1. Build a tiny Qwen3MoE model with EP=2, TP=2.
2. Build the same model with EP=1, TP=1 (4 identical DP replicas).
3. Init both engines with ``init_model_weights()``.  Because weights for EP+TP
   models are Shard(0) on ep_mesh for experts and Replicate for non-experts,
   and ``init_params`` always initialises the *full* tensor before sharding,
   the underlying full weight values are identical when the same RNG seed is
   active on all ranks.
4. Sync expert weights from EP=1 engine to EP=2 engine via DCP so the two
   models start from the exact same checkpoint.
5. Run one ``train_step`` + ``clip_grad_norm`` on both engines with the same
   input.
6. Assert:
   - losses agree within tolerance
   - gate (router) gradients agree within tolerance (non-expert, replicated
     on all ranks in both configs)
"""

from __future__ import annotations

import os

# 本测试关注 FSDP + EP + expert TP 的 loss/梯度校准。
# Triton TMA grouped-GEMM 在部分本地 Triton/LLVM 组合下会编译失败，
# 因此沿用 .dev_scripts 的做法，用 Cutlass 后端跑真实 grouped-GEMM。
os.environ.setdefault("XTUNER_USE_CUTLASS_GROUP_GEMM", "1")

import parametrize
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.dispatcher.base import NaiveDispatcher
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.utils.device import get_device

DEVICE = get_device()

# 本测试的模型参数和主要计算是 bf16，容忍度对齐 torch.testing 的
# bf16 默认值，避免过宽阈值掩盖 expert TP 维度缺失这类校准错误。
BF16_ATOL = 1e-5
BF16_RTOL = 1.6e-2
# grouped-GEMM 和 TP 分片规约会改变 bf16 的累加顺序；逐元素梯度矩阵
# 在接近 0 的位置会有数个 ulp 的差异，不能用它承载 loss/norm 校准红灯。
BF16_GEMM_ATOL = 1e-4
BF16_GEMM_RTOL = BF16_RTOL

# Use a very small model to keep test runtime manageable.
_TINY_LAYERS = 2
_SEQ_LEN = 32
_VOCAB_SIZE = 128


def _build_tiny_moe_cfg(ep_size: int = 1, expert_tp_size: int = 1) -> Qwen3MoEConfig:
    return Qwen3MoEConfig(
        vocab_size=_VOCAB_SIZE,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_hidden_layers=_TINY_LAYERS,
        hidden_size=128,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=4, num_key_value_heads=2, head_dim=32, qk_norm=True),
        tie_word_embeddings=False,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=64,
        router=GreedyRouterConfig(scoring_func="softmax", norm_topk_prob=True, router_scaling_factor=1.0),
        ep_size=ep_size,
        expert_tp_size=expert_tp_size,
        dispatcher="all2all" if ep_size > 1 else None,
        compile_cfg=False,
        # Disable auxiliary losses to keep the comparison clean.
        balancing_loss_cfg=None,
        z_loss_cfg=None,
    )


def _build_engine(ep_size: int, expert_tp_size: int, data_tp_size: int = 1) -> TrainEngine:
    moe_cfg = _build_tiny_moe_cfg(ep_size, expert_tp_size)
    optim_cfg = AdamWConfig()
    fsdp_cfg = FSDPConfig(
        ep_size=ep_size,
        tp_size=data_tp_size,
        cpu_offload=False,
    )
    return TrainEngine(model_cfg=moe_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg)


def _make_engine_input(device: torch.device, seed_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids [1, SEQ_LEN-1], shifted_labels [1, SEQ_LEN-1]) on *device*."""
    torch.manual_seed(12345 + seed_offset)
    full_ids = torch.randint(0, _VOCAB_SIZE, (1, _SEQ_LEN), dtype=torch.long, device=device)
    input_ids = full_ids[:, :-1]  # [1, SEQ_LEN-1]
    labels = full_ids[:, 1:]      # [1, SEQ_LEN-1] already shifted
    return input_ids, labels


def _run_one_step(
    engine: TrainEngine,
    loss_cfg: CELossConfig,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, dict[str, torch.Tensor]]:
    """Run one train step; return (loss_value, {param_name: grad_tensor})."""
    loss_val, grads, _ = _run_one_step_with_norm(engine, loss_cfg, input_ids, labels)
    return loss_val, grads


def _run_one_step_with_norm(
    engine: TrainEngine,
    loss_cfg: CELossConfig,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, dict[str, torch.Tensor], torch.Tensor]:
    """Run one train step; return loss, gate grads and un-clipped grad norm."""
    loss_val = _run_train_step_without_clip(engine, loss_cfg, input_ids, labels)
    grad_norm = engine.clip_grad_norm(do_clip=False)

    # Collect gradients from gate (router) parameters; these are non-expert
    # parameters replicated on all ranks in both configs, so they're easy to
    # compare directly.
    grads: dict[str, torch.Tensor] = {}
    for name, param in engine.model.named_parameters():
        if "gate.weight" in name and param.grad is not None:
            grad = param.grad
            if hasattr(grad, "full_tensor"):
                grad = grad.full_tensor()  # type: ignore[attr-defined]
            grads[name] = grad.detach().float().cpu()
            break  # one gate layer is sufficient

    return loss_val, grads, grad_norm.detach().float().cpu()


def _run_train_step_without_clip(
    engine: TrainEngine,
    loss_cfg: CELossConfig,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    shifted_labels = labels.to(DEVICE)

    LossContext = loss_cfg.loss_ctx_cls
    loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
    loss_ctx_list = LossContext.build_batches([loss_ctx])
    loss_ctx = loss_ctx_list[0]

    engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})]
    step_info = engine.train_step(engine_input)
    return step_info["logs_info"]["reduced_llm_loss"]


def _get_param_grad(engine: TrainEngine, name_suffix: str) -> torch.Tensor:
    for name, param in engine.model.named_parameters():
        if _canonical_name(name).endswith(name_suffix):
            grad = param.grad
            assert grad is not None, f"Missing gradient for {name}"
            if hasattr(grad, "full_tensor"):
                grad = grad.full_tensor()  # type: ignore[attr-defined]
            return grad.detach().float().cpu()
    raise AssertionError(f"Cannot find parameter ending with {name_suffix}")


def _get_tpep_grouped_linear(engine: TrainEngine, module_suffix: str) -> GroupedLinear:
    for name, module in engine.model.named_modules():
        if _canonical_name(name).endswith(module_suffix):
            assert isinstance(module, GroupedLinear)
            return module
    raise AssertionError(f"Cannot find grouped linear module ending with {module_suffix}")


def _canonical_name(name: str) -> str:
    # 第一层会被 activation checkpoint wrapper 包一层，比较逻辑不关心该包装。
    return name.replace("._checkpoint_wrapped_module", "")


def _zero_non_expert_grads(engine: TrainEngine) -> None:
    with torch.no_grad():
        for name, param in engine.model.named_parameters():
            if ".experts" not in _canonical_name(name) and param.grad is not None:
                param.grad.zero_()


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


def _copy_param_from_full(param: torch.nn.Parameter, full_tensor: torch.Tensor) -> None:
    if isinstance(param, DTensor):
        param.copy_(distribute_tensor(full_tensor, param.device_mesh, param.placements))
    else:
        param.copy_(full_tensor)


def _sync_engine_weights(engine_ref: TrainEngine, engine_tpep: TrainEngine) -> None:
    """Synchronize a non-TP reference model into the EP+TP model layout."""
    ref_params = dict(engine_ref.model.named_parameters())
    ref_modules = dict(engine_ref.model.named_modules())
    tpep_modules = dict(engine_tpep.model.named_modules())

    with torch.no_grad():
        for name, param in engine_tpep.model.named_parameters():
            ref_param = ref_params[name]
            full_param = _full_tensor(ref_param.detach()).to(device=param.device, dtype=param.dtype)

            module_name, _, param_name = name.rpartition(".")
            module = tpep_modules[module_name]
            ref_module = ref_modules[module_name]
            if isinstance(module, GroupedLinear) and getattr(module, "tp_enabled", False):
                if param_name == "weight":
                    shard = _slice_tpep_weight(module, full_param, fused_gate_up="fused_w1w3" in module_name)
                    _copy_param_from_full(param, shard)
                elif param_name == "bias":
                    shard = _slice_tpep_bias(module, full_param)
                    _copy_param_from_full(param, shard)
                else:
                    raise RuntimeError(f"Unexpected GroupedLinear parameter: {name}.")
            else:
                ref_full = _full_tensor(getattr(ref_module, param_name).detach()).to(device=param.device, dtype=param.dtype)
                _copy_param_from_full(param, ref_full)


def _slice_tpep_weight(grouped_linear: GroupedLinear, full_weight: torch.Tensor, *, fused_gate_up: bool) -> torch.Tensor:
    num_experts = grouped_linear.num_routed_experts
    out_features = grouped_linear.out_features
    in_features = grouped_linear.in_features
    expert_weight = full_weight.view(num_experts, out_features, in_features)
    expert_weight = expert_weight[grouped_linear.local_expert_start : grouped_linear.local_expert_end]

    tp_rank = grouped_linear.tp_rank
    tp_size = grouped_linear.tp_size
    if grouped_linear.parallel_style == "column":
        if fused_gate_up:
            intermediate_size = out_features // 2
            local_intermediate_size = intermediate_size // tp_size
            gate_start = tp_rank * local_intermediate_size
            gate_end = gate_start + local_intermediate_size
            up_start = intermediate_size + gate_start
            up_end = intermediate_size + gate_end
            expert_weight = torch.cat(
                [
                    expert_weight[:, gate_start:gate_end, :],
                    expert_weight[:, up_start:up_end, :],
                ],
                dim=1,
            )
        else:
            local_out_features = out_features // tp_size
            out_start = tp_rank * local_out_features
            out_end = out_start + local_out_features
            expert_weight = expert_weight[:, out_start:out_end, :]
    elif grouped_linear.parallel_style == "row":
        local_in_features = in_features // tp_size
        in_start = tp_rank * local_in_features
        in_end = in_start + local_in_features
        expert_weight = expert_weight[:, :, in_start:in_end]
    else:
        raise RuntimeError(f"Unexpected grouped linear parallel style: {grouped_linear.parallel_style}.")

    return expert_weight.reshape(grouped_linear.weight.shape)


def _slice_tpep_bias(grouped_linear: GroupedLinear, full_bias: torch.Tensor) -> torch.Tensor:
    expert_bias = full_bias[grouped_linear.local_expert_start : grouped_linear.local_expert_end]
    if grouped_linear.parallel_style == "column":
        local_out_features = grouped_linear.out_features // grouped_linear.tp_size
        out_start = grouped_linear.tp_rank * local_out_features
        out_end = out_start + local_out_features
        expert_bias = expert_bias[:, out_start:out_end]
    return expert_bias.reshape(grouped_linear.bias.shape)


class TestMoETrainEngineExpertTPOnly(DeterministicDDPTestCase):
    """Verify ExpertTP-only training matches the non-ExpertTP baseline."""

    @parametrize.parametrize(
        "device,expert_tp_size",
        [
            ("cuda", 2),
        ],
    )
    def test_expert_tp_only_engine_constructs_and_trains(self, device: str, expert_tp_size: int) -> None:
        pg = self.create_pg(device)

        engine = _build_engine(ep_size=1, expert_tp_size=expert_tp_size)
        engine.init_model_weights()

        assert engine.model.ep_mesh is not None
        assert engine.model.expert_tp_mesh is not None
        assert engine.model.ep_mesh.size() == 1
        assert engine.model.expert_tp_mesh.size() == expert_tp_size
        assert engine.model.expert_tp_mesh.mesh_dim_names == (f"{engine.model.config.mesh_prefix}.etp",)
        assert isinstance(engine.model.layers["0"].dispatcher, NaiveDispatcher)

        input_ids, labels = _make_engine_input(
            torch.device(device, dist.get_rank() % torch.cuda.device_count()),
            seed_offset=dist.get_rank(),
        )
        loss_cfg = CELossConfig()

        loss_val = _run_train_step_without_clip(engine, loss_cfg, input_ids, labels)
        grad_norm = engine.clip_grad_norm()
        engine.step_optimizer(grad_norm)

        assert torch.isfinite(torch.tensor(loss_val))
        assert torch.isfinite(grad_norm)

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @parametrize.parametrize(
        "device,expert_tp_size",
        [
            ("cuda", 2),
        ],
    )
    def test_expert_tp_only_matches_single_with_distinct_source_slices(
        self, device: str, expert_tp_size: int
    ) -> None:
        pg = self.create_pg(device)

        engine_ref = _build_engine(ep_size=1, expert_tp_size=1)
        engine_ref.init_model_weights()

        engine_etp = _build_engine(ep_size=1, expert_tp_size=expert_tp_size)
        engine_etp.init_model_weights()
        _sync_engine_weights(engine_ref, engine_etp)
        dist.barrier()

        input_ids, labels = _make_engine_input(
            torch.device(device, dist.get_rank() % torch.cuda.device_count()),
            seed_offset=dist.get_rank(),
        )
        loss_cfg = CELossConfig()

        loss_etp, _, norm_etp = _run_one_step_with_norm(engine_etp, loss_cfg, input_ids, labels)
        loss_ref, _, norm_ref = _run_one_step_with_norm(engine_ref, loss_cfg, input_ids, labels)

        torch.testing.assert_close(
            torch.tensor(loss_etp),
            torch.tensor(loss_ref),
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        gate_grad_ref = _get_param_grad(engine_ref, "layers.0.gate.weight")
        gate_grad_etp = _get_param_grad(engine_etp, "layers.0.gate.weight")
        torch.testing.assert_close(
            gate_grad_etp,
            gate_grad_ref,
            atol=BF16_GEMM_ATOL,
            rtol=BF16_GEMM_RTOL,
        )

        for module_suffix, fused_gate_up in (
            ("layers.0.experts.fused_w1w3", True),
            ("layers.0.experts.fused_w2", False),
        ):
            ref_grad = _get_param_grad(engine_ref, f"{module_suffix}.weight")
            etp_grad = _get_param_grad(engine_etp, f"{module_suffix}.weight")
            etp_module = _get_tpep_grouped_linear(engine_etp, module_suffix)
            expected_etp_grad = _slice_tpep_weight(etp_module, ref_grad, fused_gate_up=fused_gate_up)
            torch.testing.assert_close(
                etp_grad,
                expected_etp_grad,
                atol=BF16_GEMM_ATOL,
                rtol=BF16_GEMM_RTOL,
            )

        torch.testing.assert_close(
            norm_etp,
            norm_ref,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @parametrize.parametrize(
        "device,expert_tp_size",
        [
            ("cuda", 2),
        ],
    )
    def test_expert_tp_only_expert_grad_norm_matches_single_with_distinct_source_slices(
        self, device: str, expert_tp_size: int
    ) -> None:
        pg = self.create_pg(device)

        engine_ref = _build_engine(ep_size=1, expert_tp_size=1)
        engine_ref.init_model_weights()

        engine_etp = _build_engine(ep_size=1, expert_tp_size=expert_tp_size)
        engine_etp.init_model_weights()
        _sync_engine_weights(engine_ref, engine_etp)
        dist.barrier()

        input_ids, labels = _make_engine_input(
            torch.device(device, dist.get_rank() % torch.cuda.device_count()),
            seed_offset=dist.get_rank(),
        )
        loss_cfg = CELossConfig()

        _run_train_step_without_clip(engine_etp, loss_cfg, input_ids, labels)
        _run_train_step_without_clip(engine_ref, loss_cfg, input_ids, labels)
        _zero_non_expert_grads(engine_etp)
        _zero_non_expert_grads(engine_ref)

        norm_etp = engine_etp.clip_grad_norm(do_clip=False).detach().float().cpu()
        norm_ref = engine_ref.clip_grad_norm(do_clip=False).detach().float().cpu()
        torch.testing.assert_close(
            norm_etp,
            norm_ref,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @property
    def world_size(self) -> int:
        # ExpertTP-only topology: EP=1, TP=2, DP=1.
        return 2

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False


class TestMoETrainEngineTPEP(DeterministicDDPTestCase):
    """Verify EP+TP training matches single-GPU (EP=1, TP=1) forward and backward."""

    @parametrize.parametrize(
        "device,ep_size,expert_tp_size",
        [
            ("cuda", 2, 2),
        ],
    )
    def test_tpep_forward_backward_matches_single(
        self, device: str, ep_size: int, expert_tp_size: int
    ) -> None:
        """Loss and gate gradients with EP+TP must match the EP=1, TP=1 baseline."""
        pg = self.create_pg(device)

        # ------------------------------------------------------------------
        # Build reference engine: EP=1, TP=1 (world acts as pure DP).
        # ------------------------------------------------------------------
        engine_ref = _build_engine(ep_size=1, expert_tp_size=1)
        engine_ref.init_model_weights()

        # ------------------------------------------------------------------
        # Build EP+TP engine.
        # ------------------------------------------------------------------
        engine_tpep = _build_engine(ep_size=ep_size, expert_tp_size=expert_tp_size)
        engine_tpep.init_model_weights()

        # ------------------------------------------------------------------
        # Sync weights by explicitly slicing full expert weights into the real
        # TP column/row shards used by GroupedLinear.
        # ------------------------------------------------------------------
        _sync_engine_weights(engine_ref, engine_tpep)
        dist.barrier()

        # ------------------------------------------------------------------
        # Prepare shared input (identical on all ranks – no SP).
        # ------------------------------------------------------------------
        input_ids, labels = _make_engine_input(torch.device(device, dist.get_rank() % torch.cuda.device_count()))
        loss_cfg = CELossConfig()

        # Run EP+TP step.
        loss_tpep, grads_tpep = _run_one_step(engine_tpep, loss_cfg, input_ids, labels)

        # Run reference step.
        loss_ref, grads_ref = _run_one_step(engine_ref, loss_cfg, input_ids, labels)

        # ------------------------------------------------------------------
        # Assert losses match.
        # ------------------------------------------------------------------
        if dist.get_rank() == 0:
            torch.testing.assert_close(
                torch.tensor(loss_tpep),
                torch.tensor(loss_ref),
                atol=BF16_ATOL,
                rtol=BF16_RTOL,
            )

        # ------------------------------------------------------------------
        # Assert gate gradients match (key non-expert parameter).
        # ------------------------------------------------------------------
        if grads_tpep and grads_ref:
            for name in grads_ref:
                if name not in grads_tpep:
                    continue
                g_tpep = grads_tpep[name]
                g_ref = grads_ref[name]
                if dist.get_rank() == 0:
                    try:
                        torch.testing.assert_close(
                            g_tpep,
                            g_ref,
                            atol=BF16_GEMM_ATOL,
                            rtol=BF16_GEMM_RTOL,
                        )
                    except AssertionError as exc:
                        max_diff = (g_tpep - g_ref).abs().max().item()
                        raise AssertionError(
                            f"Gate gradient mismatch for '{name}': "
                            f"max_abs_diff={max_diff:.4e}, EP+TP shape={g_tpep.shape}, ref shape={g_ref.shape}"
                        ) from exc

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @parametrize.parametrize(
        "device,ep_size,expert_tp_size",
        [
            ("cuda", 2, 2),
        ],
    )
    def test_tpep_expert_gradients_match_single_with_distinct_expert_tp_data(
        self, device: str, ep_size: int, expert_tp_size: int
    ) -> None:
        """Expert TP shards should match the corresponding single-model expert gradients."""
        pg = self.create_pg(device)

        engine_ref = _build_engine(ep_size=1, expert_tp_size=1)
        engine_ref.init_model_weights()

        engine_tpep = _build_engine(ep_size=ep_size, expert_tp_size=expert_tp_size)
        engine_tpep.init_model_weights()
        _sync_engine_weights(engine_ref, engine_tpep)
        dist.barrier()

        input_ids, labels = _make_engine_input(
            torch.device(device, dist.get_rank() % torch.cuda.device_count()),
            seed_offset=dist.get_rank(),
        )
        loss_cfg = CELossConfig()

        _run_one_step(engine_tpep, loss_cfg, input_ids, labels)
        _run_one_step(engine_ref, loss_cfg, input_ids, labels)

        ref_grad = _get_param_grad(engine_ref, "layers.0.experts.fused_w1w3.weight")
        tpep_grad = _get_param_grad(engine_tpep, "layers.0.experts.fused_w1w3.weight")
        tpep_module = _get_tpep_grouped_linear(engine_tpep, "layers.0.experts.fused_w1w3")
        expected_tpep_grad = _slice_tpep_weight(tpep_module, ref_grad, fused_gate_up=True)

        torch.testing.assert_close(
            tpep_grad,
            expected_tpep_grad,
            atol=BF16_GEMM_ATOL,
            rtol=BF16_GEMM_RTOL,
        )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @parametrize.parametrize(
        "device,ep_size,expert_tp_size",
        [
            ("cuda", 2, 2),
        ],
    )
    def test_tpep_replicated_gradients_and_norm_match_single_with_distinct_expert_tp_data(
        self, device: str, ep_size: int, expert_tp_size: int
    ) -> None:
        """Non-expert replicas and grad norm should match the single-model baseline."""
        pg = self.create_pg(device)

        engine_ref = _build_engine(ep_size=1, expert_tp_size=1)
        engine_ref.init_model_weights()

        engine_tpep = _build_engine(ep_size=ep_size, expert_tp_size=expert_tp_size)
        engine_tpep.init_model_weights()
        _sync_engine_weights(engine_ref, engine_tpep)
        dist.barrier()

        input_ids, labels = _make_engine_input(
            torch.device(device, dist.get_rank() % torch.cuda.device_count()),
            seed_offset=dist.get_rank(),
        )
        loss_cfg = CELossConfig()

        _, _, norm_tpep = _run_one_step_with_norm(engine_tpep, loss_cfg, input_ids, labels)
        _, _, norm_ref = _run_one_step_with_norm(engine_ref, loss_cfg, input_ids, labels)

        gate_grad_ref = _get_param_grad(engine_ref, "layers.0.gate.weight")
        gate_grad_tpep = _get_param_grad(engine_tpep, "layers.0.gate.weight")

        torch.testing.assert_close(
            gate_grad_tpep,
            gate_grad_ref,
            atol=BF16_GEMM_ATOL,
            rtol=BF16_GEMM_RTOL,
        )
        torch.testing.assert_close(
            norm_tpep,
            norm_ref,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @parametrize.parametrize(
        "device,ep_size,expert_tp_size",
        [
            ("cuda", 2, 2),
        ],
    )
    def test_tpep_expert_only_grad_norm_matches_single_with_distinct_expert_tp_data(
        self, device: str, ep_size: int, expert_tp_size: int
    ) -> None:
        """Expert-only grad norm must sum norm square across EP and expert TP shards."""
        pg = self.create_pg(device)

        engine_ref = _build_engine(ep_size=1, expert_tp_size=1)
        engine_ref.init_model_weights()

        engine_tpep = _build_engine(ep_size=ep_size, expert_tp_size=expert_tp_size)
        engine_tpep.init_model_weights()
        _sync_engine_weights(engine_ref, engine_tpep)
        dist.barrier()

        input_ids, labels = _make_engine_input(
            torch.device(device, dist.get_rank() % torch.cuda.device_count()),
            seed_offset=dist.get_rank(),
        )
        loss_cfg = CELossConfig()

        _run_train_step_without_clip(engine_tpep, loss_cfg, input_ids, labels)
        _run_train_step_without_clip(engine_ref, loss_cfg, input_ids, labels)
        _zero_non_expert_grads(engine_tpep)
        _zero_non_expert_grads(engine_ref)

        norm_tpep = engine_tpep.clip_grad_norm(do_clip=False).detach().float().cpu()
        norm_ref = engine_ref.clip_grad_norm(do_clip=False).detach().float().cpu()

        torch.testing.assert_close(
            norm_tpep,
            norm_ref,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @parametrize.parametrize(
        "device,ep_size,expert_tp_size",
        [
            ("cuda", 2, 2),
        ],
    )
    def test_tpep_training_stability(self, device: str, ep_size: int, expert_tp_size: int) -> None:
        """EP+TP training should produce finite losses and decreasing trend."""
        pg = self.create_pg(device)

        engine = _build_engine(ep_size=ep_size, expert_tp_size=expert_tp_size)
        engine.init_model_weights()

        input_ids, labels = _make_engine_input(torch.device(device, dist.get_rank() % torch.cuda.device_count()))
        loss_cfg = CELossConfig()

        losses: list[float] = []
        for _ in range(4):
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
            shifted_labels = labels.to(DEVICE)
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
            loss_ctx_list = LossContext.build_batches([loss_ctx])
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx_list[0]})]
            step_info = engine.train_step(engine_input)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            losses.append(step_info["logs_info"]["reduced_llm_loss"])

        if dist.get_rank() == 0:
            for i, loss_val in enumerate(losses):
                self.assertTrue(
                    torch.isfinite(torch.tensor(loss_val)),
                    f"Loss at step {i} is not finite: {loss_val}",
                )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @property
    def world_size(self) -> int:
        # EP=2, TP=2, DP=1 → 4 GPUs
        return 4

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
