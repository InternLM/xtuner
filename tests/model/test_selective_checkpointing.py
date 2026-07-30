"""Region-level selective checkpointing regression tests.

TestKeptRegions
    test_kept_region_reproduces_full_recompute: 留驻区间与全重算的输出/梯度逐位相同且梯度存活。
    test_unbalanced_interval_is_safe: end marker 不执行时只多留驻，不影响梯度。
    test_overlapping_intervals_keep_the_union: 区间重叠时按并集留驻。
    test_marker_outside_session_is_noop: 不在会话内时埋点不做任何事。
TestContractLayering
    test_module_layer_imports_the_contract_without_the_model_layer: 契约模块必须能被 module/ 层单独导入。
TestUnsupportedRegions
    test_a_second_model_still_gets_its_own_diagnosis: 诊断去重不跨模型，第二个模型仍会告警。
    test_in_place_op_in_kept_region_is_rejected: 留驻区间内的 in-place 写会明确报错而不是静默改梯度。
TestRegionRecomputeUnderDominoEP
    test_kept_region_matches_full_recompute_under_domino_ep: domino EP 下留驻区间与全重算数值一致。
    test_kept_region_matches_full_recompute_under_compile: compile 下同上，且不触发 cached-tensor-mutated。
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.moe import MoE, MoEConfig, SequenceContext
from xtuner.v1.model.utils import MarkerInterval, apply_selective_checkpointing, checkpoint_record
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import NoAuxRouterConfig
from xtuner.v1.utils import selective_checkpointing as contract


class _MarkedBlock(nn.Module):
    """Two linear stages separated by markers, so a region can be kept or recomputed."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        checkpoint_record("first.start")
        hidden = torch.tanh(self.first(x))
        checkpoint_record("second.start")
        hidden = torch.tanh(self.second(hidden))
        checkpoint_record("second.end")
        return hidden


class _OtherMarkedBlock(_MarkedBlock):
    """A second layer class, standing in for another model living in the same process."""


class _EmptyRegionBlock(nn.Module):
    """A layer whose declared region encloses no op the policy can keep.

    Stands in for a region whose contents run inside a compiled kernel: both markers fire, so the
    interval opens, yet nothing between them ever reaches the per-op policy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(x)
        checkpoint_record("empty.begin")
        checkpoint_record("empty.end")
        return hidden


class _InPlaceBlock(nn.Module):
    """A region whose body accumulates in place, which selective checkpointing cannot keep."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        checkpoint_record("region.start")
        hidden = self.linear(x)
        accumulator = torch.zeros_like(hidden)
        accumulator.add_(hidden)
        checkpoint_record("region.end")
        return accumulator


def _run_block(module: nn.Module, intervals) -> tuple[torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(0)
    for parameter in module.parameters():
        nn.init.normal_(parameter, std=0.5)
    wrapped = apply_selective_checkpointing(module, intervals)

    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    inputs.requires_grad_(True)
    output = wrapped(inputs)
    output.sum().backward()
    return output.detach().clone(), [parameter.grad.clone() for parameter in module.parameters()]


class TestKeptRegions:
    def test_kept_region_reproduces_full_recompute(self):
        # 留驻与重算两条路都必须是精确值，不是近似：任何差异都说明 save-list 与重算对不上。
        recomputed_out, recomputed_grads = _run_block(_MarkedBlock(), ())
        kept_out, kept_grads = _run_block(_MarkedBlock(), [("first.start", "second.start")])

        torch.testing.assert_close(kept_out, recomputed_out, atol=0.0, rtol=0.0)
        for kept, recomputed in zip(kept_grads, recomputed_grads):
            torch.testing.assert_close(kept, recomputed, atol=0.0, rtol=0.0)

        # 梯度断掉时 loss 依然有限、上面的比较也依然成立（两边都是 None/零），所以单独断言存活。
        assert len(kept_grads) == 4
        for grad in kept_grads:
            assert grad is not None
            assert torch.count_nonzero(grad) > 0

    def test_unbalanced_interval_is_safe(self):
        # end marker 落在没走到的分支上，只应该多留驻一段显存，绝不影响梯度。
        recomputed_out, recomputed_grads = _run_block(_MarkedBlock(), ())
        kept_out, kept_grads = _run_block(_MarkedBlock(), [("first.start", "never.reached")])

        torch.testing.assert_close(kept_out, recomputed_out, atol=0.0, rtol=0.0)
        for kept, recomputed in zip(kept_grads, recomputed_grads):
            torch.testing.assert_close(kept, recomputed, atol=0.0, rtol=0.0)

    def test_overlapping_intervals_keep_the_union(self):
        # 重叠区间用「活跃集合非空即留驻」定义，不做配对断言，因此嵌套/交叉都只是并集。
        recomputed_out, recomputed_grads = _run_block(_MarkedBlock(), ())
        kept_out, kept_grads = _run_block(
            _MarkedBlock(),
            [("first.start", "second.start"), ("first.start", "second.end")],
        )

        torch.testing.assert_close(kept_out, recomputed_out, atol=0.0, rtol=0.0)
        for kept, recomputed in zip(kept_grads, recomputed_grads):
            torch.testing.assert_close(kept, recomputed, atol=0.0, rtol=0.0)

    def test_marker_outside_session_is_noop(self):
        # 模型可以先埋点、后开启 recompute，埋点本身不能有任何可观察行为。
        checkpoint_record("first.start")

        module = _MarkedBlock()
        inputs = torch.zeros(3, 4, requires_grad=True)
        module(inputs).sum().backward()

        assert inputs.grad is not None


class TestContractLayering:
    def test_module_layer_imports_the_contract_without_the_model_layer(self):
        # 契约（含 marker session）之所以在 xtuner/v1/utils 而不是挨着 engine，就是因为
        # `checkpoint_record` 的调用点在 xtuner/v1/module 的 forward 里：一旦契约里出现
        # 指向 model/ 或 module/ 的 import，这条独立导入就会变成循环导入而失败。
        # 必须用干净的解释器：同进程里 xtuner.v1.model 早就被导入了，测不出这个性质。
        result = subprocess.run(
            [sys.executable, "-c", "import xtuner.v1.module.decoder_layer.moe_decoder_layer"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestUnsupportedRegions:
    def test_in_place_op_in_kept_region_is_rejected(self):
        # 留驻的张量会原样交给重算那趟，read-modify-write 会在重算时二次累加：loss 有限、
        # 没有任何报错，梯度却和重算路径不同。必须报错而不是静默算错。
        with pytest.raises(RuntimeError, match="in-place op"):
            _run_block(_InPlaceBlock(), [("region.start", "region.end")])

    def test_in_place_op_outside_kept_region_is_fine(self):
        out, grads = _run_block(_InPlaceBlock(), ())
        assert torch.count_nonzero(grads[0]) > 0
        assert torch.isfinite(out).all()

    def test_a_second_model_still_gets_its_own_diagnosis(self, monkeypatch):
        # 诊断做去重是对的（一个模型几十层会喊几十遍），但去重不能跨模型：一个进程里
        # 同时存在 actor / reference 或 compose 模型的两个塔时，第二个模型的告警被第一个
        # 静默掉，用户就又回到「配了 SAVE_ATTN 却什么都没发生」的处境。
        warnings: list[str] = []
        monkeypatch.setattr(contract.log_rank0, "warning", warnings.append)

        for module in (_MarkedBlock(), _OtherMarkedBlock()):
            wrapped = apply_selective_checkpointing(module, [("never.reached", "second.start")])
            wrapped(torch.zeros(3, 4, requires_grad=True)).sum().backward()

        assert len(warnings) == 2, warnings

    def test_open_interval_that_keeps_nothing_is_reported(self, monkeypatch):
        # 这是第四次「静默无操作」：markers 都跑了，report_unreached 因此不响；区间里的 op 全在
        # 编译区域里执行、根本到不了 policy，于是什么都没留驻，用户却收不到任何提示。
        warnings: list[str] = []
        monkeypatch.setattr(contract.log_rank0, "warning", warnings.append)

        wrapped = apply_selective_checkpointing(_EmptyRegionBlock(), [("empty.begin", "empty.end")])
        wrapped(torch.zeros(3, 4, requires_grad=True)).sum().backward()

        assert len(warnings) == 1, warnings
        assert "kept nothing resident" in warnings[0]

    def test_interval_working_in_another_layer_is_not_reported(self, monkeypatch):
        # 区间图会同时覆盖 dense 与 MoE 两种层（一个 MoE 模型两者都有），所以某个 marker 在
        # 其中一种层里不出现是正常的。按层告警会让每个配置正确的模型每步都刷告警。
        warnings: list[str] = []
        monkeypatch.setattr(contract.log_rank0, "warning", warnings.append)
        owner = nn.Module()
        intervals = [("first.start", "second.start")]

        layers = [
            apply_selective_checkpointing(_EmptyRegionBlock(), intervals, owner=owner),
            apply_selective_checkpointing(_MarkedBlock(), intervals, owner=owner),
        ]
        for layer in layers:
            layer(torch.zeros(3, 4, requires_grad=True)).sum().backward()

        assert warnings == []


def _build_moe_config(ep_size: int, dispatcher: str, compile_model: bool) -> MoEConfig:
    router_config = NoAuxRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )
    attention_config = MHAConfig(num_attention_heads=8, num_key_value_heads=8, head_dim=128)
    return MoEConfig(
        vocab_size=10240,
        max_position_embeddings=2048,
        pad_token_id=0,
        eos_token_id=0,
        num_hidden_layers=4,
        hidden_size=1024,
        intermediate_size=2048,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=attention_config,
        tie_word_embeddings=False,
        n_routed_experts=32,
        n_shared_experts=1,
        num_experts_per_tok=8,
        first_k_dense_replace=1,
        hidden_factor=1.0,
        moe_intermediate_size=512,
        router=router_config,
        ep_size=ep_size,
        dispatcher=dispatcher,
        compile_cfg=None if compile_model else False,
    )


class _RegionMoE(MoE):
    """A MoE whose decoder layers keep their dispatch/combine region resident.

    The markers stand in for the ones the model layer records: they bracket the dispatcher calls,
    which are the boundaries that survive ``torch.compile`` under EP.
    """

    _REGION = ("moe.dispatch", "moe.combine.end")

    @property
    def recompute_intervals(self) -> list[MarkerInterval]:
        return [self._REGION]

    def fully_shard(self, *args, **kwargs):
        for layer in self.layers.values():
            _record_markers_around_dispatch(layer)
        return super().fully_shard(*args, **kwargs)


def _record_markers_around_dispatch(layer: nn.Module) -> None:
    dispatcher = getattr(layer, "dispatcher", None)
    if dispatcher is None:  # dense layers of `first_k_dense_replace`
        return

    def mark(method_name: str, marker: str, before: bool):
        original = getattr(dispatcher, method_name)

        def marked(*args, **kwargs):
            if before:
                checkpoint_record(marker)
                return original(*args, **kwargs)
            result = original(*args, **kwargs)
            checkpoint_record(marker)
            return result

        setattr(dispatcher, method_name, marked)

    mark("dispatch", _RegionMoE._REGION[0], before=True)
    mark("combine_postprocess", _RegionMoE._REGION[1], before=False)


class TestRegionRecomputeUnderDominoEP(DeterministicDDPTestCase):
    """Keeping a region must not change what the layer computes.

    Under domino EP the layer interleaves micro-batches across four stage loops with async
    dispatch/combine handles, so the forward and the recompute pass must still emit the same
    per-overload op sequence for the save list to line up. A mismatch shows up as a wrong gradient
    rather than an error.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))

    @pytest.mark.gpu
    def test_kept_region_matches_full_recompute_under_domino_ep(self):
        self.create_pg("cuda")
        self._assert_region_matches_baseline(compile_model=False)

    @pytest.mark.gpu
    def test_kept_region_matches_full_recompute_under_compile(self):
        # compile 下 policy 看到的是 inductor 的 `out=` extern kernel。把它们留驻会踩
        # "Tensor cached during selective activation checkpoint has been mutated"，
        # 所以这条用例同时钉住 policy 里「mutable schema 一律重算」这条规则。
        self.create_pg("cuda")
        self._assert_region_matches_baseline(compile_model=True)

    def _assert_region_matches_baseline(self, compile_model: bool):
        ep_size = self.world_size
        loss_ref, grad_norm_ref, nonzero_ref = self._run_once(ep_size, compile_model, keep_region=False)
        loss_kept, grad_norm_kept, nonzero_kept = self._run_once(ep_size, compile_model, keep_region=True)

        self.assertGreater(nonzero_ref, 0)
        self.assertEqual(nonzero_kept, nonzero_ref)
        self.assertTrue(
            torch.allclose(loss_kept, loss_ref, atol=5e-3, rtol=0.0),
            f"kept-region loss {loss_kept.item()} diverged from full recompute {loss_ref.item()}",
        )
        rel = abs(grad_norm_kept - grad_norm_ref) / (grad_norm_ref + 1e-8)
        self.assertLess(rel, 5e-2, f"kept-region grad-norm rel diff {rel} too large")

    def _run_once(self, ep_size: int, compile_model: bool, keep_region: bool):
        num_mb = 2
        seq_len = 512
        config = _build_moe_config(ep_size, "all2all", compile_model)
        model_cls = _RegionMoE if keep_region else MoE
        with torch.device("meta"):
            model = model_cls(config=config)._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        model.fully_shard(
            fsdp_config=FSDPConfig(ep_size=ep_size, recompute_ratio=1.0, torch_compile=compile_model)
        )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model.init_weights()

        loss_cfg = CELossConfig()
        seq_ctx_list = []
        loss_ctx_list = []
        gen = torch.Generator(device="cuda").manual_seed(1234)
        for _ in range(num_mb):
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len + 1), device="cuda", generator=gen)
            seq_ctx_list.append(SequenceContext.from_input_ids(input_ids=(input_ids[:, :-1],)))
            loss_ctx_list.append(loss_cfg.build(data={"shifted_labels": input_ids[:, 1:]}, sp_mesh=None))
        loss_ctx_list = loss_cfg.loss_ctx_cls.build_batches(loss_ctx_list)

        out = model(seq_ctx=seq_ctx_list, loss_ctx=[{"lm": lc} for lc in loss_ctx_list])
        loss = out["loss"]
        loss.backward()

        # FSDP hands back DTensor gradients, which do not implement every aten op; compare the local
        # shards instead.
        grads = [
            p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
            for p in model.parameters()
            if p.grad is not None
        ]
        nonzero = sum(1 for g in grads if torch.count_nonzero(g) > 0)
        grad_norm = torch.nn.utils.get_total_norm(grads)
        dist.barrier()
        return loss.detach().float().cpu(), float(grad_norm), nonzero
