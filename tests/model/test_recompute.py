"""Gradient checkpointing regression tests.

TestCheckpointWrapper
    test_wrapper_is_transparent_to_state_dict_and_attributes: 包裹后参数名/state_dict/属性访问不变。
    test_non_tensor_signature_preserves_gradients: 关键字参数 + dict 返回值下梯度与不重算一致。
TestDominoEPRecompute
    test_recompute_matches_baseline_under_domino_ep: domino EP 下重算与不重算的 loss/梯度一致。
TestRecomputeCfgResolution
    test_unset_cfg_keeps_full_recompute: `None` 不改变显存行为，解析为空区间。
    test_true_selects_every_supported_unit: `True` 选中模型声明的全部 unit。
    test_explicit_units_select_only_their_intervals: 显式 list 只解析出对应区间。
    test_string_units_are_accepted: 配置文件里的字符串能解析成 RecomputeUnit。
    test_unsupported_unit_is_rejected: 模型不支持的 unit 在构造时报错并列出支持项。
    test_disable_propagates_into_nested_configs: `False` 递归关闭嵌套子模型配置。
    test_disable_reaches_every_sub_model_of_a_real_compose_config: 真实 compose 配置的三个子配置都被关闭。
    test_units_round_trip_through_json: enum 序列化成可读字符串并能读回。
TestMarkerVocabulary
    test_declared_intervals_have_markers: default_recompute_cfg 引用的 marker 都真实埋点。
    test_micro_batch_path_covers_the_same_operations: 单路与 domino 路每个区间覆盖的算子一致。
"""

import ast
import inspect
import os
import textwrap

import pytest
import torch
import torch.distributed as dist
from torch import nn

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig, _disable_nested_switch
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLMoE30BA3Config
from xtuner.v1.model.dense.dense import DENSE_RECOMPUTE_CFG
from xtuner.v1.model.moe.glm52 import dsa_mla as glm52_dsa_mla
from xtuner.v1.model.moe.glm52.glm52 import GLM52_RECOMPUTE_CFG
from xtuner.v1.model.moe.moe import MOE_RECOMPUTE_CFG, MoE, MoEConfig, SequenceContext
from xtuner.v1.model.utils import (
    RecomputeUnit,
    apply_gradient_checkpointing,
)
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer import dense_decoder_layer, moe_decoder_layer
from xtuner.v1.module.router import NoAuxRouterConfig


class _KeywordOnlyBlock(nn.Module):
    """A forward shape only the non-reentrant implementation supports.

    Tensors arrive nested in a dict and behind a keyword-only argument, and the result is returned
    as a dict rather than a tensor or a tuple of tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.tag = "block"

    def forward(self, inputs: dict[str, torch.Tensor], *, scale: float) -> dict[str, torch.Tensor]:
        return {"out": self.linear(inputs["x"]) * scale}


class TestCheckpointWrapper:
    def test_wrapper_is_transparent_to_state_dict_and_attributes(self):
        # 包裹层不能出现在参数名里，否则 checkpoint 的存/取与非重算模型不兼容。
        plain = _KeywordOnlyBlock()
        wrapped = apply_gradient_checkpointing(_KeywordOnlyBlock())
        wrapped.load_state_dict(plain.state_dict())

        assert sorted(wrapped.state_dict()) == sorted(plain.state_dict())
        assert sorted(name for name, _ in wrapped.named_parameters()) == sorted(
            name for name, _ in plain.named_parameters()
        )
        torch.testing.assert_close(wrapped.state_dict()["linear.weight"], plain.state_dict()["linear.weight"])
        assert wrapped.tag == "block"

    def test_non_tensor_signature_preserves_gradients(self):
        # 非 tensor 签名下梯度必须与不重算完全一致。
        torch.manual_seed(0)
        plain = _KeywordOnlyBlock()
        wrapped = apply_gradient_checkpointing(_KeywordOnlyBlock())
        wrapped.load_state_dict(plain.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        plain({"x": x}, scale=2.0)["out"].square().sum().backward()
        baseline_input_grad, x.grad = x.grad.clone(), None

        wrapped({"x": x}, scale=2.0)["out"].square().sum().backward()

        torch.testing.assert_close(x.grad, baseline_input_grad)
        torch.testing.assert_close(wrapped.linear.weight.grad, plain.linear.weight.grad)


def _build_moe_config(ep_size: int, dispatcher: str) -> MoEConfig:
    router_config = NoAuxRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
    )
    attention_config = MHAConfig(num_attention_heads=32, num_key_value_heads=32, head_dim=16)
    return MoEConfig(
        vocab_size=10240,
        max_position_embeddings=2048,
        pad_token_id=0,
        eos_token_id=0,
        num_hidden_layers=4,
        hidden_size=512,
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
        compile_cfg=False,
    )


class TestDominoEPRecompute(DeterministicDDPTestCase):
    """Regression guard for non-reentrant recompute under domino EP.

    ``checkpoint_wrapper`` used to pin ``CheckpointImpl.REENTRANT`` for the decoder layers. The
    reentrant implementation only tracks gradients for top-level ``torch.Tensor`` arguments, which
    is what forced the decoder layers to pass hidden states positionally and to return a flat tuple.
    This test asserts that, under domino EP (``intra_layer_micro_batch > 1``, the case that pinned
    the choice), enabling recompute reproduces the no-recompute baseline loss and gradients.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))

    @pytest.mark.gpu
    def test_recompute_matches_baseline_under_domino_ep(self):
        self.create_pg("cuda")
        ep_size = self.world_size

        loss_ref, grad_norm_ref, finite_ref = self._run_once(ep_size, "all2all", recompute_ratio=0.0)
        loss_rc, grad_norm_rc, finite_rc = self._run_once(ep_size, "all2all", recompute_ratio=1.0)

        # A broken checkpoint graph shows up as non-finite or missing gradients.
        self.assertTrue(finite_ref)
        self.assertTrue(finite_rc)

        # Recompute is mathematically equivalent to the baseline; only bf16 rounding and the
        # nondeterministic async EP reduction order separate them, so compare with a band that
        # is loose enough for that noise but tight enough to catch a corrupted gradient.
        self.assertTrue(
            torch.allclose(loss_rc, loss_ref, atol=5e-3, rtol=0.0),
            f"recompute loss {loss_rc.item()} diverged from baseline {loss_ref.item()}",
        )
        rel = abs(grad_norm_rc - grad_norm_ref) / (grad_norm_ref + 1e-8)
        self.assertLess(rel, 5e-2, f"recompute grad-norm rel diff {rel} too large")

    def _run_once(self, ep_size: int, dispatcher: str, recompute_ratio: float):
        num_mb = 2
        seq_len = 512
        config = _build_moe_config(ep_size, dispatcher)
        with torch.device("meta"):
            model = MoE(config=config)._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        model.fully_shard(
            fsdp_config=FSDPConfig(ep_size=ep_size, recompute_ratio=recompute_ratio, torch_compile=False)
        )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model.init_weights()

        loss_cfg = CELossConfig()
        seq_ctx_list = []
        loss_ctx_list = []
        # Fixed seed so the baseline and the recompute model consume identical data.
        gen = torch.Generator(device="cuda").manual_seed(1234)
        for _ in range(num_mb):
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len + 1), device="cuda", generator=gen)
            seq_ctx_list.append(SequenceContext.from_input_ids(input_ids=(input_ids[:, :-1],)))
            loss_ctx_list.append(loss_cfg.build(data={"shifted_labels": input_ids[:, 1:]}, sp_mesh=None))
        loss_ctx_list = loss_cfg.loss_ctx_cls.build_batches(loss_ctx_list)

        out = model(seq_ctx=seq_ctx_list, loss_ctx=[{"lm": lc} for lc in loss_ctx_list])
        loss = out["loss"]
        loss.backward()

        grad_sq = torch.zeros((), device="cuda", dtype=torch.float32)
        all_finite = True
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            all_finite = all_finite and bool(torch.isfinite(g).all())
            grad_sq += g.float().pow(2).sum()
        dist.all_reduce(grad_sq, op=dist.ReduceOp.SUM)
        grad_norm = grad_sq.sqrt().item()

        loss_val = loss.detach().float()
        dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)

        del model, out, loss
        torch.cuda.empty_cache()
        return loss_val, grad_norm, all_finite


def _build_tiny_moe_config(**overrides) -> MoEConfig:
    """A MoE small enough to instantiate on the meta device in a config-only test."""
    router_config = NoAuxRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )
    return MoEConfig(
        vocab_size=256,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=0,
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=4, num_key_value_heads=4, head_dim=16),
        tie_word_embeddings=False,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        hidden_factor=1.0,
        moe_intermediate_size=64,
        router=router_config,
        compile_cfg=False,
        **overrides,
    )


def _resolve_intervals(**overrides) -> list[tuple[str, str]]:
    with torch.device("meta"):
        return MoE(config=_build_tiny_moe_config(**overrides)).recompute_intervals


class _NestedProbeConfig(XTunerBaseModelConfig):
    """Stand-in for a sub-model config, as a compose model nests one."""


class _ProbeConfig(XTunerBaseModelConfig):
    text_config: XTunerBaseModelConfig


class _ProbeModel(BaseModel):
    """A model that contributes nothing but ``BaseModel.__init__``'s config resolution."""

    config: _ProbeConfig


class TestRecomputeCfgResolution:
    def test_unset_cfg_keeps_full_recompute(self):
        # `None` must not change the memory profile of an existing training run: unlike `compile_cfg`,
        # it resolves to "retain nothing" rather than to the model's declared units.
        assert _resolve_intervals(recompute_cfg=None) == []

    def test_true_selects_every_supported_unit(self):
        intervals = _resolve_intervals(recompute_cfg=True)

        expected = [interval for unit_intervals in MOE_RECOMPUTE_CFG.values() for interval in unit_intervals]
        assert intervals == expected

    def test_explicit_units_select_only_their_intervals(self):
        intervals = _resolve_intervals(recompute_cfg=[RecomputeUnit.SAVE_MOE_DISPATCH])

        assert intervals == MOE_RECOMPUTE_CFG[RecomputeUnit.SAVE_MOE_DISPATCH]

    def test_string_units_are_accepted(self):
        # Configs arrive as JSON/py files where units are written as plain strings.
        assert _resolve_intervals(recompute_cfg=["save_attn"]) == MOE_RECOMPUTE_CFG[RecomputeUnit.SAVE_ATTN]

    def test_unsupported_unit_is_rejected(self):
        # A model declaring no units cannot honour any selection, so this is a user configuration
        # error rather than something to silently drop. It surfaces at construction, before the run
        # spends anything on materializing and sharding weights.
        with pytest.raises(ValueError, match="does not support"):
            _ProbeModel(_ProbeConfig(text_config=_NestedProbeConfig(), recompute_cfg=[RecomputeUnit.SAVE_ATTN]))

    def test_disable_propagates_into_nested_configs(self):
        # A sub-model resolves its own switch, so `False` on the outer config only means something
        # if it reaches the nested ones. `compile_cfg` must stay untouched: the walk is per switch.
        config = _ProbeConfig(text_config=_NestedProbeConfig(), recompute_cfg=False)

        model = _ProbeModel(config)

        assert model.recompute_intervals == []
        assert config.text_config.recompute_cfg is False
        assert config.text_config.compile_cfg is None

    def test_disable_reaches_every_sub_model_of_a_real_compose_config(self):
        # The probe above has one nested config; a shipped compose config has three, one of them a
        # further-derived MoE config. Exercised on the config walk rather than through the model,
        # because constructing a 30B compose model is the expensive part and contributes nothing:
        # what can regress here is which nested configs the walk reaches.
        config = Qwen3VLMoE30BA3Config(recompute_cfg=False)

        _disable_nested_switch(config, "recompute_cfg")

        for sub_config in (config.vision_config, config.projector_config, config.text_config):
            assert sub_config.recompute_cfg is False

    def test_units_round_trip_through_json(self):
        # Trainer resume reads the config back, and serialized runs are read by humans, so units
        # must survive as their readable names.
        config = _build_tiny_moe_config(recompute_cfg=[RecomputeUnit.SAVE_ATTN, RecomputeUnit.SAVE_MLP])

        dumped = config.model_dump(mode="json")["recompute_cfg"]
        assert dumped == ["save_attn", "save_mlp"]

        restored = _build_tiny_moe_config(recompute_cfg=dumped)
        assert restored.recompute_cfg == [RecomputeUnit.SAVE_ATTN, RecomputeUnit.SAVE_MLP]


def _recorded_markers(func) -> set[str]:
    """Marker names a function passes to ``checkpoint_record``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "checkpoint_record"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


def _region_coverage(func) -> dict[str, set[str]]:
    """Which of the layer's own operations each marker region encloses.

    Regions are keyed by the shared prefix of a ``<name>.begin`` / ``<name>.end`` pair, and an operation is a call on
    ``self`` -- ``self.experts(...)``, ``self.dispatcher.dispatch(...)``. Calls on tensors are ignored: a region is
    defined by the sub-modules it covers, not by the reshapes threaded between them.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    # `ast.walk` is breadth-first; the marker state machine needs source order.
    nodes = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )

    coverage: dict[str, set[str]] = {}
    active: set[str] = set()
    for node in nodes:
        name = _called_name(node)
        if name == "checkpoint_record" and node.args and isinstance(node.args[0], ast.Constant):
            region, _, edge = node.args[0].value.rpartition(".")
            if edge == "begin":
                active.add(region)
                coverage.setdefault(region, set())
            elif edge == "end":
                active.discard(region)
        elif name is not None and name.startswith("self."):
            for region in active:
                coverage[region].add(name.removeprefix("self."))
    return coverage


def _called_name(node: ast.Call) -> str | None:
    """Dotted name of a call target, e.g. ``self.dispatcher.dispatch``."""
    parts: list[str] = []
    target: ast.expr = node.func
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if not isinstance(target, ast.Name):
        return None
    parts.append(target.id)
    return ".".join(reversed(parts))


class TestMarkerVocabulary:
    @pytest.mark.parametrize(
        "interval_map",
        [MOE_RECOMPUTE_CFG, DENSE_RECOMPUTE_CFG, GLM52_RECOMPUTE_CFG],
        ids=["moe", "dense", "glm52"],
    )
    def test_declared_intervals_have_markers(self, interval_map):
        # A renamed marker would not fail anywhere at runtime: the interval would simply never open
        # and the region would stay recomputed, silently costing the memory the user asked to keep.
        recorded: set[str] = set()
        for layer_module in (moe_decoder_layer, dense_decoder_layer, glm52_dsa_mla):
            for _, member in inspect.getmembers(layer_module, inspect.isclass):
                if member.__module__ != layer_module.__name__:
                    continue
                for _, method in inspect.getmembers(member, inspect.isfunction):
                    recorded |= _recorded_markers(method)

        declared = {name for intervals in interval_map.values() for interval in intervals for name in interval}
        assert declared <= recorded, f"markers referenced but never recorded: {sorted(declared - recorded)}"

    def test_micro_batch_path_covers_the_same_operations(self):
        # `_micro_batch_forward` re-implements the dispatch/combine chain across four stage loops.
        # Comparing marker names alone would pass even if the domino path wrapped entirely different
        # operations, so compare what each region actually encloses.
        single = _region_coverage(moe_decoder_layer.MoEDecoderLayer._forward)
        micro_batch = _region_coverage(moe_decoder_layer.MoEDecoderLayer._micro_batch_forward)

        assert single == micro_batch
