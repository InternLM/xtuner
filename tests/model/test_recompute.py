"""Gradient checkpointing and recompute-unit regression tests.

TestCheckpointWrapper
    test_wrapper_is_transparent_to_state_dict_and_attributes: 包裹后参数名/state_dict/属性访问不变。
    test_non_tensor_signature_preserves_gradients: 关键字参数 + dict 返回值下梯度与不重算一致。
    test_wrapper_forwards_container_protocols: 被包裹模块的 len/iter/in/索引协议在包裹后仍可用。
    test_wrapper_does_not_claim_protocols_the_module_lacks: 被包裹模块没有的协议不会出现在包裹层上。
TestRecomputeCfgResolution
    test_unset_cfg_keeps_full_recompute: `None` 不改变显存行为，解析为不留驻。
    test_true_selects_every_supported_unit: `True` 选中模型声明的全部 unit。
    test_explicit_units_select_only_themselves: 显式 list 只选中对应 unit。
    test_string_units_are_accepted: 配置文件里的字符串能解析成 RecomputeUnit。
    test_unsupported_unit_is_rejected: 模型不支持的 unit 在构造时报错并列出支持项。
    test_disable_propagates_into_nested_configs: `False` 递归关闭嵌套子模型配置。
    test_disable_reaches_every_sub_model_of_a_real_compose_config: 真实 compose 配置的三个子配置都被关闭。
    test_units_round_trip_through_json: enum 序列化成可读字符串并能读回。
TestDeclaredTargets
    test_declared_targets_resolve: 声明表里的 op 名与 callable 名都能解析到真实对象。
    test_no_unit_names_the_method_that_holds_most_compilation: 没有 unit 点名承载最多编译的那个方法。
TestUnitCostIsProportionate
    test_an_op_identity_unit_costs_no_compilation: KeptOps 不改动编译集合。
    test_a_callable_unit_keeps_its_callers_compiled: KeptCallables 只退出自身，调用者仍编译。
    test_no_unit_withdraws_the_method_that_holds_most_compilation: 没有 unit 撤出编译占比最大的方法。
    test_attention_is_kept_by_op_identity: attention 走 op identity 而非撤出 callable。
"""


import ast
import inspect
import pydoc
import textwrap

import pytest
import torch
from torch import nn

from xtuner.v1.model.base import BaseModel, TorchCompileOption, XTunerBaseModelConfig, _disable_nested_switch
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLMoE30BA3Config
from xtuner.v1.model.dense.dense import DENSE_RECOMPUTE_CFG
from xtuner.v1.model.moe.moe import MOE_RECOMPUTE_CFG, MoE, MoEConfig
from xtuner.v1.model.utils import (
    KeptCallables,
    KeptOps,
    RecomputeUnit,
    apply_gradient_checkpointing,
    resolve_kept_ops,
)
from xtuner.v1.module.attention import MHAConfig
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


class _ContainerBlock(nn.Module):
    """A container module, the shape whose protocols the wrapper has to forward."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> nn.Module:
        return self.layers[index]

    def __iter__(self):
        return iter(self.layers)

    def __contains__(self, item: object) -> bool:
        return item in self.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


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
        assert torch.equal(wrapped.state_dict()["linear.weight"], plain.state_dict()["linear.weight"])
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

        assert torch.equal(x.grad, baseline_input_grad)
        assert torch.equal(wrapped.linear.weight.grad, plain.linear.weight.grad)

    def test_wrapper_forwards_container_protocols(self):
        # 特殊方法在类型上查找，__getattr__ 看不到，必须逐个转发；只转发 __getitem__ 时
        # iter() 会退化到序列协议，报出与真实原因无关的 "not subscriptable"。
        plain = _ContainerBlock()
        wrapped = apply_gradient_checkpointing(plain)

        assert len(wrapped) == 3
        assert list(wrapped) == list(plain)
        assert wrapped[0] is plain[0]
        assert plain[1] in wrapped

    def test_wrapper_does_not_claim_protocols_the_module_lacks(self):
        # 协议按被包裹类型逐个决定，不能无条件定义：`__len__` 一旦恒存在，`bool(wrapper)` 就会
        # 去调它，于是 `module or default`（nn.Module 恒为真）对任何非 Sized 模块都会抛错——
        # `BaseModel._fully_shard` 里的 `target = module or self` 正是这样被打挂的。
        wrapped = apply_gradient_checkpointing(_KeywordOnlyBlock())

        assert bool(wrapped) is True
        assert not hasattr(type(wrapped), "__len__")
        with pytest.raises(TypeError, match="CheckpointWrapper"):
            len(wrapped)


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


def _resolve_units(**overrides) -> set:
    with torch.device("meta"):
        return set(MoE(config=_build_tiny_moe_config(**overrides))._selected_recompute_units)


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
        assert _resolve_units(recompute_cfg=None) == set()

    def test_true_selects_every_supported_unit(self):
        assert _resolve_units(recompute_cfg=True) == set(MOE_RECOMPUTE_CFG)

    def test_explicit_units_select_only_themselves(self):
        assert _resolve_units(recompute_cfg=[RecomputeUnit.SAVE_MOE_GATE]) == {RecomputeUnit.SAVE_MOE_GATE}

    def test_string_units_are_accepted(self):
        # Configs arrive as JSON/py files where units are written as plain strings.
        assert _resolve_units(recompute_cfg=["save_attn"]) == {RecomputeUnit.SAVE_ATTN}

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

        assert model._selected_recompute_units == set()
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
        config = _build_tiny_moe_config(recompute_cfg=[RecomputeUnit.SAVE_ATTN, RecomputeUnit.SAVE_MOE_GATE])

        dumped = config.model_dump(mode="json")["recompute_cfg"]
        assert dumped == ["save_attn", "save_moe_gate"]

        restored = _build_tiny_moe_config(recompute_cfg=dumped)
        assert restored.recompute_cfg == [RecomputeUnit.SAVE_ATTN, RecomputeUnit.SAVE_MOE_GATE]


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


class TestDeclaredTargets:
    @pytest.mark.parametrize("target_map", [MOE_RECOMPUTE_CFG, DENSE_RECOMPUTE_CFG], ids=["moe", "dense"])
    def test_declared_targets_resolve(self, target_map):
        # A renamed method or op would not fail anywhere at runtime on its own: the unit would
        # simply keep nothing and the region would stay recomputed, silently costing the memory the
        # user asked to keep. Resolve every name a model declares so a rename fails here instead.
        for unit, target in target_map.items():
            if isinstance(target, KeptOps):
                # A build registers only one flash-attention version, so *some* name must resolve
                # rather than all of them.
                assert resolve_kept_ops(target.names), f"{unit} names no op that resolves: {target.names}"
            else:
                for name in target.names:
                    assert pydoc.locate(name) is not None, f"{unit} names {name}, which does not resolve"

    def test_no_unit_names_the_method_that_holds_most_compilation(self):
        # `_pre_moe_forward` exists to give `torch.compile` the ops on either side of attention.
        # Naming it would withdraw most of what an MoE layer compiles, which is the cost this
        # design exists to avoid -- attention is kept by op identity and the gate by its own
        # callable instead.
        for unit, target in MOE_RECOMPUTE_CFG.items():
            if isinstance(target, KeptCallables):
                assert _PRE_MOE_FORWARD not in target.names, f"{unit} withdraws {_PRE_MOE_FORWARD}"


_PRE_MOE_FORWARD = "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._pre_moe_forward"


class TestUnitCostIsProportionate:
    """What a unit costs follows from how it is resolved, and nothing costs a withdrawn method.

    An op-identity unit changes nothing about compilation. A callable unit is excluded from the
    compiled set, which a compiled caller sees as a graph break -- so the callers are relaxed to
    `fullgraph=False`, but they stay compiled. No unit withdraws `_pre_moe_forward`, where most of
    an MoE layer's compilation lives.
    """

    @staticmethod
    def _compile_cfg(**overrides) -> dict[str, TorchCompileOption]:
        # The tiny config disables compilation; this asks what would be compiled if it did not.
        config = _build_tiny_moe_config(**overrides).model_copy(update={"compile_cfg": None})
        with torch.device("meta"):
            return MoE(config=config).compile_cfg

    def test_an_op_identity_unit_costs_no_compilation(self):
        assert self._compile_cfg(recompute_cfg=[RecomputeUnit.SAVE_ATTN]) == self._compile_cfg()

    def test_a_callable_unit_keeps_its_callers_compiled(self):
        unit = RecomputeUnit.SAVE_MOE_GATE
        # Relaxed, not removed: the caller still compiles, it just splits at the excluded callee.
        # Being absent from `compile_cfg` is not enough to be outside the compiled set, because
        # Dynamo inlines a callee into whichever compiled caller reaches it.
        baseline = self._compile_cfg()
        relaxed = self._compile_cfg(recompute_cfg=[unit])

        assert set(relaxed) == set(baseline), "a unit must not remove a caller from the compiled set"
        assert not any(option.get("fullgraph") for option in relaxed.values())

    def test_no_unit_withdraws_the_method_that_holds_most_compilation(self):
        for unit in MOE_RECOMPUTE_CFG:
            assert _PRE_MOE_FORWARD in self._compile_cfg(recompute_cfg=[unit]), f"{unit} withdrew {_PRE_MOE_FORWARD}"

    def test_attention_is_kept_by_op_identity(self):
        # The attention kernel is a custom op, so it reaches the policy from inside a compiled
        # region -- which is why this unit costs nothing.
        config = _build_tiny_moe_config(recompute_cfg=[RecomputeUnit.SAVE_ATTN])
        with torch.device("meta"):
            model = MoE(config=config)
        assert model.kept_ops
        assert model.keeps_any_recompute_unit
