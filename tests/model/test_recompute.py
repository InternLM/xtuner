"""Gradient checkpointing and recompute-unit regression tests.

TestCheckpointWrapper
    test_wrapper_is_transparent_to_state_dict_and_attributes: 包裹后参数名/state_dict/属性访问不变。
    test_reentrant_is_the_default: 默认 original forward 在 no_grad 下执行。
    test_context_fn_requires_explicit_non_reentrant: selective checkpoint 必须显式选择 non-reentrant。
    test_non_tensor_signature_preserves_gradients: 关键字参数 + dict 返回值下梯度与不重算一致。
    test_root_parameter_names_can_be_normalized: 根模型递归产生的 wrapper FQN 可恢复为逻辑参数名。
TestRecomputeCfgResolution
    test_unset_cfg_keeps_full_recompute: `None` 不改变显存行为，解析为不留驻。
    test_true_selects_every_supported_unit: `True` 选中模型声明的全部 unit。
    test_explicit_units_select_only_themselves: 显式 list 只选中对应 unit。
    test_string_units_are_accepted: 配置文件里的字符串能解析成 SaveUnit。
    test_unsupported_unit_is_rejected: 模型不支持的 unit 在构造时报错并列出支持项。
    test_disable_propagates_into_nested_configs: `False` 递归关闭嵌套子模型配置。
    test_disable_reaches_every_sub_model_of_a_real_compose_config: 真实 compose 配置的三个子配置都被关闭。
    test_units_round_trip_through_json: enum 序列化成可读字符串并能读回。
TestRecomputeRatioMigration
    test_old_field_still_takes_effect: 旧的 fsdp_cfg.recompute_ratio 仍然生效。
    test_unset_old_field_leaves_the_new_one_alone: 没设旧字段时不覆盖新位置的值。
    test_setting_both_is_an_error: 两处都设时报错而不是静默择一。
TestDeclaredTargets
    test_declared_targets_resolve: 声明表里的 op 名与 callable 名都能解析到真实对象。
    test_no_unit_names_the_method_that_holds_most_compilation: 没有 unit 点名承载最多编译的那个方法。
TestUnitCostIsProportionate
    test_an_op_identity_unit_costs_no_compilation: KeptOps 不改动编译集合。
    test_a_callable_unit_keeps_its_callers_compiled: KeptCallables 只退出自身，调用者仍编译。
    test_no_unit_withdraws_the_method_that_holds_most_compilation: 没有 unit 撤出编译占比最大的方法。
    test_attention_is_kept_by_op_identity: attention 走 op identity 而非撤出 callable。
    test_input_tensors_reach_the_ambient_saved_tensor_hooks: 嵌套/关键字传入的输入也能进外层 hook。
"""

import pydoc
from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks

from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.base import BaseModel, TorchCompileOption, XTunerBaseModelConfig, _disable_nested_switch
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLMoE30BA3Config
from xtuner.v1.model.dense.dense import DENSE_RECOMPUTE_CFG
from xtuner.v1.model.moe.glm52.glm52 import GLM52_RECOMPUTE_CFG
from xtuner.v1.model.moe.moe import MOE_RECOMPUTE_CFG, MoE, MoEConfig
from xtuner.v1.model.utils import (
    KeptCallables,
    KeptOps,
    RecomputeConfig,
    SaveUnit,
    apply_gradient_checkpointing,
    resolve_kept_ops,
)
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import NoAuxRouterConfig
from xtuner.v1.utils import clean_param_name


class _KeywordOnlyBlock(nn.Module):
    """A forward shape that requires pytree adaptation with reentrant checkpointing.

    Tensors arrive nested in a dict and behind a keyword-only argument, and the result is returned
    as a dict rather than a tensor or a tuple of tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.tag = "block"

    def forward(self, inputs: dict[str, torch.Tensor], *, scale: float) -> dict[str, torch.Tensor]:
        return {"out": self.linear(inputs["x"]) * scale}


class _FlexibleBlock(nn.Module):
    """接受任意摆放的输入：位置的容器、字典、关键字参数，用来覆盖各种嵌套形状。"""

    def __init__(self) -> None:
        super().__init__()
        # 输入 4 维、输出 6 维：输出与输入形状不同，断言才不会把输出误当成输入。
        self.linear = nn.Linear(4, 6)

    def forward(self, inputs, *, scale: float, extra: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tensors = list(inputs.values()) if isinstance(inputs, dict) else list(inputs)
        if extra is not None:
            tensors.append(extra)
        return {"out": sum(self.linear(t) * scale for t in tensors)}


class _GradModeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.grad_modes: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.grad_modes.append(torch.is_grad_enabled())
        return self.linear(x)


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

    def test_reentrant_is_the_default(self):
        wrapped = apply_gradient_checkpointing(_GradModeBlock())
        wrapped(torch.randn(2, 4, requires_grad=True)).sum().backward()

        # Reentrant checkpoint runs the original pass without a graph, then replays it with grad.
        assert wrapped.grad_modes == [False, True]

    def test_context_fn_requires_explicit_non_reentrant(self):
        def context_fn():
            return nullcontext(), nullcontext()

        wrapped = apply_gradient_checkpointing(_KeywordOnlyBlock(), context_fn=context_fn)
        x = torch.randn(2, 4, requires_grad=True)

        with pytest.raises(ValueError, match="context_fn.*use_reentrant=False"):
            wrapped({"x": x}, scale=2.0)

        wrapped = apply_gradient_checkpointing(
            _KeywordOnlyBlock(),
            use_reentrant=False,
            context_fn=context_fn,
        )
        wrapped({"x": x}, scale=2.0)["out"].sum().backward()

        assert x.grad is not None
        assert wrapped.linear.weight.grad is not None

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

    def test_root_parameter_names_can_be_normalized(self):
        root = nn.Module()
        root.block = apply_gradient_checkpointing(_KeywordOnlyBlock())

        names = {clean_param_name(name) for name, _ in root.named_parameters()}

        assert names == {"block.linear.weight", "block.linear.bias"}

    @pytest.mark.parametrize(
        "make_call",
        [
            pytest.param(lambda block, x: block([x], scale=2.0), id="nested-in-list"),
            pytest.param(lambda block, x: block({"x": x}, scale=2.0), id="nested-in-dict"),
            pytest.param(lambda block, x: block([], scale=2.0, extra=x), id="passed-by-keyword"),
        ],
    )
    def test_input_tensors_reach_the_ambient_saved_tensor_hooks(self, make_call):
        # 激活 offload 是靠外层 saved_tensors_hooks 拿到层输入的，而 checkpoint 只把**顶层**
        # tensor 参数包成 SavedVariable（构造它才会触发 hook）。所以嵌套在容器里、或走关键字
        # 传进来的 tensor 会一个 hook 都不经过——offload 静默空转，梯度却完全正确，没有任何
        # 现象能暴露它。这里直接断言 hook 收得到。
        packed: list[int] = []

        class _Record(saved_tensors_hooks):
            # 按 data_ptr 认张量，不按 shape：区域的输出很容易和输入同形，
            # 按 shape 断言会把输出当成输入，测试变成恒绿。
            def __init__(self) -> None:
                super().__init__(lambda t: (packed.append(t.data_ptr()), t)[1], lambda t: t)

        wrapped = apply_gradient_checkpointing(_FlexibleBlock(), use_reentrant=False)
        x = torch.randn(2, 4, requires_grad=True)

        with _Record():
            make_call(wrapped, x)["out"].square().sum().backward()

        assert x.data_ptr() in packed


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
        assert _resolve_units(recompute_cfg=RecomputeConfig(save=None)) == set()

    def test_true_selects_every_supported_unit(self):
        assert _resolve_units(recompute_cfg=RecomputeConfig(save=True)) == set(MOE_RECOMPUTE_CFG)

    def test_explicit_units_select_only_themselves(self):
        assert _resolve_units(recompute_cfg=RecomputeConfig(save=[SaveUnit.MOE_GATE])) == {SaveUnit.MOE_GATE}

    def test_string_units_are_accepted(self):
        # Configs arrive as JSON/py files where units are written as plain strings.
        assert _resolve_units(recompute_cfg=RecomputeConfig(save=["attn"])) == {SaveUnit.ATTN}

    def test_unsupported_unit_is_rejected(self):
        # A model declaring no units cannot honour any selection, so this is a user configuration
        # error rather than something to silently drop. It surfaces at construction, before the run
        # spends anything on materializing and sharding weights.
        with pytest.raises(ValueError, match="does not support"):
            _ProbeModel(
                _ProbeConfig(text_config=_NestedProbeConfig(), recompute_cfg=RecomputeConfig(save=[SaveUnit.ATTN]))
            )

    def test_disable_propagates_into_nested_configs(self):
        # A sub-model resolves its own switch, so `False` on the outer config only means something
        # if it reaches the nested ones. `compile_cfg` must stay untouched: the walk is per switch.
        config = _ProbeConfig(text_config=_NestedProbeConfig(), recompute_cfg=RecomputeConfig(save=False))

        model = _ProbeModel(config)

        assert model._selected_recompute_units == set()
        assert config.text_config.recompute_cfg.save is False
        assert config.text_config.compile_cfg is None

    def test_disable_reaches_every_sub_model_of_a_real_compose_config(self):
        # The probe above has one nested config; a shipped compose config has three, one of them a
        # further-derived MoE config. Exercised on the config walk rather than through the model,
        # because constructing a 30B compose model is the expensive part and contributes nothing:
        # what can regress here is which nested configs the walk reaches.
        config = Qwen3VLMoE30BA3Config(recompute_cfg=RecomputeConfig(save=False))

        _disable_nested_switch(config, "recompute_cfg", subfield="save")

        for sub_config in (config.vision_config, config.projector_config, config.text_config):
            assert sub_config.recompute_cfg.save is False

    def test_units_round_trip_through_json(self):
        # Trainer resume reads the config back, and serialized runs are read by humans, so units
        # must survive as their readable names.
        config = _build_tiny_moe_config(recompute_cfg=RecomputeConfig(save=[SaveUnit.ATTN, SaveUnit.MOE_GATE]))

        dumped = config.model_dump(mode="json")["recompute_cfg"]
        assert dumped["save"] == ["attn", "moe_gate"]

        restored = _build_tiny_moe_config(recompute_cfg=RecomputeConfig(**dumped))
        assert restored.recompute_cfg.save == [SaveUnit.ATTN, SaveUnit.MOE_GATE]


class TestDeclaredTargets:
    @pytest.mark.parametrize(
        "target_map",
        [MOE_RECOMPUTE_CFG, DENSE_RECOMPUTE_CFG, GLM52_RECOMPUTE_CFG],
        ids=["moe", "dense", "glm52"],
    )
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
        assert self._compile_cfg(recompute_cfg=RecomputeConfig(save=[SaveUnit.ATTN])) == self._compile_cfg()

    def test_a_callable_unit_keeps_its_callers_compiled(self):
        unit = SaveUnit.MOE_GATE
        # Relaxed, not removed: the caller still compiles, it just splits at the excluded callee.
        # Being absent from `compile_cfg` is not enough to be outside the compiled set, because
        # Dynamo inlines a callee into whichever compiled caller reaches it.
        baseline = self._compile_cfg()
        relaxed = self._compile_cfg(recompute_cfg=RecomputeConfig(save=[unit]))

        assert set(relaxed) == set(baseline), "a unit must not remove a caller from the compiled set"
        assert not any(option.get("fullgraph") for option in relaxed.values())

    def test_no_unit_withdraws_the_method_that_holds_most_compilation(self):
        for unit in MOE_RECOMPUTE_CFG:
            assert _PRE_MOE_FORWARD in self._compile_cfg(recompute_cfg=RecomputeConfig(save=[unit])), (
                f"{unit} withdrew {_PRE_MOE_FORWARD}"
            )

    def test_attention_is_kept_by_op_identity(self):
        # The attention kernel is a custom op, so it reaches the policy from inside a compiled
        # region -- which is why this unit costs nothing.
        config = _build_tiny_moe_config(recompute_cfg=RecomputeConfig(save=[SaveUnit.ATTN]))
        with torch.device("meta"):
            model = MoE(config=config)
        assert model.kept_ops
        assert model.keeps_any_recompute_unit


class TestRecomputeRatioMigration:
    """`fsdp_cfg.recompute_ratio` 已迁到 `recompute_cfg.ratio`，旧字段留作过渡。"""

    def _model(self, fsdp_kwargs, **config_kwargs):
        model = _ProbeModel(_ProbeConfig(text_config=_NestedProbeConfig(), **config_kwargs))
        model._migrate_recompute_ratio(FSDPConfig(**fsdp_kwargs))
        return model

    def test_old_field_still_takes_effect(self):
        # 过渡期内旧配置不能突然失效，值要被搬到新位置。
        model = self._model({"recompute_ratio": 0.25})

        assert model.config.recompute_cfg.ratio == 0.25

    def test_unset_old_field_leaves_the_new_one_alone(self):
        # 旧字段的哨兵是 None 而不是 1.0，否则「没设过」会被当成「显式设成 1.0」，
        # 把用户在新位置写的值覆盖掉。
        model = self._model({}, recompute_cfg=RecomputeConfig(ratio=0.5))

        assert model.config.recompute_cfg.ratio == 0.5

    def test_setting_both_is_an_error(self):
        # 同一个设置有两个写法时，静默挑一个是最糟的处理。
        with pytest.raises(ValueError, match="both set"):
            self._model({"recompute_ratio": 0.25}, recompute_cfg=RecomputeConfig(ratio=0.5))
