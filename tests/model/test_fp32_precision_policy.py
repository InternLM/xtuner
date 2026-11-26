"""FP32 declaration contract for `PrecisionPolicy`.

TestPrecisionPolicyDeclarations
    declarations resolve to runtime names, drive the checkpoint dtype, and are validated

TestPinnedGradReduction
    pinned parameters are reduced exactly once, on every model family that can own them
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig
from xtuner.v1.model.compose.base import BaseComposeModel
from xtuner.v1.model.dense.qwen3 import Qwen3Dense, Qwen3DenseConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig


def _tiny_dense_cfg() -> Qwen3DenseConfig:
    return Qwen3DenseConfig(
        vocab_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=2, num_key_value_heads=1, head_dim=8, qk_norm=True),
        tie_word_embeddings=False,
        compile_cfg=False,
    )


class _WeightOnlyModel(BaseModel):
    """Smallest model that can own a pinned parameter and a pinned buffer."""

    def __init__(self) -> None:
        super().__init__(XTunerBaseModelConfig())
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.bias = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.register_buffer("running_stat", torch.ones(4, dtype=torch.float32))
        self.fp32_param_names = ("weight",)
        self.fp32_buffer_names = ("running_stat",)
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]


class _GatedDeltaNetHolder(BaseModel):
    def __init__(self) -> None:
        super().__init__(XTunerBaseModelConfig())
        self.self_attn = GatedDeltaNetConfig(
            num_value_heads=2,
            num_key_heads=1,
            key_head_dim=8,
            value_head_dim=8,
            conv_kernel_dim=4,
            hidden_act="silu",
            rms_norm_eps=1e-6,
        ).build(hidden_size=16)
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]


class TestPrecisionPolicyDeclarations:
    def test_declarations_resolve_to_runtime_names(self):
        model = _WeightOnlyModel()
        assert model.precision.param_names == frozenset({"weight"})

    def test_gated_deltanet_declaration_is_in_runtime_names(self):
        # Qwen3.5 calls the module `self_attn` while the HF checkpoint calls it `linear_attn`; the
        # declaration follows the runtime name, so no HF key namespace leaks into the module.
        model = _GatedDeltaNetHolder()
        assert model.precision.param_names == frozenset({"self_attn.A_log", "self_attn.norm.weight"})

    def test_unknown_declaration_fails_loudly(self):
        model = _WeightOnlyModel()
        model.fp32_param_names = ("wieght",)
        try:
            model.precision.param_names
        except ValueError as exc:
            assert "wieght" in str(exc)
        else:
            raise AssertionError("a misspelt declaration must not resolve silently")

    def test_save_dtype_follows_the_runtime_declaration(self):
        model = _tiny_dense_cfg().build()
        # A real module declaration reaches the checkpoint through its runtime name, with no HF key
        # namespace involved.
        model.layers["0"].input_layernorm.fp32_param_names = ("weight",)
        assert model.precision.save_dtype("layers.0.input_layernorm.weight", torch.bfloat16, is_buffer=False) == (
            torch.float32
        )
        assert model.precision.save_dtype("layers.0.mlp.gate_proj.weight", torch.bfloat16, is_buffer=False) == (
            torch.bfloat16
        )

    def test_declared_buffer_keeps_dtype_while_others_follow_the_cast(self):
        model = _WeightOnlyModel()
        model._to_device_dtype(dtype=torch.bfloat16)
        assert model.running_stat.dtype == torch.float32
        assert model.weight.dtype == torch.bfloat16
        assert model.bias.dtype == torch.bfloat16

    def test_undeclared_buffer_keeps_its_runtime_dtype_on_save(self):
        model = _WeightOnlyModel()
        model.fp32_buffer_names = ()
        assert model.precision.save_dtype("running_stat", torch.float32, is_buffer=True) == torch.float32
        # A bf16 persistent buffer stays bf16; saving must not rewrite unrelated buffers.
        assert model.precision.save_dtype("running_stat", torch.bfloat16, is_buffer=True) == torch.bfloat16


class _TinyCompose(BaseComposeModel):
    """Compose shell that skips building a real vision tower / projector."""

    def __init__(self, children: tuple[BaseModel, BaseModel, BaseModel]) -> None:
        BaseModel.__init__(self, XTunerBaseModelConfig())
        self.language_model, self.vision_tower, self.multi_modal_projector = children


class TestPinnedGradReduction(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def _fully_shard(self, model: BaseModel) -> None:
        model.fully_shard(
            FSDPConfig(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, recompute_ratio=0, torch_compile=False)
        )

    def _pinned_weight(self, model: Qwen3Dense) -> nn.Parameter:
        model.layers["0"].input_layernorm.fp32_param_names = ("weight",)
        self._fully_shard(model)
        weight = model.layers["0"].input_layernorm.weight
        assert isinstance(weight, DTensor)
        assert weight.device_mesh == model.world_mesh
        assert all(isinstance(placement, Replicate) for placement in weight.placements)
        return weight

    def test_dense_pinned_grad_is_averaged_and_stays_in_sync(self):
        # The FP32 pin takes the parameter out of FSDP's param groups, so nobody reduces its gradient
        # unless `BaseModel.scale_and_reduce_grad` does. Different per-rank gradients make the mean
        # observable.
        torch.cuda.set_device(self.rank)
        self.create_pg("cuda")
        with torch.device("meta"):
            model = _tiny_dense_cfg().build()
        weight = self._pinned_weight(model)
        with torch.no_grad():
            weight.to_local().fill_(1.0)
        optimizer = torch.optim.SGD([weight], lr=0.1)

        expected_weight = 1.0
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            local_grad = float(self.rank + 1) * (step + 1)
            weight.grad = DTensor.from_local(
                torch.full_like(weight.to_local(), local_grad),
                weight.device_mesh,
                weight.placements,
                run_check=False,
            )
            model.scale_and_reduce_grad()
            expected_grad = (self.world_size + 1) / 2 * (step + 1)
            torch.testing.assert_close(
                weight.grad.to_local(), torch.full_like(weight.to_local(), expected_grad), rtol=0, atol=0
            )
            optimizer.step()
            expected_weight -= 0.1 * expected_grad
            torch.testing.assert_close(
                weight.to_local(), torch.full_like(weight.to_local(), expected_weight), rtol=1e-6, atol=1e-7
            )

    def test_compose_reduces_every_sub_model(self):
        # `BaseComposeModel.scale_and_reduce_grad` used to forward to `language_model` alone, so a
        # pinned vision or projector parameter drifted apart across ranks.
        torch.cuda.set_device(self.rank)
        self.create_pg("cuda")
        children = []
        for _ in range(3):
            child = _WeightOnlyModel().cuda()
            child.precision  # noqa: B018 - force the declarations to resolve before sharding
            self._fully_shard(child)
            children.append(child)
        compose = _TinyCompose(children)  # type: ignore[arg-type]

        for child in children:
            child.weight.grad = DTensor.from_local(
                torch.full_like(child.weight.to_local(), float(self.rank + 1)),
                child.weight.device_mesh,
                child.weight.placements,
                run_check=False,
            )
        compose.scale_and_reduce_grad()

        expected = (self.world_size + 1) / 2
        for child in children:
            local = child.weight.grad.to_local()
            torch.testing.assert_close(local, torch.full_like(local, expected), rtol=0, atol=0)
