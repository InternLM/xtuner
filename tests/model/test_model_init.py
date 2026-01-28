import torch
from torch import nn
from torch.distributed.tensor import DTensor
from typing import cast

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEGate
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear


def _get_model_config() -> Qwen3MoEConfig:
    return Qwen3MoEConfig(
        vocab_size=1024,
        max_position_embeddings=1024,
        pad_token_id=0,
        bos_token_id=151643,
        eos_token_id=151645,
        num_hidden_layers=2,
        hidden_size=1024,
        intermediate_size=2048,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        hidden_act="silu",
        attention=MHAConfig(
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=128,
        ),
        tie_word_embeddings=False,
        n_routed_experts=8,
        n_shared_experts=0,
        num_experts_per_tok=1,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=256,
        router=GreedyRouterConfig(
            scoring_func="softmax",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        ),
    )


@torch.no_grad()
def _examine_weights(model: nn.Module):
    def _check_params_mean_std(param: torch.Tensor, expected_mean: float, expected_std: float):
        param = param.full_tensor() if isinstance(param, DTensor) else param
        actual_mean = torch.mean(param)
        actual_std = torch.std(param)
        assert torch.allclose(
            actual_mean,
            torch.tensor(expected_mean, device=actual_mean.device),
            atol=1e-3,
            rtol=1e-3,
        ), (
            f"model init weight mean assertion failed for param {name}, "
            f"expected {expected_mean:.1f}, got {actual_mean.item():.2f}"
        )
        assert torch.allclose(
            actual_std,
            torch.tensor(expected_std, device=actual_std.device),
            atol=1e-3,
            rtol=1e-3,
        ), (
            f"model init weight std assertion failed for param {name}, "
            f"expected {expected_std:.1f}, got {actual_std.item():.2f}"
        )

    def _check_params_constant(param: torch.Tensor, expected_value: float):
        param = param.full_tensor() if isinstance(param, DTensor) else param
        assert torch.all(torch.isclose(
            param,
            torch.tensor(expected_value, device=param.device),
            rtol=1e-5,
            atol=1e-5
        )), (
            f"Constant initialization assertion failed for param {name}, "
            f"expected all values to be ~{expected_value}"
        )

    examined_params = set()

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, MoEGate, GroupedLinear, TileWiseFloat8GroupedLinear)):
            # linear weights are initialized with N(0, 0.02)
            _check_params_mean_std(cast(torch.Tensor, module.weight), expected_mean=0.0, expected_std=0.02)
            examined_params.add(f"{name}.weight")

            # biases are always zero initialized
            if hasattr(module, "bias") and module.bias is not None:
                _check_params_constant(cast(torch.Tensor, module.bias), expected_value=0.0)
                examined_params.add(f"{name}.bias")

        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, RMSNorm)):
            # normalization weights are initialized to 1.0
            _check_params_constant(cast(torch.Tensor, module.weight), expected_value=1.0)
            examined_params.add(f"{name}.weight")

            # biases are always zero initialized
            if hasattr(module, "bias") and module.bias is not None:
                _check_params_constant(cast(torch.Tensor, module.bias), expected_value=0.0)
                examined_params.add(f"{name}.bias")

    if missing := {name for name, _ in model.named_parameters()} - examined_params:
        raise RuntimeError(f"{missing} is not initialized")


def test_model_default_init():
    config = _get_model_config()
    model = config.build()
    model.init_weights()
    _examine_weights(model)


class TestDistributedModelInit(DeterministicDDPTestCase):
    def test_model_default_init(self):
        self.create_pg("cuda")

        config = _get_model_config()
        with torch.device("meta"):
            model = config.build()

        fsdp_config = FSDPConfig()
        model.fully_shard(fsdp_config=fsdp_config)
        model.init_weights()
        _examine_weights(model)
