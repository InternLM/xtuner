from typing import TYPE_CHECKING, Annotated, Generic, Literal, Optional, TypedDict, TypeVar

import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, computed_field
from typing_extensions import NotRequired

from xtuner.v1.config.float8 import Float8Config
from xtuner.v1.config.loss import BalancingLossConfig, ZLossConfig


if TYPE_CHECKING:
    from xtuner.v1.model.base import BaseModel as _BaseModel
    from xtuner.v1.model.moe.moe import MoE


T = TypeVar("T")


class GenerateConfig(BaseModel):
    max_batch_size: Annotated[int, Parameter(group="generate")] = 32
    max_prefill_batch: Annotated[int, Parameter(group="generate")] = 16
    max_length: Annotated[int, Parameter(group="generate")] = 2048
    block_size: Annotated[int, Parameter(group="generate")] = 128
    dtype: Annotated[Literal["bf16", "fp8"], Parameter(group="generate")] = "bf16"


class BaseAttnConfig(BaseModel, Generic[T]):
    model_config = ConfigDict(title="Base attention config for xtuner", extra="allow")
    num_attention_heads: Annotated[int, Parameter(group="attention")]
    head_dim: Annotated[int, Parameter(group="attention")]
    dropout: Annotated[bool, Parameter(group="attention")] = False
    # casual: bool = True
    qkv_bias: Annotated[bool, Parameter(group="attention")] = False
    o_bias: Annotated[bool, Parameter(group="attention")] = False

    def build(
        self,
        hidden_size: int,
        layer_idx: int = 0,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Optional["Float8Config"] = None,
    ) -> T:
        raise NotImplementedError


class BaseRouterConfig(BaseModel, Generic[T]):
    scoring_func: Annotated[Literal["sigmoid", "softmax"], Parameter(group="router")]
    router_scaling_factor: Annotated[float, Parameter(group="router")]
    norm_topk_prob: Annotated[bool, Parameter(group="router")]

    def build(
        self,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ) -> T:
        """Build the router module."""
        raise NotImplementedError


class TransformerConfig(BaseModel):
    """XTuner follows the principle that all modules are constructed from the
    top-level model config, which inevitably leads to lower-level modules
    depending on upper-level interfaces.

    If the model config were declared in the model module, it would cause sub-modules like attention to import the
    model, resulting in circular import issues. For this reason, we choose to declare the base class for ModelConfig in
    the base config.
    """

    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    vocab_size: Annotated[int, Parameter(group="model")]
    max_position_embeddings: Annotated[int, Parameter(group="model")]
    padding_idx: Annotated[int, Parameter(group="model")]
    num_hidden_layers: Annotated[int, Parameter(group="model")]
    hidden_size: Annotated[int, Parameter(group="model")]
    intermediate_size: Annotated[int, Parameter(group="model")]
    rms_norm_eps: Annotated[float, Parameter(group="model")]
    rope_theta: Annotated[float, Parameter(group="model")]  # required by transformers's build rope
    hidden_act: Annotated[str, Parameter(group="model")]  # key defined in `transformers.activations.ACT2CLS`
    attention: BaseAttnConfig
    mlp_bias: Annotated[bool, Parameter(group="model")] = False
    tie_word_embeddings: Annotated[bool, Parameter(group="model")] = False
    model_type: Annotated[Literal["qwen"] | None, Parameter(group="model")] = None
    generate_config: GenerateConfig | None = None
    float8_cfg: Optional["Float8Config"] = None
    return_hidden_states: Annotated[bool, Parameter(group="model")] = False

    @computed_field
    def num_attention_heads(self) -> int:
        return self.attention.num_attention_heads

    @computed_field
    def head_dim(self) -> int:
        return self.attention.head_dim

    def build(self) -> "_BaseModel":
        raise NotImplementedError


class MoEConfig(TransformerConfig):
    n_routed_experts: Annotated[int, Parameter(group="moe")]
    n_shared_experts: Annotated[int, Parameter(group="moe")]
    num_experts_per_tok: Annotated[int, Parameter(group="moe")]
    first_k_dense_replace: Annotated[int, Parameter(group="moe")] = 0
    hidden_factor: Annotated[float, Parameter(group="moe")] = 1.0
    moe_intermediate_size: Annotated[int, Parameter(group="moe")]
    ep_size: Annotated[int, Parameter(group="moe")] = 1
    dispatcher: Annotated[Literal["deepep", "all2all"] | None, Parameter(group="moe")] = None
    router: BaseRouterConfig
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
    z_loss_cfg: Optional["ZLossConfig"] = None
    return_router_results: bool = True

    def build(self) -> "MoE":
        from xtuner.v1.model.moe.moe import MoE

        return MoE(self)


class ModelOutputs(TypedDict):
    hidden_states: NotRequired[list[torch.Tensor]]
    logits: NotRequired[torch.Tensor]
    loss: torch.Tensor


class MoEModelOutputs(ModelOutputs):
    router_logits: NotRequired[torch.Tensor]
    balancing_loss: NotRequired[torch.Tensor]
    z_loss: NotRequired[torch.Tensor]
    tokens_per_expert_global: NotRequired[torch.Tensor]
