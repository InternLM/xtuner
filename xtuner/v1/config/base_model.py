from typing import Literal, TypedDict

import torch
from pydantic import BaseModel, ConfigDict, computed_field
from typing_extensions import NotRequired


class BaseAttnConfig(BaseModel):
    model_config = ConfigDict(title="Base attention config for xtuner", extra="allow")
    num_attention_heads: int
    head_dim: int
    dropout: bool = False
    casual: bool = True
    qkv_bias: bool = False
    o_bias: bool = False


class BaseRouterConfig(BaseModel):
    scoring_func: Literal["sigmoid", "softmax"]
    router_scaling_factor: float
    norm_topk_prob: bool


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
    vocab_size: int
    max_position_embeddings: int
    padding_idx: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float  # required by transformers's build rope
    hidden_act: str  # key defined in `transformers.activations.ACT2CLS`
    attention: BaseAttnConfig
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    training_dtype: Literal["bf16", "fp8"] = "bf16"
    chunked_loss: bool = False
    model_type: Literal["qwen"] | None = None

    @computed_field
    def num_attention_heads(self) -> int:
        return self.attention.num_attention_heads

    @computed_field
    def head_dim(self) -> int:
        return self.attention.head_dim


class MoEConfig(TransformerConfig):
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int = 0
    hidden_factor: float = 1.0
    moe_intermediate_size: int
    dispatcher: Literal["deepep", "naive", "all2all"] = "deepep"
    router: BaseRouterConfig


class ModelOutputs(TypedDict):
    hidden_states: NotRequired[list[torch.Tensor]]
    logits: NotRequired[torch.Tensor]
    loss: torch.Tensor


class MoEModelOutputs(ModelOutputs):
    router_logits: NotRequired[torch.Tensor]
