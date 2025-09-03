from typing import TYPE_CHECKING, Annotated, Generic, Literal, Optional, TypeVar

import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, computed_field
from typing_extensions import NotRequired, TypedDict

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
    """Base attention configuration for transformer attention mechanisms.

    This class defines the fundamental parameters for attention configurations
    in transformer models. It serves as a base class for specific attention
    implementations and provides common attention-related parameters.

    Attributes:
        num_attention_heads (int): Number of attention heads in the multi-head
            attention mechanism.
        head_dim (int): Dimension of each attention head.
        dropout (bool): Whether to apply dropout to attention weights.
            Defaults to False.
        qkv_bias (bool): Whether to use bias in the query, key, and value
            projection layers. Defaults to False.
        o_bias (bool): Whether to use bias in the output projection layer.
            Defaults to False.
        sliding_window (int | None): Size of the sliding window for local
            attention. Use -1 to disable sliding window attention. Defaults to -1.
    """

    model_config = ConfigDict(title="Base attention config for xtuner", extra="allow")
    num_attention_heads: Annotated[int, Parameter(group="attention")]
    head_dim: Annotated[int, Parameter(group="attention")]
    dropout: Annotated[bool, Parameter(group="attention")] = False
    # casual: bool = True
    qkv_bias: Annotated[bool, Parameter(group="attention")] = False
    o_bias: Annotated[bool, Parameter(group="attention")] = False
    sliding_window: Annotated[int | None, Parameter(group="attention")] = -1

    def build(
        self,
        hidden_size: int,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Optional["Float8Config"] = None,
    ) -> T:
        """Build the attention module.

        Args:
            hidden_size (int): Hidden size of the transformer model.
            layer_type (Literal["full_attention", "sliding_attention"] | None): Type of
                attention layer to build. If None, uses default behavior.
            layer_idx (int): Index of the current layer. Defaults to 0.
            generate_config (GenerateConfig | None): Configuration for generation.
                Defaults to None.
            float8_cfg (Float8Config | None): Float8 quantization configuration.
                Defaults to None.

        Returns:
            T: The subclass attention module.
        """
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
    """Transformer model configuration for XTuner.

    XTuner follows the principle that all modules are constructed from the
    top-level model config, which inevitably leads to lower-level modules
    depending on upper-level interfaces.

    If the model config were declared in the model module, it would cause sub-modules like attention to import the
    model, resulting in circular import issues. For this reason, we choose to declare the base class for ModelConfig in
    the base config.

    This class defines the fundamental configuration parameters for transformer-based
    language models, including vocabulary size, model dimensions, attention
    configurations, and various model-specific settings.

    Attributes:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        max_position_embeddings (int): Maximum sequence length that the model
            can handle.
        pad_token_id (int): ID of the padding token in the vocabulary.
        num_hidden_layers (int): Number of transformer blocks/layers in the model.
        hidden_size (int): Dimension of the hidden states throughout the model.
        intermediate_size (int): Dimension of the feed-forward network's
            intermediate layer.
        rms_norm_eps (float): Epsilon value for RMS normalization layers.
        rope_theta (float): Base frequency for RoPE (Rotary Position Embedding).
        hidden_act (str): Activation function used in the feed-forward network.
        attention (BaseAttnConfig): Configuration for the attention mechanism.
        mlp_bias (bool): Whether to use bias in the MLP layers. Defaults to False.
        tie_word_embeddings (bool): Whether to tie input and output word embeddings.
            Defaults to False.
        generate_config (GenerateConfig | None): Configuration for text generation.
            Defaults to None.
        float8_cfg (Float8Config | None): Float8 quantization configuration.
            Defaults to None.
        return_hidden_states (bool): Whether to return all hidden states from
            all layers. Defaults to False.
        use_sliding_window (bool): Whether to use sliding window attention.
            Defaults to False.
        max_window_layers (int | None): Number of layers to use sliding window
            attention for. If None, uses sliding window for all layers when
            `use_sliding_window` is True. Defaults to None.
    """

    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    vocab_size: Annotated[int, Parameter(group="model")]
    max_position_embeddings: Annotated[int, Parameter(group="model")]
    pad_token_id: Annotated[int, Parameter(group="model")]
    num_hidden_layers: Annotated[int, Parameter(group="model")]
    hidden_size: Annotated[int, Parameter(group="model")]
    intermediate_size: Annotated[int, Parameter(group="model")]
    rms_norm_eps: Annotated[float, Parameter(group="model")]
    rope_theta: Annotated[float, Parameter(group="model")]  # required by transformers's build rope
    hidden_act: Annotated[str, Parameter(group="model")]  # key defined in `transformers.activations.ACT2CLS`
    attention: BaseAttnConfig
    mlp_bias: Annotated[bool, Parameter(group="model")] = False
    tie_word_embeddings: Annotated[bool, Parameter(group="model")] = False
    model_type: Annotated[str | None, Parameter(group="model")] = None  # TODO: yehaochen maybe should be removed
    generate_config: GenerateConfig | None = None
    float8_cfg: Optional["Float8Config"] = None
    return_hidden_states: Annotated[bool, Parameter(group="model")] = False
    use_sliding_window: Annotated[bool, Parameter(group="model")] = False
    max_window_layers: Annotated[int | None, Parameter(group="model")] = None

    @computed_field
    def num_attention_heads(self) -> int:
        return self.attention.num_attention_heads

    @computed_field
    def head_dim(self) -> int:
        return self.attention.head_dim

    @computed_field
    def layers_type(self) -> list[Literal["full_attention", "sliding_attention"]]:
        if not self.use_sliding_window:
            return ["full_attention"] * self.num_hidden_layers
        else:
            if self.max_window_layers is None:
                return ["sliding_attention"] * self.num_hidden_layers
            return [
                "sliding_attention" if i >= self.max_window_layers else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

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
    return_router_results: bool = False

    def build(self) -> "MoE":
        from xtuner.v1.model.moe.moe import MoE

        return MoE(self)


class ModelOutputs(TypedDict):
    hidden_states: NotRequired[list[torch.Tensor]]
    logits: NotRequired[torch.Tensor]
    loss: torch.Tensor


class MoEModelOutputs(ModelOutputs):
    router_logits: NotRequired[dict[str, torch.Tensor]]
    balancing_loss: NotRequired[torch.Tensor]
    z_loss: NotRequired[torch.Tensor]
    tokens_per_expert_global: NotRequired[torch.Tensor]
