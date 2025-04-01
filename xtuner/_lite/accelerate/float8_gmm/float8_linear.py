# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/float8_linear.py
# 1. Support num_gpus > out_features. Related issue: https://github.com/pytorch/ao/issues/1938
# 2. Support linear's weight is still DTensor after fsdp all_gather (EP related)

from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint

# modified
from torch.distributed._tensor import DTensor

from xtuner._lite.accelerate.float8_gmm.config import (
    Float8LinearConfig,
    ScalingGranularity,
    ScalingType,
)
from xtuner._lite.accelerate.float8_gmm.distributed_utils import (
    tensor_already_casted_to_fp8,
)
from xtuner._lite.accelerate.float8_gmm.float8_scaling_utils import (
    NoopFwToFloat8BwDynamic,
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_dynamic,
)
from xtuner._lite.accelerate.float8_gmm.float8_tensor import (
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
    hp_tensor_and_scale_to_float8,
)
from xtuner._lite.accelerate.float8_gmm.float8_utils import tensor_to_scale
from xtuner._lite.accelerate.float8_gmm.fsdp_utils import (
    WeightWithDynamicFloat8CastTensor,
)


@torch._dynamo.allow_in_graph
class manual_float8_matmul_with_args_in_float8(torch.autograd.Function):
    """Like torch.matmul, but with the arguments in float8.

    Note: this function requires all arguments to already be Float8Tensor objects,
    which only supports tensorwise scaling granularity. The reason we didn't just make this
    function support axiswise scaling granularity is because that would need very
    careful testing of delayed scaling, as delayed scaling modifies buffers inplace.

    In the future we'll probably have to unify, just postponing that until a future PR.
    """

    @staticmethod
    def forward(
        ctx,
        input_fp8,
        weight_fp8_t,
    ):
        ctx.save_for_backward(input_fp8, weight_fp8_t)
        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        orig_shape = input_fp8.shape
        input_fp8_reshaped = input_fp8.reshape(-1, orig_shape[-1])
        res_bits = torch.mm(input_fp8_reshaped, weight_fp8_t)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, grad_output_fp8):
        input_fp8, weight_fp8_t = ctx.saved_tensors

        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        grad_output_fp8_orig_shape = grad_output_fp8.shape
        grad_output_fp8_reshaped = grad_output_fp8.reshape(
            -1, grad_output_fp8_orig_shape[-1]
        )

        # calculate grad_input
        grad_input = torch.mm(
            grad_output_fp8_reshaped,
            weight_fp8_t.t(),
        )
        grad_input = grad_input.reshape(
            *grad_output_fp8_orig_shape[:-1], grad_input.shape[-1]
        )

        input_fp8_orig_shape = input_fp8.shape
        input_fp8_reshaped = input_fp8.reshape(-1, input_fp8_orig_shape[-1])

        # calculate grad_weight
        # Note: the variant below is slightly faster on LLaMa 3 8B pretraining
        # compared to than calculating `grad_weight_t = input_fp8_t @ grad_output_fp8_reshaped`
        grad_weight = torch.mm(
            grad_output_fp8_reshaped.t(),
            input_fp8_reshaped,
        )

        return grad_input, grad_weight.t()


@torch._dynamo.allow_in_graph
class manual_float8_matmul_with_args_in_hp(torch.autograd.Function):
    """Like torch.matmul, but with the arguments in high precision and the cast
    to float8 defined inside of this function.

    Note: this function currently only supports dynamic scaling type and
    axiswise granularity. We will have to unify this with other scaling types
    and other granularities in a separate PR.
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp_t: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        config: Float8LinearConfig,
    ):
        ctx.save_for_backward(input_hp, weight_hp_t)
        ctx.linear_mm_config = linear_mm_config
        ctx.config = config

        c = config

        if c.cast_config_input.scaling_type is ScalingType.DISABLED:
            input_maybe_fp8 = input_hp
        else:
            input_maybe_fp8 = hp_tensor_to_float8_dynamic(
                input_hp,
                c.cast_config_input.target_dtype,
                linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    -1, c.cast_config_input.scaling_granularity
                ),
            )

        if c.cast_config_weight.scaling_type is ScalingType.DISABLED:
            weight_maybe_fp8_t = weight_hp_t
        else:
            weight_maybe_fp8_t = hp_tensor_to_float8_dynamic(
                weight_hp_t,
                c.cast_config_weight.target_dtype,
                linear_mm_config,
                gemm_input_role=GemmInputRole.WEIGHT,
                scaling_granularity=c.cast_config_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_weight.scaling_granularity
                ),
            )

        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        orig_shape = input_maybe_fp8.shape
        input_maybe_fp8_reshaped = input_maybe_fp8.reshape(-1, orig_shape[-1])
        res_bits = torch.mm(input_maybe_fp8_reshaped, weight_maybe_fp8_t)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, grad_output):
        input_hp, weight_hp_t = ctx.saved_tensors
        c = ctx.config

        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        grad_output_orig_shape = grad_output.shape
        grad_output_reshaped = grad_output.reshape(-1, grad_output_orig_shape[-1])

        #
        # calculate grad_input
        #

        if c.cast_config_grad_output.scaling_type is ScalingType.DISABLED:
            grad_output_reshaped_maybe_fp8_dim0 = grad_output_reshaped
        else:
            grad_output_reshaped_maybe_fp8_dim0 = hp_tensor_to_float8_dynamic(
                grad_output_reshaped,
                c.cast_config_grad_output.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    -1, c.cast_config_grad_output.scaling_granularity
                ),
            )

        if c.cast_config_weight_for_grad_input.scaling_type is ScalingType.DISABLED:
            weight_t_maybe_fp8_dim0 = weight_hp_t
        else:
            # Note: we need https://github.com/pytorch/pytorch/issues/136267
            # to be solved to have a chance to reuse max(abs(weight, dim=...))
            # from the forward to get max(abs(weight)) here without reading
            # the entire tensor.
            weight_t_maybe_fp8_dim0 = hp_tensor_to_float8_dynamic(
                weight_hp_t,
                c.cast_config_weight_for_grad_input.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.WEIGHT,
                scaling_granularity=c.cast_config_weight_for_grad_input.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    -1, c.cast_config_weight_for_grad_input.scaling_granularity
                ),
            )

        grad_input = torch.mm(
            grad_output_reshaped_maybe_fp8_dim0,
            weight_t_maybe_fp8_dim0.t(),
        )
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        input_hp_orig_shape = input_hp.shape
        input_hp_reshaped = input_hp.reshape(-1, input_hp_orig_shape[-1])

        #
        # calculate grad_weight
        #

        if (
            c.cast_config_grad_output_for_grad_weight.scaling_type
            is ScalingType.DISABLED
        ):
            grad_output_reshaped_maybe_fp8_dim1 = grad_output_reshaped
        else:
            grad_output_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                grad_output_reshaped,
                c.cast_config_grad_output_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_grad_output_for_grad_weight.scaling_granularity
                ),
            )

        if c.cast_config_input_for_grad_weight.scaling_type is ScalingType.DISABLED:
            input_reshaped_maybe_fp8_dim1 = input_hp_reshaped
        else:
            input_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                input_hp_reshaped,
                c.cast_config_input_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_input_for_grad_weight.scaling_granularity
                ),
            )

        grad_weight = torch.mm(
            grad_output_reshaped_maybe_fp8_dim1.t(),
            input_reshaped_maybe_fp8_dim1,
        )

        empty_grads = None, None

        return grad_input, grad_weight.t(), *empty_grads


class Float8Linear(torch.nn.Linear):
    """
    Note: this is **not** a public API and is only intended to be used
    inside of this repository. Please file an issue if you would benefit
    from this being a public API.

    A wrapper around a `torch.nn.Linear` module which does fp8 compute, and tracks
    scales in way friendly to delayed scaling.
    """

    def __init__(self, *args, **kwargs):
        """Additional arguments on top of `torch.nn.Linear`'s arguments:

        * `config`: Float8LinearConfig
        """

        config = kwargs.pop("config")
        super().__init__(*args, **kwargs)

        # Defines the scaling behavior of input, weight, grad_output
        self.scaling_type_input = config.cast_config_input.scaling_type
        self.scaling_type_weight = config.cast_config_weight.scaling_type
        self.scaling_type_grad_output = config.cast_config_grad_output.scaling_type
        self.config = config

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

        self.weight = torch.nn.Parameter(
            self.weight.view(-1)
        )  # hardcode for fp8 linear when fsdp_size > self.out_features
        self.register_load_state_dict_hook()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)

        weight_key = prefix + "weight"
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            state_dict[weight_key] = weight.reshape(self.out_features, self.in_features)

        return state_dict

    def register_load_state_dict_hook(self):
        def hook(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            weight_key = prefix + "weight"
            if weight_key in state_dict:
                weight = state_dict[weight_key]

                if weight.shape == (self.out_features, self.in_features):
                    state_dict[weight_key] = weight.view(-1)  # 展平为 1D

        self._register_load_state_dict_pre_hook(hook)

    def cast_input_to_float8(self, input: torch.Tensor) -> torch.Tensor:
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)

        assert self.scaling_type_input is ScalingType.DYNAMIC
        input_fp8 = hp_tensor_to_float8_dynamic(
            input,
            self.config.cast_config_input.target_dtype,
            self.linear_mm_config,
            gemm_input_role=GemmInputRole.INPUT,
        )
        return input_fp8

    def get_weight_scale(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if tensor_already_casted_to_fp8(weight):
            return None
        assert self.scaling_type_weight is ScalingType.DYNAMIC
        return tensor_to_scale(weight, self.config.cast_config_weight.target_dtype)

    def cast_weight_to_float8_t(
        self,
        weight: torch.Tensor,
        weight_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tensor_already_casted_to_fp8(weight):
            return weight.t()
        weight_fp8 = hp_tensor_and_scale_to_float8(
            weight,
            weight_scale,
            self.config.cast_config_weight.target_dtype,
            self.linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        )
        return weight_fp8.t()

    def cast_output_to_float8_in_bw(self, output: torch.Tensor) -> torch.Tensor:
        assert self.scaling_type_grad_output is ScalingType.DYNAMIC
        output = NoopFwToFloat8BwDynamic.apply(
            output,
            self.linear_mm_config,
            self.config.cast_config_grad_output.target_dtype,
        )
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        has_any_axiswise_scaling = any(
            cc.scaling_granularity is ScalingGranularity.AXISWISE
            for cc in [
                self.config.cast_config_input,
                self.config.cast_config_weight,
                self.config.cast_config_grad_output,
                self.config.cast_config_input_for_grad_weight,
                self.config.cast_config_weight_for_grad_input,
                self.config.cast_config_grad_output_for_grad_weight,
            ]
        )

        weight = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        weight = weight.view(self.out_features, self.in_features)

        if not has_any_axiswise_scaling:
            input_fp8 = self.cast_input_to_float8(input)
            # If force_recompute_fp8_weight_in_bwd, we only recompute the fp8 weight,
            # weight_scale should be saved.
            weight_scale = self.get_weight_scale(weight)

            if self.config.force_recompute_fp8_weight_in_bwd:
                weight_fp8_t = checkpoint.checkpoint(
                    self.cast_weight_to_float8_t,
                    weight,
                    weight_scale,
                )
            else:
                weight_fp8_t = self.cast_weight_to_float8_t(weight, weight_scale)

            output = manual_float8_matmul_with_args_in_float8.apply(
                input_fp8, weight_fp8_t
            )

            # Cast grad_output to float8_e5m2 during backward
            output = self.cast_output_to_float8_in_bw(output)

        else:
            # for now, axiswise path is separate
            # TODO(future PR): unify to support mix and match
            output = manual_float8_matmul_with_args_in_hp.apply(
                input,
                (weight).t(),
                self.linear_mm_config,
                self.config,
            )

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    def extra_repr(self):
        c = self.config
        ci = f"i:{c.cast_config_input.short_str()}"
        cw = f"w:{c.cast_config_weight.short_str()}"
        cgo = f"go:{c.cast_config_grad_output.short_str()}"
        parts = [ci, cw, cgo]
        if c.cast_config_input_for_grad_weight != c.cast_config_input:
            parts.append(f"i_gw:{c.cast_config_input_for_grad_weight.short_str()}")
        if c.cast_config_weight_for_grad_input != c.cast_config_weight:
            parts.append(f"w_gi:{c.cast_config_weight_for_grad_input.short_str()}")
        if c.cast_config_grad_output_for_grad_weight != c.cast_config_grad_output:
            parts.append(
                f"go_gw:{c.cast_config_grad_output_for_grad_weight.short_str()}"
            )
        cast_config_str = ",".join(parts)
        s = f'{super().extra_repr()}, cast_configs={cast_config_str}"'
        return s

    @classmethod
    def from_float(
        cls,
        mod,
        config: Optional[Float8LinearConfig] = None,
    ):
        """Create an nn.Linear with fp8 compute from a regular nn.Linear.

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8
        """
        if config is None:
            config = Float8LinearConfig()
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
                config=config,
            )
        new_mod.weight = torch.nn.Parameter(mod.weight.view(*new_mod.weight.shape))
        new_mod.bias = mod.bias

        # If FSDP float8 all-gather is on, wrap the weight in a float8-aware
        # tensor subclass. This must happen last because:
        # 1. weight needs to be on the correct device to create the buffers
        # 2. buffers need to be already created for the delayed scaling version
        #    of the weight wrapper to be initialized
        if config.enable_fsdp_float8_all_gather:
            assert config.cast_config_weight.scaling_type is ScalingType.DYNAMIC
            new_mod.weight = torch.nn.Parameter(
                WeightWithDynamicFloat8CastTensor(
                    new_mod.weight,
                    new_mod.linear_mm_config,
                    new_mod.config.cast_config_weight.target_dtype,
                )
            )

        return new_mod
