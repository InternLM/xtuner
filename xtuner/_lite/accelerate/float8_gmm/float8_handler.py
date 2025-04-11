# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn as nn

from xtuner._lite import get_logger

logger = get_logger()


def _is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


class Float8Handler:
    def __init__(
        self,
        enable_fsdp_float8_all_gather=True,
        precompute_float8_dynamic_scale_for_fsdp=True,
        scaling_type_input="dynamic",
        scaling_type_weight="dynamic",
        scaling_type_grad_output="dynamic",
        scaling_granularity_gemm="tensorwise",
        scaling_granularity_grouped_gemm="channelwise",
        compile=True,
        pad_inner_dim=False,
    ):
        self.enabled = False

        if not _is_sm89_or_later():
            logger.warning(
                "Failed to enable float8 training because float8 is only supported on SM89 or later",
            )
            return

        from xtuner._lite.accelerate.float8_gmm import (
            CastConfig,
            Float8LinearConfig,
            ScalingType,
        )
        from xtuner._lite.accelerate.float8_gmm.config import ScalingGranularity

        # Mutates the model inplace replacing instances of torch.nn.Linear with Float8Linear
        enable_fsdp_float8_all_gather = enable_fsdp_float8_all_gather

        scaling_type_input = ScalingType(scaling_type_input)
        scaling_type_weight = ScalingType(scaling_type_weight)
        scaling_type_grad_output = ScalingType(scaling_type_grad_output)
        scaling_granularity_gemm = ScalingGranularity(scaling_granularity_gemm)
        self.config = Float8LinearConfig(
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            cast_config_input=CastConfig(
                scaling_type=scaling_type_input,
                scaling_granularity=scaling_granularity_gemm,
            ),
            cast_config_weight=CastConfig(
                scaling_type=scaling_type_weight,
                scaling_granularity=scaling_granularity_gemm,
            ),
            cast_config_grad_output=CastConfig(
                scaling_type=scaling_type_grad_output,
                scaling_granularity=scaling_granularity_gemm,
            ),
            enable_pre_and_post_forward=False,
            pad_inner_dim=pad_inner_dim,
        )
        self.scaling_granularity_grouped_gemm = scaling_granularity_grouped_gemm

        self.enabled = True

        # for precompute_float8_dynamic_scale_for_fsdp
        self.precompute_scale = (
            enable_fsdp_float8_all_gather and precompute_float8_dynamic_scale_for_fsdp
        )

        # for sync_float8_amax_and_scale_history
        self.delayed_scaling = (
            scaling_type_input == "delayed"
            or scaling_type_weight == "delayed"
            or scaling_type_grad_output == "delayed"
        )
        self._sync_float8_amax_and_scale_history = None
        self.compile = compile

    def convert_to_float8_training(
        self, model: nn.Module, amax_need_reduce: bool = False
    ):
        """This function converts the linear layers of `model` to
        `Float8Linear`.

        Note that today, only dynamic tensor scaling (the default) is supported. This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from xtuner._lite.accelerate.float8_gmm import (
            ChannelWiseFloat8GroupedLinear,
            TileWiseFloat8GroupedLinear,
        )

        def traverse(module):
            for name, child in module.named_children():
                if type(child).__name__ == "GroupedLinear":
                    if self.scaling_granularity_grouped_gemm == "channelwise":
                        child = ChannelWiseFloat8GroupedLinear.from_float(
                            child, amax_need_reduce
                        )
                    elif self.scaling_granularity_grouped_gemm == "tilewise":
                        child = TileWiseFloat8GroupedLinear.from_float(
                            child, amax_need_reduce
                        )
                    else:
                        raise NotImplementedError
                    module.add_module(name, child)
                else:
                    traverse(child)

        traverse(model)

        from xtuner._lite.accelerate.float8_gmm import convert_to_float8_training

        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=lambda mod, fqn: (fqn != "lm_head" and fqn[-4:] != "gate"),
        )

        logger.info("FP8 training enabled.")

    def precompute_float8_dynamic_scale_for_fsdp(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        from xtuner._lite.accelerate.float8_gmm import (
            precompute_float8_dynamic_scale_for_fsdp,
        )

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)

    def sync_float8_amax_and_scale_history(
        self, model: Union[nn.Module, List[nn.Module]]
    ):
        if not self.enabled:
            return

        if not self.delayed_scaling:
            return

        from xtuner._lite.accelerate.float8_gmm import (
            sync_float8_amax_and_scale_history,
        )

        # TODO(vkuzo): see if precalculating the modules to sync over is going to
        # meaningfully help performance

        if self._sync_float8_amax_and_scale_history is None:
            if self.compile:
                self._sync_float8_amax_and_scale_history = torch.compile(
                    sync_float8_amax_and_scale_history
                )
            else:
                self._sync_float8_amax_and_scale_history = (
                    sync_float8_amax_and_scale_history
                )

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            self._sync_float8_amax_and_scale_history(m)
