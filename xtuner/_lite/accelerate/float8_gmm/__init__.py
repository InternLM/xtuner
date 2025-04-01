# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/__init__.py
import torch
from mmengine.utils import digit_version

from xtuner._lite.accelerate.float8_gmm.config import (
    CastConfig,
    DelayedScalingConfig,
    Float8GemmConfig,
    Float8LinearConfig,
    ScalingType,
)
from xtuner._lite.accelerate.float8_gmm.float8_gmm_channel_wise import (
    ChannelWiseFloat8GroupedLinear,
)
from xtuner._lite.accelerate.float8_gmm.float8_gmm_tile_wise import (
    TileWiseFloat8GroupedLinear,
)
from xtuner._lite.accelerate.float8_gmm.float8_handler import Float8Handler
from xtuner._lite.accelerate.float8_gmm.float8_linear import Float8Linear
from xtuner._lite.accelerate.float8_gmm.float8_linear_utils import (
    convert_to_float8_training,
    linear_requires_sync,
    sync_float8_amax_and_scale_history,
)
from xtuner._lite.accelerate.float8_gmm.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
)
from xtuner._lite.accelerate.float8_gmm.fsdp_utils import (
    precompute_float8_dynamic_scale_for_fsdp,
)

if digit_version(torch.__version__) >= digit_version("2.5.0"):
    # Needed to load Float8Tensor with weights_only = True
    from torch.serialization import add_safe_globals

    add_safe_globals(
        [
            Float8Tensor,
            ScaledMMConfig,
            GemmInputRole,
            LinearMMConfig,
        ]
    )

__all__ = [
    # configuration
    "DelayedScalingConfig",
    "ScalingType",
    "Float8GemmConfig",
    "Float8LinearConfig",
    "CastConfig",
    # top level UX
    "convert_to_float8_training",
    "linear_requires_sync",
    "sync_float8_amax_and_scale_history",
    "precompute_float8_dynamic_scale_for_fsdp",
    # note: Float8Tensor and Float8Linear are not public APIs
    "Float8Linear",
    "ChannelWiseFloat8GroupedLinear",
    "TileWiseFloat8GroupedLinear",
    "Float8Handler",
]
