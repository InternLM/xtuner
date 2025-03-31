# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/__init__.py
import re

import torch

from xtuner._lite.accelerate.float8_gmm.config import (
    CastConfig,
    DelayedScalingConfig,
    Float8GemmConfig,
    Float8LinearConfig,
    ScalingType,
)
from xtuner._lite.accelerate.float8_gmm.float8_gmm import Float8GroupedLinearACWE
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


def parse_version(version_string):
    # Extract just the X.Y.Z part from the version string
    match = re.match(r"(\d+\.\d+\.\d+)", version_string)
    if match:
        version = match.group(1)
        return [int(x) for x in version.split(".")]
    else:
        raise ValueError(f"Invalid version string format: {version_string}")


def compare_versions(v1, v2):
    v1_parts = parse_version(v1)
    v2_parts = parse_version(v2)
    return (v1_parts > v2_parts) - (v1_parts < v2_parts)


def is_fbcode():
    return not hasattr(torch.version, "git_version")


def torch_version_at_least(min_version):
    return is_fbcode() or compare_versions(torch.__version__, min_version) >= 0


if torch_version_at_least("2.5.0"):
    # Needed to load Float8Tensor with weights_only = True
    from torch.serialization import add_safe_globals

    add_safe_globals(
        [
            Float8Tensor,
            ScaledMMConfig,
            GemmInputRole,
            LinearMMConfig,
            # WeightWithDelayedFloat8CastTensor,
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
    "Float8GroupedLinearACWE",
    "Float8Handler",
]
