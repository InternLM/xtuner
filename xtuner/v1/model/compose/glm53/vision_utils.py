# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash vision grid helpers, see doc/xtuner_glm5p3flash_design.md
F1.d/F2 §7.3."""

import torch


def flatten_video_grid_thw(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """Expand ``[num_videos, 3]`` (t, h, w) rows into ``[sum(t), 3]`` rows of
    ``[1, h, w]``.

    GLM-5.3-Flash's vision tower only accepts the expanded form (§7.3): mixing the two
    representations is silent (both have the same total patch count) but breaks downstream
    feature-grouping/`num_img_tokens` semantics, so callers must expand before the tower and
    never pass an unexpanded ``[t, h, w]`` row through.
    """
    t = video_grid_thw[:, 0]
    hw = video_grid_thw[:, 1:]
    flattened_hw = torch.repeat_interleave(hw, t, dim=0)
    ones = torch.ones(flattened_hw.shape[0], 1, dtype=video_grid_thw.dtype, device=video_grid_thw.device)
    return torch.cat([ones, flattened_hw], dim=1)
