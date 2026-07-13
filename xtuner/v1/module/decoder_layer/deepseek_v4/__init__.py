# Copyright (c) OpenMMLab. All rights reserved.
"""DeepSeek-V4 decoder layer — the V4 block plus its Hyper-Connections (HC)
helpers.

Coheres the V4 decoder layer with the HC pre/post stream-mix and the HC sinkhorn
split it depends on, which were previously scattered as ``decoder_layer`` siblings
(``hc_block`` / ``hc_sinkhorn``). Public symbols are re-exported here so call sites
import from ``xtuner.v1.module.decoder_layer.deepseek_v4``.
"""

from .decoder_layer import V4DecoderLayer, V4FFNState
from .hc_block import HCWrapperConfig, hc_post, hc_pre
from .hc_sinkhorn import hc_split_sinkhorn


__all__ = [
    "V4DecoderLayer",
    "V4FFNState",
    "HCWrapperConfig",
    "hc_post",
    "hc_pre",
    "hc_split_sinkhorn",
]
