# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import STRATEGIES as MMENGINE_STRATEGIES
from mmengine.registry import Registry

__all__ = ['BUILDER', 'MAP_FUNC', 'STRATEGIES']

BUILDER = Registry('builder')
MAP_FUNC = Registry('map_fn')
STRATEGIES = Registry('strategy', parent=MMENGINE_STRATEGIES)
