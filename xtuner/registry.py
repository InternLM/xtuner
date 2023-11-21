# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

__all__ = ['BUILDER', 'MAP_FUNC']

BUILDER = Registry('builder')
MAP_FUNC = Registry('map_fn')
