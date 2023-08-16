# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import digit_version

from .entry_point import cli
from .version import __version__, version_info

__all__ = ['__version__', 'version_info', 'digit_version', 'cli']
