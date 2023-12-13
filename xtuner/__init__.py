# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import digit_version

from .entry_point import cli
from .utils.fileio import \
    patch_hf_auto_from_pretrained as _patch_hf_auto_from_pretrained
from .version import __version__, version_info

_patch_hf_auto_from_pretrained()

__all__ = ['__version__', 'version_info', 'digit_version', 'cli']
