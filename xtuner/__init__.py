# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmengine.utils import digit_version

from .entry_point import cli
from .version import __version__, version_info

HF_CEPH_HUB = os.getenv('HF_CEPH_HUB', '')
HF_USE_CEPH = os.getenv('HF_USE_CEPH', 0) or HF_CEPH_HUB != ''
DS_CEPH_DIR = os.getenv('DS_CEPH_DIR', None)
if HF_USE_CEPH:
    from .utils.fileio import (patch_hf_auto_from_pretrained,
                               patch_hf_save_pretrained)
    patch_hf_auto_from_pretrained(HF_CEPH_HUB)
    patch_hf_save_pretrained()

if DS_CEPH_DIR:
    from .utils.fileio import patch_deepspeed_engine
    patch_deepspeed_engine()

__all__ = [
    '__version__', 'version_info', 'digit_version', 'cli', 'HF_USE_CEPH',
    'DS_CEPH_DIR'
]
