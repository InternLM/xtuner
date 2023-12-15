# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmengine.utils import digit_version

from .entry_point import cli
from .version import __version__, version_info

HF_PETREL_HUB = os.getenv('HF_PETREL_HUB', '')
HF_PETREL_ON = os.getenv('HF_PETREL_ON', 0) or HF_PETREL_HUB != ''
DEEPSPEED_PETREL_ON = os.getenv('DEEPSPEED_PETREL_ON', 0)
if HF_PETREL_ON:
    from .utils.fileio import (patch_hf_auto_from_pretrained,
                               patch_hf_save_pretrained)
    patch_hf_auto_from_pretrained(HF_PETREL_HUB)
    patch_hf_save_pretrained()

__all__ = [
    '__version__', 'version_info', 'digit_version', 'cli', 'HF_PETREL_ON',
    'DEEPSPEED_PETREL_ON'
]
