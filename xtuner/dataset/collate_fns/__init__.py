# Copyright (c) OpenMMLab. All rights reserved.
from .default_collate_fn import default_collate_fn
from .mmlu_collate_fn import mmlu_collate_fn

__all__ = ['default_collate_fn', 'mmlu_collate_fn']
