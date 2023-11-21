# Copyright (c) OpenMMLab. All rights reserved.
from .defalut_collate_fn import default_collate_fn
from .internlm_collate_fn import internlm_collate_fn
from .mmlu_collate_fn import mmlu_collate_fn

__all__ = ['default_collate_fn', 'mmlu_collate_fn', 'internlm_collate_fn']
