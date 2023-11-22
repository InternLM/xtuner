# Copyright (c) OpenMMLab. All rights reserved.
from .defalut_collate_fn import default_collate_fn
from .intern_repo_collate_fn import intern_repo_collate_fn
from .mmlu_collate_fn import mmlu_collate_fn

__all__ = ['default_collate_fn', 'mmlu_collate_fn', 'intern_repo_collate_fn']
