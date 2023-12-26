# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_projector import ProjectorConfig
from .modeling_projector import ProjectorModel

AutoConfig.register('projector', ProjectorConfig)
AutoModel.register(ProjectorConfig, ProjectorModel)

__all__ = ['ProjectorConfig', 'ProjectorModel']
