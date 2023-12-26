# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .configuration_projector import ProjectorConfig


class ProjectorModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = ProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        modules = [
            nn.Linear(
                config.visual_hidden_size,
                config.llm_hidden_size,
                bias=config.bias)
        ]
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(
                nn.Linear(
                    config.llm_hidden_size,
                    config.llm_hidden_size,
                    bias=config.bias))
        self.model = nn.Sequential(*modules)

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ProjectorModel):
            module.gradient_checkpointing = value

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs
