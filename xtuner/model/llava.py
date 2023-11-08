# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from peft import get_peft_model, prepare_model_for_kbit_training

from ..registry import BUILDER
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)


class LLaVAModel(BaseModel):

    def __init__(self,
                 llm,
                 visual_encoder,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 peft_model=None,
                 use_activation_checkpointing=True):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        projector_config = ProjectorConfig(
            visual_hidden_size=self.visual_encoder.config.hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth)
        self.projector = ProjectorModel(projector_config).to(
            self.visual_encoder.dtype)

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                self.visual_encoder.enable_input_require_grads()
            else:
                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        if isinstance(llm_lora, dict) or isinstance(
                llm_lora, Config) or isinstance(llm_lora, ConfigDict):
            self.llm_lora = BUILDER.build(llm_lora)
        else:
            self.llm_lora = llm_lora
        self.peft_model = peft_model
        self.use_lora = llm_lora is not None
        if self.use_lora:
            self._prepare_for_lora(peft_model, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.visual_encoder.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def _prepare_for_lora(self,
                          peft_model=None,
                          use_activation_checkpointing=True):
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if self.llm_lora.target_modules is None:
            modules = find_all_linear_names(self.llm)
            self.llm_lora.target_modules = modules

        self.llm = get_peft_model(self.llm, self.llm_lora)
        if peft_model is not None:
            _ = load_checkpoint(self, peft_model)

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # Step 1. With VisualEncoder
        to_return = OrderedDict()
        if not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. With LLM
        if self.use_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        return to_return

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):
        if 'pixel_values' in data:
            visual_outputs = self.visual_encoder(
                data['pixel_values'], output_hidden_states=True)
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
            data['pixel_values'] = pixel_values
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
