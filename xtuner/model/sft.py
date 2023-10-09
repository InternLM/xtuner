# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from peft import PeftType, get_peft_model, prepare_model_for_kbit_training
from torch import nn

from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .utils import LoadWoInit, find_all_linear_names, traverse_dict


class SupervisedFinetune(BaseModel):

    def __init__(self,
                 llm,
                 lora=None,
                 peft_model=None,
                 use_activation_checkpointing=True):
        super().__init__()
        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)

            # enable gradient checkpointing for memory efficiency
            self.llm.gradient_checkpointing_enable()

        if isinstance(lora, dict) or isinstance(lora, Config) or isinstance(
                lora, ConfigDict):
            self.lora = BUILDER.build(lora)
        else:
            self.lora = lora
        self.peft_model = peft_model
        self.use_lora = lora is not None
        if self.use_lora:
            self._prepare_for_lora(peft_model, use_activation_checkpointing)

        self._is_init = True

    def _prepare_for_lora(self,
                          peft_model=None,
                          use_activation_checkpointing=True):
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if self.lora.target_modules is None:
            modules = find_all_linear_names(self.llm)
            self.lora.target_modules = modules

        self.llm = get_peft_model(self.llm, self.lora)
        if peft_model is not None:
            _ = load_checkpoint(self, peft_model)

    def init_weights(self):
        pass

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):

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

    def state_dict(self, destination=None, prefix='', keep_vars=False):

        def get_peft_model_state_dict(model,
                                      state_dict=None,
                                      adapter_name='default'):
            # Modified from `https://github.com/huggingface/peft/blob/main/src
            # /peft/utils/save_and_load.py`

            config = model.peft_config[adapter_name]
            if state_dict is None:
                state_dict = model.state_dict()
            if config.peft_type == PeftType.LORA:
                # adapted from `https://github.com/microsoft/LoRA/blob/main/
                # loralib/utils.py`
                # to be used directly with the state dict which is necessary
                # when using DeepSpeed or FSDP
                bias = config.bias
                if bias == 'none':
                    to_return = {
                        k: state_dict[k]
                        for k in state_dict if 'lora_' in k
                    }
                elif bias == 'all':
                    to_return = {
                        k: state_dict[k]
                        for k in state_dict if 'lora_' in k or 'bias' in k
                    }
                elif bias == 'lora_only':
                    to_return = {}
                    for k in state_dict:
                        if 'lora_' in k:
                            to_return[k] = state_dict[k]
                            bias_name = k.split('lora_')[0] + 'bias'
                            if bias_name in state_dict:
                                to_return[bias_name] = state_dict[bias_name]
                else:
                    raise NotImplementedError
                to_return = {
                    k: v
                    for k, v in to_return.items()
                    if (('lora_' in k and adapter_name in k) or ('bias' in k))
                }
            else:
                # Currently we only support lora
                raise NotImplementedError
            if model.modules_to_save is not None:
                for key, value in state_dict.items():
                    if any(f'{module_name}.modules_to_save.{adapter_name}' in
                           key for module_name in model.modules_to_save):
                        to_return[key] = value

            return to_return

        if not self.use_lora:
            return super().state_dict()
        state_dict = super().state_dict()
        to_return = get_peft_model_state_dict(self.llm, state_dict=state_dict)
        return OrderedDict(to_return)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
