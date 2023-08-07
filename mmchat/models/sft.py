# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
from collections import OrderedDict

import torch
import transformers
from mmengine import print_log
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from peft import PeftType, get_peft_model, prepare_model_for_kbit_training
from torch import nn

from mmchat.registry import LLM, MODELS, TOKENIZER


def traverse_dict(d):
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                if 'type' in value and dataclasses.is_dataclass(value['type']):
                    builder = value.pop('type')
                    new_value = builder(**value)
                    d[key] = new_value
                    print_log(f'{key} convert to {builder}')
                else:
                    traverse_dict(value)
    elif isinstance(d, list):
        for element in d:
            traverse_dict(element)


def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size
    not be divisible by 64.
    """
    model_vocab_size = model.get_output_embeddings().weight.size(0)
    model.resize_token_embeddings(len(tokenizer))
    num_new_tokens = len(tokenizer) - model_vocab_size

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SupervisedFinetune(BaseModel):

    def __init__(self,
                 llm,
                 data_preprocessor=None,
                 tokenizer=None,
                 lora=None,
                 peft_model=None):
        super().__init__(data_preprocessor)
        self.llm = self._build_from_cfg_or_module(llm, LLM)
        self.llm.config.use_cache = False
        tokenizer = TOKENIZER.build(tokenizer)
        smart_tokenizer_and_embedding_resize(tokenizer, self.llm)

        self.lora = lora
        self.peft_model = peft_model
        self.use_lora = lora is not None
        if self.use_lora:
            self._prepare_for_lora(lora, peft_model)

        self._is_init = True

    def _prepare_for_lora(self, lora, peft_model=None):
        self.llm = prepare_model_for_kbit_training(self.llm)
        lora = MODELS.build(lora)
        if lora.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora.target_modules = modules

        self.llm = get_peft_model(self.llm, lora)
        if peft_model is not None:
            _ = load_checkpoint(self, peft_model)

    def init_weights(self):
        pass

    def _build_from_cfg_or_module(self, cfg_or_mod, registry):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return registry.build(cfg_or_mod)
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

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

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
