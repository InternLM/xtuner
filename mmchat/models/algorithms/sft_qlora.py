from collections import OrderedDict

import bitsandbytes as bnb
import torch
import torch.nn as nn
from peft import (PeftType, PromptLearningConfig, PeftConfig, get_peft_model,
                  prepare_model_for_kbit_training)

from mmchat.registry import MODELS
from .sft import SupervisedFinetune


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    # cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SupervisedQloraFinetune(SupervisedFinetune):

    def __init__(self, llm, lora, data_preprocessor=None):
        super().__init__(llm, data_preprocessor)

        self.llm = prepare_model_for_kbit_training(self.llm)

        modules = find_all_linear_names(self.llm)

        if isinstance(lora, PeftConfig):
            lora = lora
        elif isinstance(lora, dict):
            lora = MODELS.build(lora)
        else:
            raise NotImplementedError
        
        lora.target_modules = modules

        self.llm = get_peft_model(self.llm, lora)

        for name, module in self.llm.named_modules():
            # todo
            # if isinstance(module, LoraLayer):
            #     module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            # if 'lm_head' in name or 'embed_tokens' in name:
            #     if hasattr(module, 'weight'):
            #         if module.weight.dtype == torch.float32:
            #             module = module.to(torch.float16)
        self._is_init = True

    def init_weights(self):
        pass

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):

        def get_peft_model_state_dict(model,
                                      state_dict=None,
                                      adapter_name='default'):
            # Modified from `https://github.com/huggingface/peft/blob/main/src
            # /peft/utils/save_and_load.py`

            config = model.peft_config[adapter_name]
            if state_dict is None:
                state_dict = model.state_dict()
            if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
                # to_return = lora_state_dict(model,
                #                             bias=model.peft_config.bias)
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
                if config.peft_type == PeftType.ADALORA:
                    rank_pattern = config.rank_pattern
                    if rank_pattern is not None:
                        rank_pattern = {
                            k.replace(f'.{adapter_name}', ''): v
                            for k, v in rank_pattern.items()
                        }
                        config.rank_pattern = rank_pattern

            elif config.peft_type == PeftType.ADAPTION_PROMPT:
                to_return = {
                    k: state_dict[k]
                    for k in state_dict
                    if k.split('.')[-1].startswith('adaption_')
                }
            elif isinstance(config, PromptLearningConfig):
                to_return = {}
                if config.inference_mode:
                    prompt_embeddings = model.prompt_encoder[
                        adapter_name].embedding.weight
                else:
                    prompt_embeddings = model.get_prompt_embedding_to_save(
                        adapter_name)
                to_return['prompt_embeddings'] = prompt_embeddings
            else:
                raise NotImplementedError
            if model.modules_to_save is not None:
                for key, value in state_dict.items():
                    if any(f'{module_name}.modules_to_save.{adapter_name}' in
                           key for module_name in model.modules_to_save):
                        to_return[key.replace('modules_to_save.', '')] = value

            return to_return

        state_dict = super().state_dict()
        to_return = get_peft_model_state_dict(self.llm, state_dict=state_dict)
        return OrderedDict(to_return)
