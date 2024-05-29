from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


def _get_model_class(model_name_or_path: str):
    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True)
    config_class = type(config)
    if config_class in AutoModel._model_mapping:
        model_class = AutoModel._model_mapping[type(config)]
        model_base_class = model_class.__base__
        return model_class, model_base_class

    if 'AutoModel' in config.auto_map:
        module_file, causal_model_name = config.auto_map['AutoModel'].split(
            '.')
    elif 'AutoModelForCausalLM' in config.auto_map:
        module_file, causal_model_name = config.auto_map[
            'AutoModelForCausalLM'].split('.')
    else:
        raise Exception(
            f'config of {model_name_or_path} has no AutoModel or AutoModelForCausalLM in auto_map'  # noqa: E501
        )

    model_class_name = (causal_model_name.split('For')[0] + 'Model'
                        )  # e.g., "InternLM2Model"
    model_class = get_class_from_dynamic_module(
        f'{module_file}.{model_class_name}', model_name_or_path)
    model_base_class_name = (causal_model_name.split('For')[0] +
                             'PreTrainedModel'
                             )  # e.g., "InternLM2PreTrainedModel"
    model_base_class = get_class_from_dynamic_module(
        f'{module_file}.{model_base_class_name}', model_name_or_path)
    return model_class, model_base_class


def get_critic_model(model_name_or_path: str, head_name):
    model_class, model_base_class = _get_model_class(model_name_or_path)

    class CriticModel(model_base_class):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.model = model_class(config)
            self.head_name = head_name
            setattr(self, head_name,
                    nn.Linear(config.hidden_size, 1, bias=False))

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **_ignored,
        ) -> torch.Tensor:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = outputs[0]
            logits = getattr(self,
                             self.head_name)(hidden_states).squeeze(-1)[:, :-1]

            return SequenceClassifierOutputWithPast(logits=logits, )

    return CriticModel


def get_reward_model(model_name_or_path: str, head_name):
    model_class, model_base_class = _get_model_class(model_name_or_path)

    class RewardModel(model_base_class):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.model = model_class(config)
            self.head_name = head_name
            setattr(self, head_name,
                    nn.Linear(config.hidden_size, 1, bias=False))

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **_ignored,
        ) -> torch.Tensor:
            eos_indices = (
                attention_mask.size(1) - 1 -
                attention_mask.long().fliplr().argmax(dim=1, keepdim=True))
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = outputs[0]
            values = getattr(self, self.head_name)(hidden_states).squeeze(-1)
            reward_scores = values.gather(dim=1, index=eos_indices).squeeze(1)

            return SequenceClassifierOutputWithPast(logits=reward_scores, )

    return RewardModel
