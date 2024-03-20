# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn.functional as F
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from peft import get_peft_model, prepare_model_for_kbit_training
from torch import nn

from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .utils import (LoadWoInit, create_reference_model, find_all_linear_names,
                    get_peft_model_state_dict, make_inputs_require_grad,
                    traverse_dict)


class DPO(BaseModel):

    def __init__(self,
                 llm,
                 beta=0.1,
                 lora=None,
                 peft_model=None,
                 use_activation_checkpointing=True,
                 use_varlen_attn=False):
        super().__init__()
        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
        self.llm.config.use_cache = False
        self.beta = beta
        dispatch_modules(self.llm, use_varlen_attn=use_varlen_attn)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)

            # enable gradient checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

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
        # Determines whether to calculate attention based on the
        # seq_len dimension (use_varlen_attn = False) or the actual length of
        # the sequence.
        self.use_varlen_attn = use_varlen_attn
        # TODO: a more feasible way to set ref_model.
        # Now the ref_model is a deepcopy of self.llm
        self.ref_model = create_reference_model(self.llm)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

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
        len_chosen = len(data['input_ids']) // 2
        assert len_chosen > 0

        all_logits = self.llm(**data).logits
        all_ref_logits = self.ref_model(**data).logits

        labels = data['labels']
        labels[labels == -100] = 0
        loss_mask = labels != 0

        per_token_logps = torch.gather(
            all_logits.log_softmax(-1), dim=2,
            index=labels.unsqueeze(2)).squeeze(2)
        per_ref_token_logps = torch.gather(
            all_ref_logits.log_softmax(-1), dim=2,
            index=labels.unsqueeze(2)).squeeze(2)

        all_logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        all_ref_logps = (per_ref_token_logps *
                         loss_mask).sum(-1) / loss_mask.sum(-1)

        policy_chosen_logps = all_logps[:len_chosen]
        policy_rejected_logps = all_logps[len_chosen:]
        reference_chosen_logps = all_ref_logps[:len_chosen]
        reference_rejected_logps = all_ref_logps[len_chosen:]

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (
            policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (
            policy_rejected_logps - reference_rejected_logps)

        loss_dict = {
            'loss': loss,
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards
        }
        return loss_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if not self.use_lora:
            return state_dict
        to_return = get_peft_model_state_dict(self.llm, state_dict=state_dict)
        return OrderedDict(to_return)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
