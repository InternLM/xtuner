# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmengine.model import BaseModel
from peft import LoraConfig
from torch import nn

from xtuner.registry import BUILDER
from xtuner.utils.config import build_from_cfg_or_obj
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .utils import (LoadWoInit, enable_hf_model_gradient_checkpointing,
                    get_peft_model_state_dict, prepare_for_llm_lora,
                    prepare_for_vision_lora,
                    smart_tokenizer_and_embedding_resize)


class AgentFinetune(BaseModel):

    def __init__(
        self,
        llm,
        tokenizer=None,
        llm_lora=None,
        use_activation_checkpointing=True,
        use_varlen_attn=False,
    ):
        super().__init__()

        # Build the base language model without initialization.
        # This will greatly reduce the time to build the model.
        with LoadWoInit():
            self.llm = build_from_cfg_or_obj(llm, nn.Module)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm, use_varlen_attn=use_varlen_attn)

        if tokenizer is not None:
            if isinstance(tokenizer, dict):
                tokenizer = BUILDER.build(tokenizer)
            smart_tokenizer_and_embedding_resize(tokenizer, self.llm)

        if use_activation_checkpointing:
            # For backward compatibility
            enable_hf_model_gradient_checkpointing(self.llm)

        self.use_llm_lora = llm_lora is not None

        # Prepare the model for LoRA if specified
        if self.use_llm_lora:
            lora_conf = build_from_cfg_or_obj(llm_lora, accept=LoraConfig)
            self.llm = prepare_for_llm_lora(self.llm, lora_conf,
                                            use_activation_checkpointing)

        self._is_init = True

        # Determines whether to calculate attention based on the
        # seq_len dimension (use_varlen_attn = False) or the actual length of
        # the sequence.
        self.use_varlen_attn = use_varlen_attn

    def init_weights(self):
        """Parent class method.

        To avoid overwriting the loaded weights, overload it to an empty
        function.
        """
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        """Overload parent class method, only support training."""

        if mode == 'loss':
            return self.compute_loss(data)
        else:
            raise NotImplementedError(
                f"{type(self)}'s forward is only supported for use during "
                'training. If you want to get predictions or chat, please '
                "directly use `llm`'s forward.")

    def compute_loss(self, data):

        input_ids = data['input_ids']
        labels = data['labels']
        # position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        # breakpoint()
        bs, tokens = input_ids.shape
        if self.use_varlen_attn:
            assert bs == 1

            cumulative_len = data['cumulative_len'][0]
            max_seqlen = (cumulative_len[1:] - cumulative_len[:-1]).max()

            position_ids = []
            for i in range(1, len(cumulative_len)):
                chunk_tokens = cumulative_len[i] - cumulative_len[i - 1]
                position_ids.append(torch.arange(chunk_tokens))
            position_ids = torch.cat(position_ids, dim=0).unsqueeze(0)

            from mmengine import MessageHub
            rank = dist.get_rank()
            message_hub = MessageHub.get_instance('varlen_attn_args')
            message_hub.update_info(f'cumulative_len_rank_{rank}',
                                    cumulative_len)
            message_hub.update_info(f'max_seqlen_rank_{rank}', max_seqlen)
        else:

            position_ids = torch.arange(0, tokens).unsqueeze(0).repeat(bs, 1)

        outputs = self.llm(
            input_ids=input_ids,
            # position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels)

        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        else:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})

        return to_return

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
