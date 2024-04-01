# Copyright (c) OpenMMLab. All rights reserved.

from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from accelerate import load_checkpoint_in_model
from peft import LoraConfig
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (GenerationConfig, PreTrainedModel,
                          PreTrainedTokenizer, PreTrainedTokenizerFast)

from xtuner.chat.streamer import HFTextIteratorStreamer, HFTextStreamer
from xtuner.model.utils import guess_load_checkpoint
from xtuner.tools.utils import get_stop_criteria
from xtuner.types import ChatMessages, ChatTemplate, SampleParams
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from xtuner.utils.config import build_from_cfg_or_obj
from .auto import download_model_from_hub
from .base import BaseAlgorithm
from .modules import dispatch_modules
from .utils import (LoadWoInit, get_peft_model_state_dict,
                    prepare_for_llm_lora, smart_tokenizer_and_embedding_resize)


class TextFinetune(BaseAlgorithm):

    def __init__(
        self,
        llm: Union[PreTrainedModel, Dict],
        tokenizer: Union[PreTrainedTokenizer, Dict],
        chat_template: Union[ChatTemplate, Dict],
        llm_lora: Optional[Union[LoraConfig, Dict]] = None,
        use_gradient_checkpointing: bool = True,
        use_varlen_attn: bool = False,
    ):
        super().__init__()

        _chat_tmpl = build_from_cfg_or_obj(chat_template, accept=ChatTemplate)
        self._chat_template = _chat_tmpl

        # Build the base language model without initialization.
        # This will greatly reduce the time to build the model.
        with LoadWoInit():
            self._llm: PreTrainedModel = build_from_cfg_or_obj(llm, nn.Module)
            self._llm.config.use_cache = False

        tokenizer = build_from_cfg_or_obj(
            tokenizer, accept=(PreTrainedTokenizer, PreTrainedTokenizerFast))
        smart_tokenizer_and_embedding_resize(tokenizer, self.llm)
        self._tokenizer: PreTrainedModel = tokenizer

        self.with_lora = llm_lora is not None
        # Prepare the model for LoRA if specified
        if self.with_lora:
            lora_conf = build_from_cfg_or_obj(llm_lora, accept=LoraConfig)
            self.llm = prepare_for_llm_lora(self.llm, lora_conf)

        # Determines whether to calculate attention based on the
        # seq_len dimension (use_varlen_attn = False) or the actual length of
        # the sequence.
        self.use_varlen_attn = use_varlen_attn
        dispatch_modules(self.llm, use_varlen_attn=use_varlen_attn)

        if use_gradient_checkpointing:
            self.gradient_checkpointing_enable()

        self.avoid_override_weights()

    @property
    def llm(self) -> PreTrainedModel:
        return self._llm

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    def chat_template(self) -> ChatTemplate:
        return self._chat_template

    def gradient_checkpointing_enable(self):
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

    def forward(self, data, data_samples=None, mode='loss'):
        """Overload parent class method, only support training."""

        if mode == 'loss':
            return self.compute_loss(data)
        else:
            raise NotImplementedError(
                f"{type(self)}'s forward is only supported for use during "
                'training. If you want to get predictions or chat, please '
                "directly use `llm`'s forward.")

    def _compute_postion_ids(self, data):
        input_ids = data['input_ids']
        bs, tokens = input_ids.shape
        if self.use_varlen_attn:
            # TODO(pppppM) support bs>1 when use varlen attn
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

    def compute_loss(self, data):

        input_ids = data['input_ids']
        labels = data['labels']
        attention_mask = data['attention_mask']

        position_ids = self._compute_postion_ids(data)

        outputs = self.llm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels)

        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()

        if not self.with_lora:
            return state_dict
        to_return = get_peft_model_state_dict(self.llm, state_dict=state_dict)
        return OrderedDict(to_return)

    def parse_sample_params(self, params: SampleParams) -> GenerationConfig:

        if params is None:
            params = SampleParams()

        hf_gen_config = GenerationConfig(
            max_new_tokens=params.max_new_tokens,
            do_sample=params.temperature > 0,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            seed=params.seed,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id)

        stop_words = params.stop_words
        stop_words.extend(self.chat_template.stop_words)

        return hf_gen_config, stop_words

    def create_streamer(self, iterable=False):
        if iterable:
            return HFTextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                chat_template=self.chat_template)
        else:
            return HFTextStreamer(
                self.tokenizer,
                skip_prompt=True,
                chat_template=self.chat_template)

    def chat(self,
             messages: ChatMessages,
             sample_params: Optional[SampleParams] = None,
             streamer=None):

        prompt = messages.get_prompt(self.chat_template)
        ids = self.tokenizer.encode(prompt, return_tensors='pt')

        hf_gen_config, stop_words = self.parse_sample_params(sample_params)

        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)

        generate_output = self.llm.generate(
            inputs=ids.to(self.llm.device),
            streamer=streamer,
            generation_config=hf_gen_config,
            stopping_criteria=stop_criteria)

        output = self.tokenizer.decode(
            generate_output[0][len(ids[0]):], skip_special_tokens=True)

        for word in stop_words:
            output = output.rstrip(word)

        return output

    def batch_infer(self, messages: List[ChatMessages],
                    sample_params: SampleParams):
        pass

    def save_checkpoint(self,
                        save_dir: str,
                        to_hub: bool = True) -> 'TextFinetune':

        if to_hub:
            self.llm.save_pretrained(save_dir, safe_serialization=False)
        else:
            raise NotImplementedError

    def load_checkpoint(self,
                        ckpt_dir: str,
                        from_hub: bool = False) -> BaseAlgorithm:

        if from_hub:
            ckpt_dir = download_model_from_hub(ckpt_dir, from_hub)

            load_checkpoint_in_model(self.llm, ckpt_dir)

        else:
            state_dict = guess_load_checkpoint(ckpt_dir)
            self.load_state_dict(state_dict)

    @classmethod
    def dataloader_collate_fn(cls, instances):

        pad_index = DEFAULT_PAD_TOKEN_INDEX

        input_ids = []
        labels = []
        cumulative_len = []

        for i, data in enumerate(instances):
            input_ids.append(torch.LongTensor(data['input_ids']))
            labels.append(torch.LongTensor(data['labels']))

            if 'cumulative_len' in data:
                cumulative_len.append(torch.IntTensor(data['cumulative_len']))

        if len(instances) > 1:
            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=pad_index)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX)
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)

        if len(cumulative_len) == 0:
            cumulative_len = None

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(pad_index),
            'labels': labels,
            'cumulative_len': cumulative_len,
        }

        return {'data': data_dict, 'data_samples': None}
