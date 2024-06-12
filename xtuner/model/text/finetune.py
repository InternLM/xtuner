# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.distributed as dist
from accelerate import load_checkpoint_in_model
from mmengine import Config, print_log
from peft import LoraConfig
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (GenerationConfig, PreTrainedModel,
                          PreTrainedTokenizer, PreTrainedTokenizerFast)

from xtuner.chat.streamer import HFTextIteratorStreamer, HFTextStreamer
from xtuner.parallel.sequence import (get_sequence_parallel_group,
                                      get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel,
                                      reduce_sequence_parallel_loss,
                                      split_for_sequence_parallel)
from xtuner.registry import BUILDER
from xtuner.tools.utils import get_stop_criteria
from xtuner.types import ChatMessages, ChatTemplate, SampleParams
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from xtuner.utils.config import build_from_cfg_or_obj
from ..auto import download_model_from_hub
from ..base import BaseAlgorithm
from ..modules import dispatch_modules
from ..utils import (LoadWoInit, get_peft_model_state_dict,
                     prepare_for_llm_lora,
                     smart_tokenizer_and_embedding_resize)


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
            self.llm: PreTrainedModel = build_from_cfg_or_obj(llm, nn.Module)
            self.llm.config.use_cache = False

        tokenizer = build_from_cfg_or_obj(
            tokenizer, accept=(PreTrainedTokenizer, PreTrainedTokenizerFast))
        smart_tokenizer_and_embedding_resize(tokenizer, self.llm)
        self.tokenizer: PreTrainedModel = tokenizer

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

    @property
    def chat_template(self):
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

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def forward(self, data, data_samples=None, mode='loss'):
        """Overload parent class method, only support training."""

        if mode == 'loss':
            return self.compute_loss(data)
        else:
            raise NotImplementedError(
                f"{type(self)}'s forward is only supported for use during "
                'training. If you want to get predictions or chat, please '
                "directly use `llm`'s forward.")

    def _compute_sequence_parallel_loss(self, input_ids, labels,
                                        attention_mask, position_ids):

        sp_group = get_sequence_parallel_group()
        # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
        input_ids = split_for_sequence_parallel(
            input_ids, dim=1, sp_group=sp_group)
        labels = split_for_sequence_parallel(labels, dim=1, sp_group=sp_group)
        position_ids = split_for_sequence_parallel(
            position_ids, dim=1, sp_group=sp_group)

        outputs = self.llm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels)

        num_tokens = (labels != IGNORE_INDEX).sum()
        loss = reduce_sequence_parallel_loss(outputs.loss, num_tokens)
        return {'loss': loss}

    def compute_loss(self, data):

        input_ids = data['input_ids']
        labels = data['labels']
        # attention_mask = data['attention_mask']
        attention_mask = data.get('attention_mask', None)

        position_ids = self._compute_postion_ids(data)

        if self.use_varlen_attn:
            self._send_msg_to_dispatched_model(data)

        if get_sequence_parallel_world_size() > 1:
            return self._compute_sequence_parallel_loss(
                input_ids, labels, attention_mask, position_ids)

        outputs = self.llm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels)

        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def _compute_postion_ids(self, data):
        input_ids = data['input_ids']
        bs, tokens = input_ids.shape
        if self.use_varlen_attn:
            # TODO(pppppM) support bs>1 when use varlen attn
            assert bs == 1

            cumulative_len = data['cumulative_len'][0]
            position_ids = []
            for i in range(1, len(cumulative_len)):
                chunk_tokens = cumulative_len[i] - cumulative_len[i - 1]
                position_ids.append(torch.arange(chunk_tokens))
            position_ids = torch.cat(position_ids, dim=0).unsqueeze(0)
        else:

            position_ids = torch.arange(0, tokens).unsqueeze(0).repeat(bs, 1)
        return position_ids

    def _send_msg_to_dispatched_model(self, data):

        # TODO(pppppM) support bs>1 when use varlen attn
        assert len(data['cumulative_len']) == 1
        cumulative_len = data['cumulative_len'][0]
        max_seqlen = (cumulative_len[1:] - cumulative_len[:-1]).max()

        from mmengine import MessageHub
        rank = dist.get_rank()
        message_hub = MessageHub.get_instance('varlen_attn_args')
        message_hub.update_info(f'cumulative_len_rank_{rank}', cumulative_len)
        message_hub.update_info(f'max_seqlen_rank_{rank}', max_seqlen)

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
             prompt_or_messages: Union[str, ChatMessages],
             sample_params: Optional[SampleParams] = None,
             streamer=None) -> str:

        if isinstance(prompt_or_messages, str):
            messages = ChatMessages.from_str(prompt_or_messages)
        else:
            messages = prompt_or_messages

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

    def batch_infer(self, prompt_or_messages_list: Union[List[str],
                                                         List[ChatMessages]],
                    sample_params: SampleParams) -> List[str]:
        responses = []
        for p in prompt_or_messages_list:
            responses.append(self.chat(p, sample_params))
        return responses

    def save_pretrained(self, save_dir: str, config: str) -> 'TextFinetune':

        self.llm.save_pretrained(save_dir, safe_serialization=False)
        self.tokenizer.save_pretrained(save_dir)

        shutil.copy(config, os.path.join(save_dir, 'xtuner_config.py'))

    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        config: str,
                        from_hub: Literal['huggingface', 'modelscope'],
                        cache_dir: Optional[str] = None) -> 'TextFinetune':
        """Automatically load models from local storage or the HUB.

        Args:
            model_name_or_path (str): The model name, model path or repo id.
            config (str | None): The config path. Default is None.
            from_hub (str): The model hosting hub, modelscope, or huggingface.
                Default is huggingface.
            cache_dir (str | None):
                The save path when downloading the model. If it is None, it
                will be stored in the default location of the HUB. For
                Huggingface, it's ~/.cache/huggingface/hub, for ModelScope,
                it's ~/.cache/modelscope/hub.
        """
        model_name_or_path = download_model_from_hub(model_name_or_path,
                                                     from_hub, cache_dir)

        llm_conf = os.path.join(model_name_or_path, 'config.json')
        xtuner_conf = os.path.join(model_name_or_path, 'xtuner_config.py')
        tok_conf = os.path.join(model_name_or_path, 'tokenizer_config.json')

        has_llm = os.path.exists(llm_conf)
        has_conf = os.path.exists(xtuner_conf)
        has_tok = os.path.exists(tok_conf)

        if config:
            conf_path = config
            print_log(
                'A config has been detected as input, the model will be '
                'built with priority using the provided config'
                f'({config})',
                logger='current')
        elif not config and has_conf:
            conf_path = xtuner_conf
        else:
            raise RuntimeError('`xtuner_config.py` was not found in '
                               '{model_name_or_path}, please input a config '
                               'path.')

        config = Config.fromfile(conf_path)

        if has_tok:
            config.model.tokenizer.pretrained_model_name_or_path = model_name_or_path  # noqa: E501

        if has_llm:
            config.model.llm.pretrained_model_name_or_path = model_name_or_path

        model: TextFinetune = BUILDER.build(config.model)
        load_checkpoint_in_model(model.llm, model_name_or_path)

        return model

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

        ori_length = [len(ids) for ids in input_ids]
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

        # Some tokenizers have the same eos token and pad token, so input_ids
        # cannot be masked directly based on the pad token id.
        attention_mask = torch.zeros_like(input_ids).bool()
        for i in ori_length:
            attention_mask[:i] = True

        input_ids, labels, _, attention_mask = \
            pad_for_sequence_parallel(input_ids, labels, None,
                                      attention_mask)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(pad_index),
            'labels': labels,
            'cumulative_len': cumulative_len,
        }

        return {'data': data_dict, 'data_samples': None}

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
