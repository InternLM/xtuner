# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict
from contextlib import nullcontext

import torch
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from peft import get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import AutoConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.parallel.sequence import (get_sequence_parallel_group,
                                      get_sequence_parallel_world_size,
                                      reduce_sequence_parallel_loss,
                                      split_for_sequence_parallel)
from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, make_inputs_require_grad,
                    traverse_dict)


def smart_tokenizer_and_embedding_resize(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize embedding."""
    if is_deepspeed_zero3_enabled():
        import deepspeed

        params = [model.get_input_embeddings().weight]
        if model.get_output_embeddings(
        ) is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(
            params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    if len(tokenizer) > current_embedding_size:
        assert isinstance(model.get_output_embeddings(), nn.Linear)

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            num_new_tokens = len(tokenizer) - current_embedding_size
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        print_log(
            f'Resized token embeddings from {current_embedding_size} to '
            f'{len(tokenizer)}.', 'current')


class SupervisedFinetune(BaseModel):

    def __init__(self,
                 llm,
                 lora=None,
                 peft_model=None,
                 use_activation_checkpointing=True,
                 use_varlen_attn=False,
                 tokenizer=None,
                 max_position_embeddings=None):
        super().__init__()

        self.llm = self.build_llm_from_cfg(llm, use_varlen_attn,
                                           max_position_embeddings)

        if tokenizer is not None:
            if isinstance(tokenizer, dict):
                tokenizer = BUILDER.build(tokenizer)
            smart_tokenizer_and_embedding_resize(tokenizer, self.llm)

        self.llm.config.use_cache = False
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

    def build_llm_from_cfg(self, llm_cfg, use_varlen_attn,
                           max_position_embeddings):
        # For forward
        with LoadWoInit():
            if isinstance(llm_cfg, dict):
                llm = self._dispatch_lm_model_cfg(llm_cfg,
                                                  max_position_embeddings)
            llm = self._build_from_cfg_or_module(llm)

        llm.config.use_cache = False
        dispatch_modules(llm, use_varlen_attn=use_varlen_attn)
        return llm

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

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg,
                                           max_position_embeddings):
        if not hasattr(llm_cfg, 'rope_scaling'):
            print_log('Current model does not support RoPE scaling.',
                      'current')
            return

        current_max_length = getattr(llm_cfg, 'max_position_embeddings', None)
        if current_max_length and max_position_embeddings > current_max_length:
            print_log(
                f'Enlarge max model length from {current_max_length} '
                f'to {max_position_embeddings}.', 'current')
            scaling_factor = float(
                math.ceil(max_position_embeddings / current_max_length))
        else:
            print_log(
                'The input `max_position_embeddings` is smaller than '
                'origin max length. Consider increase input length.',
                'current')
            scaling_factor = 1.0
        cfg.rope_scaling = {'type': 'linear', 'factor': scaling_factor}

        return cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = ('LlamaConfig', 'GemmaConfig', 'MistralConfig',
                             'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig',
                             'Starcoder2Config', 'Starcoder2Config',
                             'Phi3Config')
        SUPPORT_FLASH_ATTN2 = ('InternLM2Config', 'LlamaConfig', 'GemmaConfig',
                               'MistralConfig', 'MixtralConfig', 'Qwen2Config',
                               'Qwen2MoeConfig', 'Starcoder2Config',
                               'Starcoder2Config', 'Phi3Config',
                               'DeepseekV2Config')

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        if getattr(cfg, 'attn_implementation', None) is not None:
            # Flash Attention 2.0 only supports torch.float16 and
            # torch.bfloat16 dtypes
            if cfg.attn_implementation == 'flash_attention_2':
                cfg.torch_dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = 'sdpa'

        return cfg

    @staticmethod
    def _prepare_for_qlora_zero3(cfg):
        if (not is_deepspeed_zero3_enabled()) or (not hasattr(
                cfg, 'quantization_config')):
            return cfg

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        cfg.torch_dtype = torch_dtype
        quantization_config = cfg.quantization_config
        quantization_config.bnb_4bit_compute_dtype = torch_dtype
        quantization_config.bnb_4bit_quant_storage = torch_dtype

        return cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        cfg = self._prepare_for_qlora_zero3(cfg)
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True)
        cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg = self._prepare_for_long_context_training(
                cfg, llm_cfg, max_position_embeddings)
        return cfg

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

    @staticmethod
    def _split_for_sequence_parallel(data):
        # attention mask should not be split
        ARGS_NEED_TO_SPLIT = ('input_ids', 'labels', 'position_ids')
        sp_group = get_sequence_parallel_group()
        for key in ARGS_NEED_TO_SPLIT:
            val = data.get(key, None)
            if val is not None:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                data[key] = split_for_sequence_parallel(
                    val, dim=1, sp_group=sp_group)
        return data

    def _compute_sequence_parallel_loss(self, data):
        data = self._split_for_sequence_parallel(data)
        outputs = self.llm(**data)
        labels = data['labels']
        num_tokens = (labels != -100).sum()
        sp_group = get_sequence_parallel_group()
        loss = reduce_sequence_parallel_loss(outputs.loss, num_tokens,
                                             sp_group)
        return {'loss': loss}

    def compute_loss(self, data, data_samples=None):
        if get_sequence_parallel_world_size() > 1:
            return self._compute_sequence_parallel_loss(data)
        else:
            outputs = self.llm(**data)
            loss_dict = {'loss': outputs.loss}
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

    def to_hf(self,
              cfg,
              save_dir,
              fp32=False,
              save_pretrained_kwargs={},
              **kwargs):
        self.llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            self.llm.half()
        if self.use_lora:
            print_log(f'Saving adapter to {save_dir}', 'current')
        else:
            print_log(f'Saving LLM tokenizer to {save_dir}', 'current')
            tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(save_dir)
            print_log(f'Saving LLM to {save_dir}', 'current')
        self.llm.save_pretrained(save_dir, **save_pretrained_kwargs)
        self.llm.config.use_cache = False
