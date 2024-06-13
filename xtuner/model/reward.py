# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
import os
import warnings
from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.distributed as dist
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from peft import get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          PreTrainedModel, PreTrainedTokenizer)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import no_init_weights

from xtuner.parallel.sequence import (gather_forward_split_backward,
                                      get_sequence_parallel_group,
                                      get_sequence_parallel_world_size,
                                      split_for_sequence_parallel)
from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, make_inputs_require_grad,
                    traverse_dict)


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


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


class RewardModel(BaseModel):

    def __init__(
        self,
        llm,
        lora=None,
        peft_model=None,
        use_activation_checkpointing=True,
        use_varlen_attn=False,
        tokenizer=None,
        max_position_embeddings=None,
        reward_token_id=None,
        loss_type='ranking',
        penalty_type='log_barrier',
        penalty_weight=0.01,
    ):
        super().__init__()
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)
            self.llm = self._build_from_cfg_or_module(llm).model
            self.v_head = nn.Linear(self.llm.config.hidden_size, 1, bias=False)
            # zero init
            self.v_head.weight.data.zero_()

        self.reward_token_id = reward_token_id
        assert loss_type in ('ranking',
                             'focal'), f'Unsupported loss type {loss_type}'
        self.loss_type = loss_type
        assert penalty_type in (
            'log_barrier', 'L2',
            'none'), f'Unsupported penalty type {penalty_type}'
        self.penalty_type = penalty_type
        self.penalty_weight = penalty_weight

        if tokenizer is not None:
            if isinstance(tokenizer, dict):
                tokenizer = BUILDER.build(tokenizer)
            smart_tokenizer_and_embedding_resize(tokenizer, self.llm)

        self.llm.config.use_cache = False
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
                               'Starcoder2Config', 'Phi3Config')

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
        labels = data.pop('labels', None)
        if mode == 'loss':
            return self.compute_loss(data, labels)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):
        hidden_states = self.llm(**data)[0]
        logits = self.v_head(hidden_states)
        return logits

    def predict(self, data, data_samples=None):
        hidden_states = self.llm(**data)[0]
        logits = self.v_head(hidden_states)
        logits_dict = [{'logits': log} for log in logits]
        return logits_dict

    @staticmethod
    def _split_for_sequence_parallel(data):
        # attention mask should not be split
        ARGS_NEED_TO_SPLIT = ('input_ids', 'position_ids')
        sp_group = get_sequence_parallel_group()
        for key in ARGS_NEED_TO_SPLIT:
            val = data.get(key, None)
            if val is not None:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                data[key] = split_for_sequence_parallel(
                    val, dim=1, sp_group=sp_group)
        return data

    def compute_loss(self, data, labels=None):
        if get_sequence_parallel_world_size() > 1:
            data = self._split_for_sequence_parallel(data)

        hidden_states = self.llm(**data)[0]
        logits = self.v_head(hidden_states)

        if get_sequence_parallel_world_size() > 1:
            logits = gather_forward_split_backward(
                logits,
                dim=1,
                sp_group=get_sequence_parallel_group(),
                grad_scale='up')

        chosen_idx = torch.where(labels == 0)
        rejected_idx = torch.where(labels == 1)
        chosen_logits = logits[chosen_idx]
        rejected_logits = logits[rejected_idx]

        num_samples = torch.tensor(len(chosen_logits)).float().to(
            hidden_states.device)
        avg_factor = 1.0 / num_samples
        avg_factor = reduce_mean(avg_factor).to(hidden_states.device)

        chosen_mean = reduce_mean(chosen_logits.mean().detach())
        rejected_mean = reduce_mean(rejected_logits.mean().detach())
        acc = reduce_mean(
            (chosen_logits > rejected_logits).sum() / num_samples).detach()
        num_tokens = torch.tensor(labels.shape[1]).float()

        # ranking loss
        if self.loss_type == 'ranking':
            rank_loss = self.ranking_loss(
                chosen_logits, rejected_logits, avg_factor=avg_factor)
        elif self.loss_type == 'focal':
            rank_loss = self.focal_loss(
                chosen_logits, rejected_logits, avg_factor=avg_factor)
        else:
            raise NotImplementedError(
                f'Unsupported loss type {self.loss_type}')

        # penalty loss
        if self.penalty_type == 'log_barrier':
            penalty = self.log_barrier_penalty(
                torch.cat([chosen_logits, rejected_logits]),
                lower_bound=-5,
                upper_bound=5,
                avg_factor=avg_factor)
        elif self.penalty_type == 'L2':
            penalty = self.l2_penalty(
                torch.cat([chosen_logits, rejected_logits]),
                avg_factor=avg_factor)
        elif self.penalty_type == 'none':
            penalty = 0
        else:
            raise NotImplementedError(
                f'Unsupported penalty type {self.penalty_type}')

        loss = rank_loss + self.penalty_weight * penalty
        loss_dict = {
            'loss': loss,
            'acc': acc,
            'chosen_score_mean': chosen_mean,
            'rejected_score_mean': rejected_mean,
            'num_samples': num_samples,
            'num_tokens': num_tokens,
        }

        return loss_dict

    def ranking_loss(self, chosen_logits, rejected_logits, avg_factor):
        rank_loss = -nn.functional.logsigmoid(chosen_logits - rejected_logits)
        return rank_loss.sum() * avg_factor

    def focal_loss(self, chosen_logits, rejected_logits, avg_factor):
        # focal ranking loss from InternLM2 paper https://arxiv.org/abs/2403.17297  # noqa
        rank_loss = -nn.functional.logsigmoid(chosen_logits - rejected_logits)
        p_ij = torch.sigmoid(chosen_logits - rejected_logits)
        p = 2 * torch.relu(p_ij - 0.5)
        gamma = 2
        focal_loss = ((1 - p)**gamma) * rank_loss
        return focal_loss.sum() * avg_factor

    def log_barrier_penalty(self,
                            logits,
                            lower_bound,
                            upper_bound,
                            epsilon=1e-3,
                            avg_factor=1):
        # log barrier penalty from InternLM2 paper https://arxiv.org/abs/2403.17297  # noqa
        logits_fp32 = logits.float()
        logits_clamped = torch.clamp(logits_fp32, lower_bound + epsilon,
                                     upper_bound - epsilon)
        penalty = -torch.log(upper_bound - logits_clamped) - torch.log(
            logits_clamped - lower_bound)
        return penalty.sum() * avg_factor

    def l2_penalty(self, logits, avg_factor=1):
        return (logits**2).sum() * avg_factor

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
        print(f'Saving LLM tokenizer to {save_dir}')
        tokenizer = BUILDER.build(cfg.tokenizer)
        tokenizer.save_pretrained(save_dir)

        if 'PeftModel' in self.llm.__class__.__name__:
            # merge adapter
            self.llm = self.llm.merge_and_unload()
        if 'InternLM2' in self.llm.__class__.__name__:
            from xtuner.tools.model_converters.modeling_internlm2_reward.modeling_internlm2 import \
                InternLM2ForRewardModel  # noqa
            print(f'Saving Reward Model to {save_dir}')
            hf_cfg = self.llm.config
            hf_cfg.reward_token_id = self.reward_token_id if \
                self.reward_token_id is not None else cfg.reward_token_id
            if not fp32:
                dtype = torch.float16
            else:
                dtype = torch.float32
            with no_init_weights():
                reward_model = InternLM2ForRewardModel._from_config(
                    hf_cfg, torch_dtype=dtype)
            reward_model.model.load_state_dict(self.llm.state_dict())
            reward_model.v_head.load_state_dict(self.v_head.state_dict())
            reward_model.save_pretrained(save_dir, **save_pretrained_kwargs)
            # fix auto_map in config
            with open(os.path.join(save_dir, 'config.json')) as fp:
                config_dict = json.load(fp)
            config_dict['auto_map'][
                'AutoModel'] = 'modeling_internlm2.InternLM2ForRewardModel'
            config_dict['auto_map'].pop('AutoModelForCausalLM', None)
            with open(os.path.join(save_dir, 'config.json'), 'w') as fp:
                json.dump(config_dict, fp, indent=2)
        else:
            warnings.warn(
                f'The pretrained model type: {self.llm.__class__.__name__} '
                'has no reward model class defined. Use '
                'the SequenceClassification class instead.'
                'You can refer to `xtuner/tools/model_converters/modeling_internlm2_reward` '  # noqa
                'to implement the reward model class.')

            hf_cfg = self.llm.config
            hf_cfg.num_labels = 1  # set the output dim to 1
            try:
                with no_init_weights():
                    reward_model = \
                        AutoModelForSequenceClassification.from_config(hf_cfg)
            except Exception as e:
                warnings.warn(f'Cannot find SequenceClassification class '
                              f'from transformers: {e}, \n'
                              'try to find it in the dynamic module.')
                module_file, causal_model_name = hf_cfg.auto_map[
                    'AutoModelForCausalLM'].split('.')
                seqcls_model_name = causal_model_name.split(
                    'For')[0] + 'ForSequenceClassification'
                seqcls_class = get_class_from_dynamic_module(
                    f'{module_file}.{seqcls_model_name}', hf_cfg._name_or_path)
                with no_init_weights():
                    reward_model = seqcls_class(hf_cfg)
            reward_model.model.load_state_dict(self.llm.state_dict())
            reward_model.score.load_state_dict(self.v_head.state_dict())
            reward_model.save_pretrained(save_dir, **save_pretrained_kwargs)
