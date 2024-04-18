# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, GenerationConfig

from xtuner.registry import BUILDER
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict, s2_forward)
from xtuner.tools.utils import get_stop_criteria
from xtuner.dataset.utils import expand2square, load_image
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)
from functools import reduce
from mmengine.logging import print_log

class LLaVAModel(BaseModel):

    def __init__(self,
                 llm,
                 visual_encoder,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 visual_select_layer=-2,
                 token_merge_ratio=1,
                 s2_scales=None,  # [1, 2] or [1,2,3]
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 max_position_embeddings=None,
                 image_processor=None,
                 tokenizer=None,
                 template=None):
        super().__init__()
        self.s2_scales = s2_scales
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

            self.llm = self._build_from_cfg_or_module(llm)
            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        assert int(token_merge_ratio**0.5)**2 == token_merge_ratio, \
            '`token_merge_ratio` must be a square number.'
        self.token_merge_ratio = int(token_merge_ratio)

        visual_hidden_size = self.visual_encoder.config.hidden_size * token_merge_ratio
        self.s2_scales = s2_scales
        if s2_scales is not None:
            assert 1 in s2_scales, 'The scale of the original image must be included.'
            total_scales = reduce(lambda x, y: x * y, s2_scales)
            visual_hidden_size = visual_hidden_size * total_scales

        projector_config = ProjectorConfig(
            visual_hidden_size=visual_hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth)
        self.projector = ProjectorModel(projector_config).to(
            self.visual_encoder.dtype)

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)

        self.use_activation_checkpointing = use_activation_checkpointing
        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                self.visual_encoder.enable_input_require_grads()
            else:
                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = image_processor
        if image_processor is not None:
            self.image_processor = BUILDER.build(image_processor)

        if s2_scales is not None:
            if hasattr(self.image_processor, 'crop_size'):
                orig_img_size = self.image_processor.crop_size['height']
            else:
                orig_img_size = self.image_processor.size['height']
            self.orig_img_size = orig_img_size
            self.s2_img_sizes = [int(orig_img_size * scale) for scale in s2_scales]

        self.template = template
        print_log(self, logger='current')

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        if self.use_activation_checkpointing:
            self.activation_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if self.use_activation_checkpointing:
            self.activation_checkpointing_disable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.visual_encoder.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        return to_return

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg,
                                           max_position_embeddings):

        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {'factor': 1}

        orig_rope_scaling_factor = orig_rope_scaling[
            'factor'] if 'factor' in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(
                    math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {
                    'type': 'linear',
                    'factor': scaling_factor
                }

        # hardcode for internlm2
        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = ('LlamaConfig', 'GemmaConfig', 'MistralConfig',
                             'MixtralConfig', 'Qwen2Config',
                             'Starcoder2Config', 'Starcoder2Config')
        SUPPORT_FLASH_ATTN2 = ('InternLM2Config', 'LlamaConfig', 'GemmaConfig',
                               'MistralConfig', 'MixtralConfig', 'Qwen2Config',
                               'Starcoder2Config', 'Starcoder2Config')

        if SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch.bfloat16 \
                if torch.cuda.is_bf16_supported() else torch.float16
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(
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

    @staticmethod
    def _merge_tokens(tokens, token_merge_ratio):
        if token_merge_ratio > 1:
            # B, N, C
            b, n, c = tokens.shape
            h = w = int(n ** 0.5)
            h_ratio = w_ratio = int(token_merge_ratio ** 0.5)
            assert h * w == n
            assert n % token_merge_ratio == 0, 'The number of visual tokens is not divisible by `token_merge_ratio`.'
            # B, H, W, C
            tokens = tokens.view(b, h, w, c)
            # B, H, W // w_r, C * w_r
            tokens = tokens.view(b, h, w // w_ratio, c * w_ratio)
            # B, W // w_r, H, C * w_r
            tokens = tokens.permute(0, 2, 1, 3).contiguous()
            # B, W // w_r, H // h_r, C * w_r * h_r
            tokens = tokens.view(b, w // w_ratio, h // h_ratio, 
                                 c * w_ratio * h_ratio)
            # B, W * H // w_r // h_r, C * w_r * h_r
            tokens = tokens.view(b, w * h // w_ratio // h_ratio, 
                                 c * w_ratio * h_ratio)
        return tokens

    @staticmethod
    def _get_model_class_name(model):
        if model.__class__.__name__ == 'PeftModel':
            base_model = model.base_model.model
        else:
            base_model = model
        return base_model.__class__.__name__

    def __forward_feature(self, images):
        visual_outputs = self.visual_encoder(images.to(self.visual_encoder.dtype), output_hidden_states=True)
        visual_outputs = visual_outputs.hidden_states[self.visual_select_layer]
        if self._get_model_class_name(self.visual_encoder) != 'SiglipVisionModel':
            visual_outputs = visual_outputs[:, 1:]
        return visual_outputs

    def _prepare_data_for_llm(self, data):
        if 'pixel_values' in data:
            if self.s2_scales is None:
                visual_outputs = self.__forward_feature(data['pixel_values'])
                visual_outputs = self._merge_tokens(visual_outputs, self.token_merge_ratio)
            else:
                visual_outputs = s2_forward(self.__forward_feature, data['pixel_values'],
                                            img_sizes=self.s2_img_sizes,
                                            max_split_size=self.orig_img_size)

            pixel_values = self.projector(visual_outputs)

            data['pixel_values'] = pixel_values
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        return data

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            data = self._prepare_data_for_llm(data)
            return self.compute_loss(data, data_samples)
        elif mode == 'predict' or mode == 'generate':
            data = self._prepare_data_for_llm(data)
            return self.generate(data, data_samples)
        elif mode == 'chat':
            return self.chat(data)
        else:
            raise NotImplementedError

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def preparing_for_generation(self, metainfo: dict = None):
        default_generation_kwargs = dict(
            max_new_tokens=100,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id)
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)

        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

    def generate(self, data, data_samples=None):
        generate_output = self.llm.generate(
            **data,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)

        prediction = self.tokenizer.decode(
            generate_output[0], skip_special_tokens=True).strip()

        return dict(prediction=prediction)

    def chat(self, data, system=''):
        # single image and single text mode
        instruction = self.template.get('INSTRUCTION', '{input}')

        sample_image = data['image']
        sample_input = data['text']

        image = expand2square(
            sample_image,
            tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.to(self.visual_encoder.device)
        sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
        if system != '':
            system = self.template.get(
                'SYSTEM', '{system}\n').format(system=system)

        inputs = (system + instruction).format(input=sample_input, round=1)
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(
                    chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        input_ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            input_ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                input_ids.append(IMAGE_TOKEN_INDEX)
        input_ids = torch.tensor(input_ids).to(self.visual_encoder.device)

        data['input_ids'] = input_ids.unsqueeze(0)
        data['pixel_values'] = image.unsqueeze(0)

        mm_inputs = self._prepare_data_for_llm(data)
        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)

        prediction = self.tokenizer.decode(
            generate_output[0], skip_special_tokens=True).strip()

        return dict(prediction=prediction, inputs=inputs)
