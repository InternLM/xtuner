# Copyright (c) OpenMMLab. All rights reserved.

from .llava import LLaVAModel
import torch

from xtuner.registry import BUILDER
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .utils import (LoadWoInit, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal)

from xtuner.engine.optimizers import get_layer_depth_for_CLIPVisionModel, get_layer_depth_for_InternVisionModel
import types
from mmengine.logging import print_log
import torch.nn as nn
from fairscale.nn.checkpoint import checkpoint_wrapper


class InternVL_v1_5_LLaVAModel(LLaVAModel):
    def __init__(self, llm,
                 visual_encoder,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 max_position_embeddings=None,
                 image_processor=None,
                 tokenizer=None,
                 template=None,
                 use_lldr=False,  # LearningRateDecayOptimWrapperConstructor
                 merge_type='pixel_shuffle',  # or pixel_shuffle
                 downsample_ratio=0.5,
                 custom_mlp=False):
        super(LLaVAModel, self).__init__()
        self.downsample_ratio = downsample_ratio

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.merge_type = merge_type
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

            self.llm = self._build_from_cfg_or_module(llm)
            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder)

            if use_lldr:
                # The following code is only meaningful when the optim_wrapper configuration
                # includes `LearningRateDecayOptimWrapperConstructor`. Otherwise, it will be ignored.
                if self._get_model_class_name(self.visual_encoder) == 'CLIPVisionModel':
                    self.visual_encoder.get_layer_depth = types.MethodType(get_layer_depth_for_CLIPVisionModel,
                                                                           self.visual_encoder)
                elif self._get_model_class_name(self.visual_encoder) == 'InternVisionModel':
                    self.visual_encoder.get_layer_depth = types.MethodType(get_layer_depth_for_InternVisionModel,
                                                                           self.visual_encoder)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        self.custom_mlp = custom_mlp
        if custom_mlp is True:
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(self.visual_encoder.config.hidden_size * int(1 / self.downsample_ratio) ** 2),
                nn.Linear(self.visual_encoder.config.hidden_size * int(1 / self.downsample_ratio) ** 2,
                          self.llm.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
            )
            self.mlp1 = self.mlp1.to(self.visual_encoder.dtype)
            self.mlp1 = checkpoint_wrapper(self.mlp1)
        else:
            projector_config = ProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size * (int(1 / self.downsample_ratio) ** 2),
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

            if self.visual_encoder.__class__.__name__ == 'InternVisionModel':
                pass
            else:
                if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                    self.visual_encoder.enable_input_require_grads()
                else:
                    self.visual_encoder.get_input_embeddings(
                    ).register_forward_hook(make_inputs_require_grad)
            if custom_mlp is False:
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
        self.template = template

        print_log(self, logger='current')

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        if self.custom_mlp is False:
            self.projector.gradient_checkpointing_enable()
        if self.visual_encoder.__class__.__name__ == 'InternVisionModel':
            pass
        else:
            self.visual_encoder.gradient_checkpointing_enable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        if self.custom_mlp is False:
            self.projector.gradient_checkpointing_disable()
        if self.visual_encoder.__class__.__name__ == 'InternVisionModel':
            pass
        else:
            self.visual_encoder.gradient_checkpointing_disable()

    # The following code is only meaningful when the optim_wrapper configuration
    # includes `LearningRateDecayOptimWrapperConstructor`. Otherwise, it will be ignored.
    def get_layer_depth(self, param_name: str, prefix: str = 'visual_encoder.vision_model.'):
        assert hasattr(self.visual_encoder, 'get_layer_depth'), \
            'The visual_encoder does not have `get_layer_depth` method.'
        if self._get_model_class_name(self.visual_encoder) == 'CLIPVisionModel':
            prefix = 'visual_encoder.vision_model.'
        elif self._get_model_class_name(self.visual_encoder) == 'InternVisionModel':
            prefix = 'visual_encoder.'
        return self.visual_encoder.get_layer_depth(param_name, prefix)

    def _prepare_data_for_llm(self, data):
        if 'pixel_values' in data:
            new_image_feature = self.__preprocess_for_pixel_values(data)
            data['pixel_values'] = new_image_feature
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        return data

    def __preprocess_for_pixel_values(self, data):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat([image.to(self.visual_encoder.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        # b*n, hw, d
        visual_outputs = self.visual_encoder(concat_images, output_hidden_states=True)

        if self._get_model_class_name(self.visual_encoder) in ['CLIPVisionModel', 'InternVisionModel']:
            vit_embeds = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        elif self._get_model_class_name(self.visual_encoder) == 'SiglipVisionModel':
            vit_embeds = visual_outputs.hidden_states[self.visual_select_layer]
        else:
            raise NotImplementedError

        # n, hw, c
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        # n,h'w',c'
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        if self.custom_mlp is False:
            vit_embeds = self.projector(vit_embeds)
        else:
            vit_embeds = self.mlp1(vit_embeds)

        split_sizes = [image.shape[0] for image in pixel_values]
        image_features = torch.split(vit_embeds, split_sizes, dim=0)

        new_image_feature = []
        for image_feature in image_features:
            B, N, C = image_feature.shape
            image_feature = image_feature.reshape(B * N, C)
            new_image_feature.append(image_feature)

        # TODO:  for 这种写法无法在 zero + checkpoint 情况下使用
        # if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 5:
        #     pixel_values = [x if x.ndim == 4 else x.unsqueeze(0) for x in pixel_values]
        # assert isinstance(pixel_values, list)

        # for bs in range(len(pixel_values)):
        #     # 这样可以省一点显存，虽然会慢一点
        #     # n, c, h, w
        #     visual_outputs = self.visual_encoder(
        #         pixel_values[bs].to(self.visual_encoder.dtype), output_hidden_states=True)
        #
        #     if self._get_model_class_name(self.visual_encoder) == 'CLIPVisionModel':
        #         vit_embeds = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        #     elif self._get_model_class_name(self.visual_encoder) == 'SiglipVisionModel':
        #         vit_embeds = visual_outputs.hidden_states[self.visual_select_layer]
        #     else:
        #         raise NotImplementedError
        #     # n, hw, c
        #     h = w = int(vit_embeds.shape[1] ** 0.5)
        #     vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        #     vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        #     # n,h'w',c'
        #     vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        #
        #     vit_embeds = self.projector(vit_embeds)
        #     B, N, C = vit_embeds.shape
        #     vit_embeds = vit_embeds.reshape(B * N, C)
        #     new_image_feature.append(vit_embeds)
        return new_image_feature

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, h, w, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x
