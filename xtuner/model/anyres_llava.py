# Copyright (c) OpenMMLab. All rights reserved.

from ..dataset.utils import get_anyres_image_grid_shape, unpad_image
from .llava import LLaVAModel
from collections import OrderedDict
import torch

from xtuner.registry import BUILDER
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .utils import (LoadWoInit,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal)

import torch.nn as nn


class AnyResLLaVAModel(LLaVAModel):

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
                 image_grid_pinpoints=None,
                 merge_type='simple',  # or pixel_shuffle
                 token_merge_ratio=4):
        super(LLaVAModel, self).__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.merge_type = merge_type
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

            self.llm = self._build_from_cfg_or_module(llm)
            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        if token_merge_ratio == -1:
            projector_config = ProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth)
        else:
            projector_config = ProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size * token_merge_ratio,
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
        self.template = template

        self.token_merge_ratio = token_merge_ratio
        self.image_newline = nn.Parameter(
            torch.randn(
                self.llm.config.hidden_size, dtype=self.visual_encoder.dtype))
        self.image_grid_pinpoints = image_grid_pinpoints
        self.mm_patch_merge_type = 'spatial_unpad'
        self.image_aspect_ratio = 'anyres'

    def state_dict(self, *args, **kwargs):
        state_dict = super(LLaVAModel, self).state_dict(*args, **kwargs)
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
        # Step 4. Image Newline
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'image_newline.' in k})
        return to_return

    def _prepare_data_for_llm(self, data):
        if 'pixel_values' in data:
            new_image_feature = self.__preprocess_for_pixel_values(data)
            data['pixel_values'] = new_image_feature
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        return data

    def __preprocess_for_pixel_values(self, data):
        orig_sizes = data['orig_size']
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat([image for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        # b*n, 27*27, d
        visual_outputs = self.visual_encoder(
            concat_images.to(self.visual_encoder.dtype), output_hidden_states=True)

        if self._get_model_class_name(self.visual_encoder) == 'CLIPVisionModel':
            visual_outputs = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        elif self._get_model_class_name(self.visual_encoder) == 'SiglipVisionModel':
            visual_outputs = visual_outputs.hidden_states[self.visual_select_layer]
        else:
            raise NotImplementedError

        bs, pn, hs = visual_outputs.shape
        # token merge
        if self.token_merge_ratio != -1:
            # 27 不是偶数，不能被整除，需要 hard code 处理下
            if pn == 27 * 27:
                if self.merge_type == 'simple':
                    # 直接减掉最后 1 个 token，减掉点，确保能被整除
                    visual_outputs = visual_outputs[:, :-1]
                    visual_outputs = visual_outputs.reshape(bs, (pn - 1) // self.token_merge_ratio, int(hs * 4))
                else:
                    # 只能补 token 了
                    h_ratio = w_ratio = int(self.token_merge_ratio ** 0.5)
                    visual_outputs = visual_outputs.reshape(bs, 27, 27, -1)
                    # pad 为 28*28
                    visual_outputs = torch.cat(
                        (visual_outputs, torch.zeros(bs, 1, 27, hs, device=visual_outputs.device,dtype=visual_outputs.dtype)), dim=1)
                    visual_outputs = torch.cat(
                        (visual_outputs, torch.zeros(bs, 28, 1, hs, device=visual_outputs.device,dtype=visual_outputs.dtype)), dim=2)

                    # B, H, W // w_r, C * w_r
                    visual_outputs = visual_outputs.view(bs, 28, 28 // w_ratio, hs * w_ratio)
                    # B, W // w_r, H, C * w_r
                    visual_outputs = visual_outputs.permute(0, 2, 1, 3).contiguous()
                    # B, W // w_r, H // h_r, C * w_r * h_r
                    visual_outputs = visual_outputs.view(bs, 28 // w_ratio, 28 // h_ratio,
                                         hs * w_ratio * h_ratio)
                    # B, W * H // w_r // h_r, C * w_r * h_r
                    visual_outputs = visual_outputs.view(bs, 28 * 28 // w_ratio // h_ratio,
                                         hs * w_ratio * h_ratio).contiguous()

        # b*n, 182, d
        image_features = self.projector(visual_outputs)

        split_sizes = [image.shape[0] for image in pixel_values]
        image_features = torch.split(image_features, split_sizes, dim=0)

        new_image_feature = []
        if self.token_merge_ratio == -1:
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.visual_encoder.config.image_size \
                                     // self.visual_encoder.config.patch_size
                    assert height * width == base_image_feature.shape[0]
                    if self.image_aspect_ratio == 'anyres':
                        num_patch = get_anyres_image_grid_shape(
                            orig_sizes[image_idx], self.image_grid_pinpoints,
                            self.visual_encoder.config.image_size)
                        num_patch_width, num_patch_height = num_patch
                        image_feature = image_feature.view(num_patch_height,
                                                           num_patch_width, height,
                                                           width, -1)
                    else:
                        raise NotImplementedError

                    if 'unpad' in self.mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1,
                                                              3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature,
                                                    orig_sizes[image_idx])
                        image_feature = torch.cat(
                            (image_feature,
                             self.image_newline[:, None, None].expand(
                                 *image_feature.shape[:-1], 1)),
                            dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3,
                                                              4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat((base_image_feature, image_feature),
                                              dim=0)
                else:
                    image_feature = image_feature[0]
                    if 'unpad' in self.mm_patch_merge_type:
                        image_feature = torch.cat(
                            (image_feature, self.image_newline[None]), dim=0)
                new_image_feature.append(image_feature)
        else:
            # 由于进行了 token merge，unpad 操作不好弄，暂时不支持
            new_image_feature = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    # 182, d
                    base_image_feature = image_feature[0]
                    # 183,d
                    base_image_feature = torch.cat(
                        (base_image_feature, self.image_newline[None]), dim=0)

                    # n, 182, d
                    image_feature = image_feature[1:]

                    # n,182+1, d
                    image_feature = torch.cat(
                        (image_feature,
                         self.image_newline[None, None].expand(
                             image_feature.shape[0], 1, image_feature.shape[-1])),
                        dim=1)

                    # n*183,d
                    image_feature = image_feature.flatten(0, 1)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    new_image_feature.append(image_feature)
                else:
                    # 182, d
                    image_feature = image_feature[0]
                    # 183,d
                    image_feature = torch.cat(
                        (image_feature, self.image_newline[None]), dim=0)

                    new_image_feature.append(image_feature)
        return new_image_feature
