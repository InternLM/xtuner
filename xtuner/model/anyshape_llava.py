# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn

from ..dataset.utils import get_anyres_image_grid_shape, unpad_image
from .llava import LLaVAModel
from .utils import (get_peft_model_state_dict,
                    prepare_inputs_labels_for_multimodal, truncate_dict)


class AnyShapeLLaVAModel(LLaVAModel):

    def __init__(self, image_grid_pinpoints, *args, max_length=4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_newline = nn.Parameter(
            torch.randn(
                self.llm.config.hidden_size, dtype=self.visual_encoder.dtype))
        self.image_grid_pinpoints = image_grid_pinpoints
        self.mm_patch_merge_type = 'spatial_unpad'
        self.image_aspect_ratio = 'anyres'
        self.max_length = max_length

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

    def preprocess_for_pixel_values(self, data, data_samples=None):
        if 'pixel_values' not in data:
            raise NotImplementedError()

        orig_sizes = data['orig_sizes']
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

        visual_outputs = self.visual_encoder(
            concat_images, output_hidden_states=True)
        image_features = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

        split_sizes = [image.shape[0] for image in pixel_values]
        image_features = torch.split(image_features, split_sizes, dim=0)

        new_image_feature = []
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
        return new_image_feature

    def forward(self, data, data_samples=None, mode='loss'):
        if 'pixel_values' in data:
            new_image_feature = self.preprocess_for_pixel_values(
                data, data_samples)
            data['pixel_values'] = new_image_feature
            del data['orig_sizes']
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

            inputs_embeds = data['inputs_embeds']
            if inputs_embeds is not None:
                seq_len = inputs_embeds.shape[1]
            else:
                seq_len = data['input_ids'].shape[1]

            if seq_len > self.max_length:
                warnings.warn(
                    f'Input length {seq_len} is longer than the '
                    f'maximum length {self.max_length}. Truncating the input.')
                data = truncate_dict(data, self.max_length)

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError
