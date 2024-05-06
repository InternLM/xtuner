# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn
from .utils import (get_peft_model_state_dict, guess_load_checkpoint,
                    prepare_inputs_labels_for_multimodal)
from .llava import LLaVAModel


class MiniGeminiModel(LLaVAModel):
    def __init__(self, *args, visual_encoder_aux=None, pretrained_pth=None, **kwargs):
        super().__init__(*args, pretrained_pth=None, **kwargs)
        self.visual_encoder_aux = self._build_from_cfg_or_module(visual_encoder_aux)

        if self.freeze_visual_encoder:
            self.visual_encoder_aux.requires_grad_(False)

        if self.use_activation_checkpointing:
            self.visual_encoder_aux.activation_checkpointing_enable()

        mm_hidden_size = self.visual_encoder.config.hidden_size
        mm_hidden_size_aux = self.visual_encoder_aux.hidden_size
        self.vlm_uni_query_projector = nn.Sequential(nn.LayerNorm(mm_hidden_size),
                                                     nn.Linear(mm_hidden_size, mm_hidden_size))
        self.vlm_uni_aux_projector = nn.Sequential(nn.LayerNorm(mm_hidden_size_aux),
                                                   nn.Linear(mm_hidden_size_aux,
                                                             mm_hidden_size))
        self.vlm_uni_val_projector = nn.Sequential(nn.LayerNorm(mm_hidden_size_aux),
                                                   nn.Linear(mm_hidden_size_aux,
                                                             mm_hidden_size))

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

    def activation_checkpointing_disable(self):
        super().activation_checkpointing_disable()
        if hasattr(self, 'visual_encoder_aux'):
            self.visual_encoder_aux.activation_checkpointing_disable()

    def activation_checkpointing_enable(self):
        super().activation_checkpointing_enable()
        if hasattr(self, 'visual_encoder_aux'):
            self.visual_encoder_aux.activation_checkpointing_enable()

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

        # Step 4. visual_encoder_aux
        if not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder_aux.' in k
            })
        # Step 5. unified projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'vlm_uni_' in k})
        return to_return

    def _prepare_data_for_llm(self, data):
        if 'pixel_values' in data:
            visual_outputs = self.visual_encoder(
                data['pixel_values'].to(self.visual_encoder.dtype),
                output_hidden_states=True)
            visual_outputs = visual_outputs.hidden_states[self.visual_select_layer]

            if self._get_model_class_name(self.visual_encoder) != 'SiglipVisionModel':
                visual_outputs = visual_outputs[:, 1:]

            visual_outputs_aux = torch.stack(data['pixel_values_aux'])
            visual_outputs_aux = self.visual_encoder_aux(
                visual_outputs_aux.to(self.visual_encoder_aux.dtype)
            )
            visual_outputs = self.unified_resampler(visual_outputs, visual_outputs_aux)

            pixel_values = self.projector(visual_outputs)
            data['pixel_values'] = pixel_values
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        return data

    def unified_resampler(self, images, images_aux):
        # patchwise with square images
        patch_num = int(images.shape[1] ** 0.5)  # 27
        # 216x216
        patch_size = images_aux.shape[-1] // patch_num  # 8
        # within patch attention
        images_aux = images_aux.permute(0, 2, 3, 1)
        images_aux = images_aux.reshape(len(images_aux), patch_num, patch_size, patch_num, patch_size,
                                        images_aux.shape[-1])
        images_aux = images_aux.permute(0, 1, 3, 2, 4, 5)
        images_aux = images_aux.reshape(len(images_aux), patch_num ** 2, patch_size ** 2,
                                        images_aux.shape[-1]).contiguous()

        # token attention
        embed_query = self.vlm_uni_query_projector(images)
        embed_aux = self.vlm_uni_aux_projector(images_aux)
        embed_value = self.vlm_uni_val_projector(images_aux)
        # TODO siglip+convnext 在第一次 forward 后正常，但是 embed_att 会出现 nan
        # TODO 导致第二次迭代时候 embed_value 会出现 nan，无法训练
        # TODO 怀疑是特征不匹配，即使全部转换为 fp32 也会出现 nan, 需要进一步排查
        embed_att = embed_query[:, :, None] @ (embed_aux.transpose(-1, -2) / (embed_aux.shape[-1] ** 0.5))
        # print('=xxxx=', torch.any(torch.isnan(embed_query)).item(),
        #       torch.any(torch.isnan(embed_aux)).item(),
        #       torch.any(torch.isnan(embed_value)).item(),
        #       torch.any(torch.isnan(embed_att)).item())
        embed_att = embed_att.nan_to_num()
        embed_feat = (embed_att.softmax(-1) @ embed_value).mean(2)
        # print('=xxcccxx=', torch.any(torch.isnan(embed_feat)).item())
        image_features = images + embed_feat
        return image_features
