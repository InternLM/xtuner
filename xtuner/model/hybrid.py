# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
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


class HybridFinetune(BaseModel):

    def __init__(
        self,
        llm,
        visual_encoder=None,
        visual_select_layer=-2,
        projector_depth=2,
        pretrained_pth=None,
        tokenizer=None,
        llm_lora=None,
        visual_encoder_lora=None,
        freeze_llm=False,
        freeze_visual_encoder=False,
        use_activation_checkpointing=True,
        use_varlen_attn=False,
    ):
        super().__init__()

        # Build the base language model without initialization.
        # This will greatly reduce the time to build the model.
        with LoadWoInit():
            self.llm = build_from_cfg_or_obj(llm, nn.Module)
            if visual_encoder:
                visual_encoder = build_from_cfg_or_obj(visual_encoder,
                                                       nn.Module)
            self.visual_encoder = visual_encoder
            self.visual_select_layer = visual_select_layer
        self.llm.config.use_cache = False
        dispatch_modules(self.llm, use_varlen_attn=use_varlen_attn)

        if tokenizer is not None:
            if isinstance(tokenizer, dict):
                tokenizer = BUILDER.build(tokenizer)
            smart_tokenizer_and_embedding_resize(tokenizer, self.llm)

        projector_config = ProjectorConfig(
            visual_hidden_size=self.visual_encoder.config.hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth)
        self.projector = ProjectorModel(projector_config).to(
            self.visual_encoder.dtype)

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            enable_hf_model_gradient_checkpointing(self.llm)
            enable_hf_model_gradient_checkpointing(self.visual_encoder)

            self.projector.enable_input_require_grads()
            self.projector.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        # Prepare the model for LoRA if specified
        if self.use_llm_lora:
            lora_conf = build_from_cfg_or_obj(llm_lora, accept=LoraConfig)
            self.llm = prepare_for_llm_lora(self.llm, lora_conf,
                                            use_activation_checkpointing)

        if self.use_visual_encoder_lora:
            lora_conf = build_from_cfg_or_obj(
                visual_encoder_lora, accept=LoraConfig)
            self.visual_encoder = prepare_for_vision_lora(
                self.visual_encoder, lora_conf, use_activation_checkpointing)
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
            return self.compute_loss(data, data_samples)
        else:
            raise NotImplementedError(
                f"{type(self)}'s forward is only supported for use during "
                'training. If you want to get predictions or chat, please '
                "directly use `llm`'s forward.")

    def compute_loss(self, data, data_samples=None):

        input_ids = data['input_ids']
        labels = data['labels']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        pixel_values = data['pixel_values']
        img_rngs = data['image_ranges']
        img_belong = data['image_belong']

        input_embeds = self.llm.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            visual_outputs = self.visual_encoder(
                pixel_values, output_hidden_states=True)
            img_embeds = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

            empty_embs = torch.zeros_like(input_embeds)
            for emb, rng, b_id in zip(img_embeds, img_rngs, img_belong):
                left, right = rng
                if emb.size(0) == right - left:
                    empty_embs[b_id, left:right, :] = emb
                elif not emb.size(0) == right - left and left == 0:
                    empty_embs[b_id, left:right, :] = emb[-right:]
                elif not emb.size(
                        0) == right - left and right == empty_embs.size(1):
                    empty_embs[b_id, left:right, :] = emb[:right - left]
                else:
                    breakpoint()

            non_img_mask = (empty_embs == 0)
            input_embeds = input_embeds * non_img_mask + empty_embs

        outputs = self.llm(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            labels=labels)

        loss_dict = {'loss': outputs.loss}
        return loss_dict

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

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
