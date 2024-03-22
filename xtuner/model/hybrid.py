# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
from mmengine.model import BaseModel
from peft import LoraConfig
from mmengine import print_log
from torch import nn
import math
from xtuner.registry import BUILDER
from xtuner.utils.config import build_from_cfg_or_obj
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .utils import (LoadWoInit, enable_hf_model_gradient_checkpointing,
                    get_peft_model_state_dict, prepare_for_llm_lora,
                    prepare_for_vision_lora,
                    smart_tokenizer_and_embedding_resize)
import torch.distributed as dist
from mmengine import runner
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
            return self.compute_loss(data)
        else:
            raise NotImplementedError(
                f"{type(self)}'s forward is only supported for use during "
                'training. If you want to get predictions or chat, please '
                "directly use `llm`'s forward.")

    
    
    def _get_vision_embeds_and_ranges(self, data):
        
        input_ids = data['input_ids']
        pixel_values = data['pixel_values']
        img_rngs = data['image_ranges']
        img_belongs = data['image_belongs']
        
        bs, tokens = input_ids.shape
        
        img_embeds = []
        ranges_in_flat_batch = []

        if pixel_values is not None:
            assert isinstance(pixel_values, torch.Tensor)
            assert len(img_rngs) == len(img_belongs) == pixel_values.size(0)
                
            batch_total_imgs = len(img_rngs)
            
            visual_outputs = self.visual_encoder(
                pixel_values, output_hidden_states=True)
            features = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
            batch_total_imgs, actual_img_tokens, _ = features.shape
            
            
            for i in range(batch_total_imgs):
                img_start, img_end = img_rngs[i]
                expect_img_tokens = img_end - img_start
                img_emb = features[i]
                img_bs_ind = img_belongs[i]
                
                if actual_img_tokens == expect_img_tokens:
                    img_embeds.append(img_emb)
                elif not actual_img_tokens == expect_img_tokens and img_start == 0:
                    img_embeds.append(img_emb[actual_img_tokens-img_end:])
                elif not actual_img_tokens == expect_img_tokens and img_end == tokens:
                    img_embeds.append(img_emb[:expect_img_tokens])
                else:
                    raise RuntimeError
                
                flat_offset = tokens * img_bs_ind
                
                left = flat_offset + img_start
                right = flat_offset + img_end
                ranges_in_flat_batch.append((left, right))
                
        return img_embeds, ranges_in_flat_batch
                
    
    def _insert_mm_embeddings(self, flat_embeds, mm_embeds, ranges):
        
        assert len(mm_embeds) == len(ranges)
        if len(mm_embeds) == 0:
            return flat_embeds
        
        chunk_embeds = []
        chunk_sizes = []
        mm_chunk_ids = []
        
        cursor = 0
        _empty_embeds = torch.zeros_like(flat_embeds)
        for (start, end), emb in zip(ranges, mm_embeds):
            _empty_embeds[start: end] += emb
            # if start - cursor > 0:
            #     chunk_sizes.append(start - cursor)
            #     cursor = start
        
            # mm_chunk_ids.append(len(chunk_sizes))
            
            
            # chunk_embeds.append(emb)
            # chunk_sizes.append(end - start)
            # cursor = end
        
        # tokens = flat_embeds.size(0)
        # if sum(chunk_sizes) < tokens :
        #     chunk_sizes.append(tokens - sum(chunk_sizes))
        
        # chunk_embs = list(torch.split(flat_embeds, chunk_sizes))
        # for ind, mm_emb in zip(mm_chunk_ids, mm_embeds) :
        #     chunk_embs[ind] = mm_emb
        
        # flat_embeds = torch.cat(chunk_embs, dim=0)
        flat_embeds = flat_embeds * (_empty_embeds == 0)
         
        return flat_embeds + _empty_embeds
    
    def compute_loss(self, data):

        input_ids = data['input_ids']
        labels = data['labels']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        
        bs, tokens, dim = input_embeds.shape
        flat_embeds = input_embeds.flatten(0,1)
        
        img_embs, flat_bs_img_rngs = self._get_vision_embeds_and_ranges(data)
        flat_embeds = self._insert_mm_embeddings(flat_embeds, img_embs, flat_bs_img_rngs)
        input_embeds = flat_embeds.reshape(bs, tokens, dim)

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
