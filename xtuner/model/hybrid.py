# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
from mmengine.model import BaseModel
from peft import LoraConfig
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from xtuner.registry import BUILDER
from xtuner.types import HybridChatMessages, HybridChatTemplate
from xtuner.utils.config import build_from_cfg_or_obj
from .base import BaseTune
from .encoders import EncoderWrapper
from .modules import ProjectorConfig, dispatch_modules
from .utils import (LoadWoInit, get_peft_model_state_dict,
                    prepare_for_llm_lora, smart_tokenizer_and_embedding_resize)


class HybridFinetune(BaseTune):

    def __init__(
        self,
        llm: Union[PreTrainedModel, Dict],
        tokenizer: Union[PreTrainedTokenizer, Dict],
        chat_template: HybridChatTemplate,
        visual_encoder: Optional[Union[EncoderWrapper, Dict]] = None,
        audio_encoder: Optional[Union[EncoderWrapper, Dict]] = None,
        video_encoder: Optional[Union[EncoderWrapper, Dict]] = None,
        proj_depth: int = 2,
        llm_lora: Optional[Union[LoraConfig, Dict]] = None,
        freeze_llm: bool = False,
        use_gradient_checkpointing: bool = True,
        use_varlen_attn: bool = False,
    ):
        super().__init__()

        tokenizer = build_from_cfg_or_obj(
            tokenizer, accept=PreTrainedTokenizer)
        smart_tokenizer_and_embedding_resize(tokenizer, self.llm)
        self._tokenizer: PreTrainedModel = tokenizer

        self._chat_template = chat_template

        # Build the base language model without initialization.
        # This will greatly reduce the time to build the model.
        with LoadWoInit():
            self._llm: PreTrainedModel = build_from_cfg_or_obj(llm, nn.Module)
            self._llm.config.use_cache = False

        self.freeze_llm = freeze_llm
        if self.freeze_llm:
            self.llm.requires_grad_(False)

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

        if visual_encoder:
            visual_encoder = build_from_cfg_or_obj(visual_encoder,
                                                   EncoderWrapper)
            self.visual_encoder: EncoderWrapper = visual_encoder
            _proj_config = ProjectorConfig(
                visual_hidden_size=self.visual_encoder.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=proj_depth)

            self.visual_encoder.post_init_proj(_proj_config)
        else:
            self.visual_encoder = None

        if audio_encoder:
            audio_encoder = build_from_cfg_or_obj(audio_encoder,
                                                  EncoderWrapper)
            self.audio_encoder: EncoderWrapper = audio_encoder
            _proj_config = ProjectorConfig(
                visual_hidden_size=self.audio_encoder.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=proj_depth)

            self.audio_encoder.post_init_proj(_proj_config)
        else:
            self.audio_encoder = None

        if video_encoder:
            video_encoder = build_from_cfg_or_obj(video_encoder,
                                                  EncoderWrapper)
            self.video_encoder: EncoderWrapper = video_encoder
            _proj_config = ProjectorConfig(
                visual_hidden_size=self.video_encoder.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=proj_depth)

            self.video_encoder.post_init_proj(_proj_config)
        else:
            self.video_encoder = None

        if use_gradient_checkpointing:
            self.gradient_checkpointing_enable()

        self.avoid_override_weights()

    @property
    def llm(self) -> PreTrainedModel:
        return self._llm

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    def chat_template(self) -> HybridChatTemplate:
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
        self.visual_encoder.gradient_checkpointing_enable()

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

            features = self.visual_encoder(pixel_values)
            batch_total_imgs, real_img_tokens, _ = features.shape

            for i in range(batch_total_imgs):
                img_start, img_end = img_rngs[i]
                exp_img_tokens = img_end - img_start
                img_emb = features[i]
                img_bs_ind = img_belongs[i]

                # pack 导致的截断
                if real_img_tokens == exp_img_tokens:
                    img_embeds.append(img_emb)
                elif real_img_tokens != exp_img_tokens and img_start == 0:
                    img_embeds.append(img_emb[real_img_tokens - img_end:])
                elif (real_img_tokens != exp_img_tokens and img_end == tokens):
                    img_embeds.append(img_emb[:exp_img_tokens])
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

        _empty_embeds = torch.zeros_like(flat_embeds)
        for (start, end), emb in zip(ranges, mm_embeds):
            _empty_embeds[start:end] += emb

        flat_embeds = flat_embeds * (_empty_embeds == 0)

        return flat_embeds + _empty_embeds

    def _compute_postion_ids(self, data):
        input_ids = data['input_ids']
        bs, tokens = input_ids.shape
        if self.use_varlen_attn:
            assert bs == 1

            cumulative_len = data['cumulative_len'][0]
            max_seqlen = (cumulative_len[1:] - cumulative_len[:-1]).max()

            position_ids = []
            for i in range(1, len(cumulative_len)):
                chunk_tokens = cumulative_len[i] - cumulative_len[i - 1]
                position_ids.append(torch.arange(chunk_tokens))
            position_ids = torch.cat(position_ids, dim=0).unsqueeze(0)

            from mmengine import MessageHub
            rank = dist.get_rank()
            message_hub = MessageHub.get_instance('varlen_attn_args')
            message_hub.update_info(f'cumulative_len_rank_{rank}',
                                    cumulative_len)
            message_hub.update_info(f'max_seqlen_rank_{rank}', max_seqlen)
        else:

            position_ids = torch.arange(0, tokens).unsqueeze(0).repeat(bs, 1)

    def compute_loss(self, data):

        input_ids = data['input_ids']
        labels = data['labels']
        attention_mask = data['attention_mask']

        bs, tokens = input_ids.shape
        position_ids = self._compute_postion_ids(data)

        input_embeds = self.llm.get_input_embeddings()(input_ids)

        bs, tokens, dim = input_embeds.shape
        flat_embeds = input_embeds.flatten(0, 1)

        img_embs, flat_bs_img_rngs = self._get_vision_embeds_and_ranges(data)
        # audio_embs, flat_bs_img_rngs = self._get_vision_embeds_and_ranges(data)
        # video_embs, flat_bs_img_rngs = self._get_vision_embeds_and_ranges(data)
        flat_embeds = self._insert_mm_embeddings(flat_embeds, img_embs,
                                                 flat_bs_img_rngs)
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

        # Step 1. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if '_llm.' in k})

        # Step 2. Visual Encoder
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'visual_encoder.' in k})
        return to_return

    def chat(self, messages: HybridChatMessages, sample_params, streamer):

        prompt = messages.apply_chat_template(self.chat_template)
