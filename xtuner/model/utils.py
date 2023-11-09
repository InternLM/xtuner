# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List, Optional

import torch
from mmengine import print_log
from mmengine.utils.misc import get_object_from_string
from peft import PeftType
from torch import nn
from transformers import PreTrainedModel

from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def set_obj_dtype(d):
    for key, value in d.items():
        if value in ['torch.float16', 'torch.float32', 'torch.bfloat16']:
            d[key] = getattr(torch, value.split('.')[-1])


def traverse_dict(d):
    if isinstance(d, dict):
        set_obj_dtype(d)
        for key, value in d.items():
            if isinstance(value, dict):
                traverse_dict(value)
                if 'type' in value:
                    builder = value.pop('type')
                    if isinstance(builder, str):
                        builder = get_object_from_string(builder)
                    new_value = builder(**value)
                    d[key] = new_value
                    print_log(f'{key} convert to {builder}')
    elif isinstance(d, list):
        for element in d:
            traverse_dict(element)


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('output_layer')
    return list(lora_module_names)


class LoadWoInit:
    """Context manager that disable parameter initialization."""

    def __init__(self):
        self.constant_ = torch.nn.init.constant_
        self.zeros_ = torch.nn.init.zeros_
        self.ones_ = torch.nn.init.ones_
        self.uniform_ = torch.nn.init.uniform_
        self.normal_ = torch.nn.init.normal_
        self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        self.kaiming_normal_ = torch.nn.init.kaiming_normal_

    def __enter__(self, *args, **kwargs):
        torch.nn.init.constant_ = lambda *args, **kwargs: None
        torch.nn.init.zeros_ = lambda *args, **kwargs: None
        torch.nn.init.ones_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None

    def __exit__(self, *args, **kwargs):
        torch.nn.init.constant_ = self.constant_
        torch.nn.init.zeros_ = self.zeros_
        torch.nn.init.ones_ = self.ones_
        torch.nn.init.uniform_ = self.uniform_
        torch.nn.init.normal_ = self.normal_
        torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
        torch.nn.init.kaiming_normal_ = self.kaiming_normal_


def get_peft_model_state_dict(model, state_dict=None, adapter_name='default'):
    # Modified from `https://github.com/huggingface/peft/blob/main/src
    # /peft/utils/save_and_load.py`

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type == PeftType.LORA:
        # adapted from `https://github.com/microsoft/LoRA/blob/main/
        # loralib/utils.py`
        # to be used directly with the state dict which is necessary
        # when using DeepSpeed or FSDP
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k}
        elif bias == 'all':
            to_return = {
                k: state_dict[k]
                for k in state_dict if 'lora_' in k or 'bias' in k
            }
        elif bias == 'lora_only':
            to_return = {}
            for k in state_dict:
                if 'lora_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('lora_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {
            k: v
            for k, v in to_return.items()
            if (('lora_' in k and adapter_name in k) or ('bias' in k))
        }
    else:
        # Currently we only support lora
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f'{module_name}.modules_to_save.{adapter_name}' in key
                   for module_name in model.modules_to_save):
                to_return[key] = value

    return to_return


# Modified from https://github.com/haotian-liu/LLaVA/blob/8467850a63aa0d6f47aa150c53aca4751f0d3d14/llava/model/llava_arch.py#L99  # noqa: E501
def prepare_inputs_labels_for_multimodal(
        llm: PreTrainedModel,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None):
    if pixel_values is None or input_ids.shape[1] == 1:
        if (past_key_values is not None and pixel_values is not None
                and input_ids.shape[1] == 1):
            attention_mask = torch.ones(
                (attention_mask.shape[0],
                 past_key_values[-1][-1].shape[-2] + 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        return input_ids, attention_mask, past_key_values, None, labels

    new_inputs_embeds = []
    new_labels = [] if labels is not None else None
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
            cur_inputs_embeds = llm.get_input_embeddings()(cur_input_ids)
            new_inputs_embeds.append(cur_inputs_embeds)
            if labels is not None:
                new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
        image_token_indices = torch.where(
            cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        cur_new_inputs_embeds = []
        if labels is not None:
            cur_labels = labels[batch_idx]
            cur_new_labels = []
            assert cur_labels.shape == cur_input_ids.shape
        while image_token_indices.numel() > 0:
            cur_pixel_values = pixel_values[cur_image_idx]
            image_token_start = image_token_indices[0]
            cur_new_inputs_embeds.append(llm.get_input_embeddings()(
                cur_input_ids[:image_token_start]))
            cur_new_inputs_embeds.append(cur_pixel_values)
            if labels is not None:
                cur_new_labels.append(cur_labels[:image_token_start])
                cur_new_labels.append(
                    torch.full((cur_pixel_values.shape[0], ),
                               IGNORE_INDEX,
                               device=labels.device,
                               dtype=labels.dtype))
                cur_labels = cur_labels[image_token_start + 1:]
            cur_image_idx += 1
            cur_input_ids = cur_input_ids[image_token_start + 1:]
            image_token_indices = torch.where(
                cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        if cur_input_ids.numel() > 0:
            cur_new_inputs_embeds.append(
                llm.get_input_embeddings()(cur_input_ids))
            if labels is not None:
                cur_new_labels.append(cur_labels)
        cur_new_inputs_embeds = [
            x.to(device=llm.device) for x in cur_new_inputs_embeds
        ]
        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds, dim=0)
        new_inputs_embeds.append(cur_new_inputs_embeds)
        if labels is not None:
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

    if any(x.shape != new_inputs_embeds[0].shape for x in new_inputs_embeds):
        max_len = max(x.shape[0] for x in new_inputs_embeds)

        new_inputs_embeds_align = []
        for cur_new_embed in new_inputs_embeds:
            cur_new_embed = torch.cat(
                (
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_new_embed.shape[0],
                         cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device,
                    ),
                ),
                dim=0,
            )
            new_inputs_embeds_align.append(cur_new_embed)
        new_inputs_embeds = torch.stack(new_inputs_embeds_align, dim=0)

        if labels is not None:
            new_labels_align = []
            _new_labels = new_labels
            for cur_new_label in new_labels:
                cur_new_label = torch.cat(
                    (
                        cur_new_label,
                        torch.full(
                            (max_len - cur_new_label.shape[0], ),
                            IGNORE_INDEX,
                            dtype=cur_new_label.dtype,
                            device=cur_new_label.device,
                        ),
                    ),
                    dim=0,
                )
                new_labels_align.append(cur_new_label)
            new_labels = torch.stack(new_labels_align, dim=0)

        if attention_mask is not None:
            new_attention_mask = []
            for (cur_attention_mask, cur_new_labels,
                 cur_new_labels_align) in zip(attention_mask, _new_labels,
                                              new_labels):
                new_attn_mask_pad_left = torch.full(
                    (cur_new_labels.shape[0] - labels.shape[1], ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                new_attn_mask_pad_right = torch.full(
                    (cur_new_labels_align.shape[0] -
                     cur_new_labels.shape[0], ),
                    False,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                cur_new_attention_mask = torch.cat(
                    (new_attn_mask_pad_left, cur_attention_mask,
                     new_attn_mask_pad_right),
                    dim=0)
                new_attention_mask.append(cur_new_attention_mask)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            assert attention_mask.shape == new_labels.shape
    else:
        new_inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)

        if attention_mask is not None:
            new_attn_mask_pad_left = torch.full(
                (attention_mask.shape[0],
                 new_inputs_embeds.shape[1] - input_ids.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat(
                (new_attn_mask_pad_left, attention_mask), dim=1)
            assert attention_mask.shape == new_inputs_embeds.shape[:2]

    return {
        'input_ids': None,
        'attention_mask': attention_mask,
        'past_key_values': past_key_values,
        'inputs_embeds': new_inputs_embeds,
        'labels': new_labels
    }


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


def guess_load_checkpoint(pth_model):
    if os.path.isfile(pth_model):
        state_dict = torch.load(pth_model, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    elif os.path.isdir(pth_model):
        try:
            from deepspeed.utils.zero_to_fp32 import \
                get_fp32_state_dict_from_zero_checkpoint
        except ImportError:
            raise ImportError(
                'The provided PTH model appears to be a DeepSpeed checkpoint. '
                'However, DeepSpeed library is not detected in current '
                'environment. This suggests that DeepSpeed may not be '
                'installed or is incorrectly configured. Please verify your '
                'setup.')
        state_dict = get_fp32_state_dict_from_zero_checkpoint(
            os.path.dirname(pth_model), os.path.basename(pth_model))
    else:
        raise FileNotFoundError(f'Cannot find {pth_model}')
    return state_dict
