# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/llava/convert_llava_weights_to_hf.py  # noqa: E501
import argparse

import torch
from safetensors import safe_open
from transformers import (AddedToken, AutoConfig, AutoModelForCausalLM,
                          CLIPImageProcessor, CLIPVisionModel,
                          LlamaTokenizerFast, LlavaConfig,
                          LlavaForConditionalGeneration, LlavaProcessor)

KEYS_TO_MODIFY_MAPPING_LLM = {
    'model': 'language_model.model',
    'lm_head': 'language_model.lm_head',
}
KEYS_TO_MODIFY_MAPPING_VIT = {
    'vision_model': 'vision_tower.vision_model',
}
KEYS_TO_MODIFY_MAPPING_PROJECTOR = {
    'model.0': 'multi_modal_projector.linear_1',
    'model.2': 'multi_modal_projector.linear_2',
}


def convert_state_dict_to_hf(state_dict, mapping):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('.inv_freq'):
            continue
        for key_to_modify, new_key in mapping.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict


def convert_to_hf(text_model_id, vision_model_id, projector_weight, save_path):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(
        text_model_id, trust_remote_code=True)
    vision_config = AutoConfig.from_pretrained(vision_model_id)
    if hasattr(vision_config, 'vision_config'):
        vision_config = vision_config.vision_config

    tokenizer = LlamaTokenizerFast.from_pretrained(text_model_id)
    tokenizer.add_tokens(
        AddedToken('<image>', special=True, normalized=False),
        special_tokens=True)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)

    processor = LlavaProcessor(
        tokenizer=tokenizer, image_processor=image_processor)

    config = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
        attn_implementation='eager')

    with torch.device('meta'):
        model = LlavaForConditionalGeneration(config)

    # Pad to 64 for performance reasons
    pad_shape = 64

    projector_state_dict = {}
    with safe_open(projector_weight, framework='pt', device='cpu') as f:
        for key in f.keys():
            projector_state_dict[key] = f.get_tensor(key)

    ori_llm = AutoModelForCausalLM.from_pretrained(
        text_model_id, trust_remote_code=True)
    ori_vit = CLIPVisionModel.from_pretrained(vision_model_id)

    llm_state_dict = ori_llm.state_dict()
    vit_state_dict = ori_vit.state_dict()

    projector_state_dict = convert_state_dict_to_hf(
        projector_state_dict, KEYS_TO_MODIFY_MAPPING_PROJECTOR)
    llm_state_dict = convert_state_dict_to_hf(llm_state_dict,
                                              KEYS_TO_MODIFY_MAPPING_LLM)
    vit_state_dict = convert_state_dict_to_hf(vit_state_dict,
                                              KEYS_TO_MODIFY_MAPPING_VIT)
    state_dict = {**projector_state_dict, **llm_state_dict, **vit_state_dict}
    model.load_state_dict(state_dict, strict=True, assign=True)

    pre_expansion_embeddings = \
        model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T
             @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    ori_vocab_size = config.text_config.vocab_size
    tokenizer_vocab_size = tokenizer.encode('<pad>')[-1]
    added_token = tokenizer_vocab_size - ori_vocab_size

    if added_token > 0:
        model.resize_token_embeddings(ori_vocab_size + added_token, pad_shape)
        model.language_model.model.embed_tokens.weight.data[
            ori_vocab_size:] = torch.stack(
                tuple(dist.sample()
                      for _ in range(model.language_model.model.embed_tokens.
                                     weight.data[ori_vocab_size:].shape[0])),
                dim=0,
            )
        model.language_model.lm_head.weight.data[
            ori_vocab_size:] = torch.stack(
                tuple(dist.sample()
                      for _ in range(model.language_model.lm_head.weight.
                                     data[ori_vocab_size:].shape[0])),
                dim=0,
            )

    model.config.image_token_index = tokenizer.encode('<image>')[-1]
    model.config.pad_token_id = tokenizer.encode('<pad>')[-1]

    if ori_vit.__class__.__name__ == 'SiglipVisionModel':
        model.config.vision_feature_select_strategy = 'full'

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f'Saved to {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model_id')
    parser.add_argument('--vision_model_id')
    parser.add_argument('--projector_weight')
    parser.add_argument('--save_path')
    args = parser.parse_args()
    convert_to_hf(args.text_model_id, args.vision_model_id,
                  args.projector_weight, args.save_path)


if __name__ == '__main__':
    main()
