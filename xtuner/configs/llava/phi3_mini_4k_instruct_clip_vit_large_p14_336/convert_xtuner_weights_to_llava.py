# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch

try:
    from llava.model import LlavaConfig, LlavaLlamaForCausalLM
    from llava.utils import disable_torch_init
except ImportError:
    raise ImportError(
        'Please install llava with '
        '`pip install git+https://github.com/haotian-liu/LLaVA.git '
        '--no-deps`.')
from safetensors import safe_open
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel)

KEYS_TO_MODIFY_MAPPING_VIT = {
    'vision_model': 'model.vision_tower.vision_tower.vision_model',
}
KEYS_TO_MODIFY_MAPPING_PROJECTOR = {
    'model.0': 'model.mm_projector.0',
    'model.2': 'model.mm_projector.2',
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


def convert_to_llava(text_model_id, vision_model_id, projector_weight,
                     save_path):
    disable_torch_init()
    torch.set_default_dtype(torch.float16)

    projector_state_dict = {}
    with safe_open(projector_weight, framework='pt', device='cpu') as f:
        for key in f.keys():
            projector_state_dict[key] = f.get_tensor(key)

    ori_llm = AutoModelForCausalLM.from_pretrained(
        text_model_id, trust_remote_code=True, device_map='auto')
    ori_vit = CLIPVisionModel.from_pretrained(vision_model_id)
    llm_state_dict = ori_llm.state_dict()
    vit_state_dict = ori_vit.state_dict()

    projector_state_dict = convert_state_dict_to_hf(
        projector_state_dict, KEYS_TO_MODIFY_MAPPING_PROJECTOR)
    vit_state_dict = convert_state_dict_to_hf(vit_state_dict,
                                              KEYS_TO_MODIFY_MAPPING_VIT)
    state_dict = {**projector_state_dict, **llm_state_dict, **vit_state_dict}

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_config = AutoConfig.from_pretrained(
        text_model_id, trust_remote_code=True)

    ori_config = text_config.__dict__.copy()
    ori_config.update(
        dict(
            image_aspect_ratio='pad',
            mm_hidden_size=ori_vit.config.hidden_size,
            mm_projector_type='mlp2x_gelu',
            mm_use_im_patch_token=False,
            mm_use_im_start_end=False,
            mm_vision_select_feature='patch',
            mm_vision_select_layer=-2,
            mm_vision_tower=vision_model_id,
            unfreeze_mm_vision_tower=True,
            model_type='llava',
            use_cache=True,
            use_mm_proj=True))
    config = LlavaConfig(**ori_config)

    with torch.device('meta'):
        model = LlavaLlamaForCausalLM(config)

    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)

    model.load_state_dict(state_dict, strict=True, assign=True)
    model.save_pretrained(save_path, max_shard_size='2GB')
    image_processor.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f'Saved to {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model_id')
    parser.add_argument('--vision_model_id')
    parser.add_argument('--projector_weight')
    parser.add_argument('--save_path')
    args = parser.parse_args()
    convert_to_llava(args.text_model_id, args.vision_model_id,
                     args.projector_weight, args.save_path)


if __name__ == '__main__':
    main()
