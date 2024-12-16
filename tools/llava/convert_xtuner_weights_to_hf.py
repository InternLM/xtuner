# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/llava/convert_llava_weights_to_hf.py  # noqa: E501
import argparse

from transformers import (CLIPImageProcessor, CLIPVisionModel)
from xtuner._lite.modelings.llava import LlavaForConditionalGeneration, EnhancedLlavaConfig, LlavaProcessor
from mmengine import Config
from xtuner.registry import BUILDER
from mmengine import print_log
from xtuner._lite.parallel.fsdp import LoadWoInit



LLM_PREFIX = 'language_model'
VIT_PREFIX = 'vision_tower'
PROJECTOR_MAPPING = {
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

def add_prefix(state_dict,prefix):
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[f'{prefix}.{key}'] = value
        
    return new_state_dict
        
def convert_to_hf(cfg, save_dir):
    
    print_log('Loading XTuner Checkpoint...', 'current')
    model = BUILDER.build(cfg.model)
    # get state_dict
    llm = model.llm
    if model.use_llm_lora:
        llm = model.llm.merge_and_unload()
    llm.config.use_cache = True
    

    llm_state_dict = llm.state_dict()
    llm_state_dict = add_prefix(llm_state_dict, LLM_PREFIX)

    
    visual_encoder = model.visual_encoder
    if model.use_visual_encoder_lora:
        visual_encoder = model.visual_encoder.merge_and_unload()
    assert isinstance(visual_encoder, CLIPVisionModel),\
        'This conversion format only supports CLIPVisionModel.'
    
    visual_encoder_state_dict = visual_encoder.state_dict()
    visual_encoder_state_dict = add_prefix(
        visual_encoder_state_dict, VIT_PREFIX)
    
    projector_state_dict = model.projector.state_dict()
    projector_state_dict = convert_state_dict_to_hf(
        projector_state_dict, PROJECTOR_MAPPING)

    state_dict = {
        **projector_state_dict,
        **llm_state_dict,
        **visual_encoder_state_dict
    }
    
    tokenizer = BUILDER.build(cfg.tokenizer)

    # init model
    text_config = llm.config
    vision_config = visual_encoder.config
    
    img_token = '<image>'
    need_resize = False
    if len(tokenizer.encode(img_token,  add_special_tokens=False)) > 1:
        tokenizer.add_tokens([img_token], special_tokens=True)
        img_token_id = tokenizer.convert_tokens_to_ids([img_token])[0]
        
        print_log(f'[Tokenizer] Added a new token `{img_token}`, '
                    f'token id is {img_token_id}, the new vocab size is '
                    f'{len(tokenizer)}', 'current')
        
        llm_vocab_size = text_config.vocab_size
        if llm_vocab_size < len(tokenizer):
            # We add an image token so we need to resize the model
            need_resize = True
    else:
        img_token_id = tokenizer.convert_tokens_to_ids([img_token])[0]
    
    print_log('Building an empty HF Llava...', 'current')
    config = EnhancedLlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=img_token_id,
        attn_implementation='eager')

    with LoadWoInit():
        llava = LlavaForConditionalGeneration(config)
    
    print_log('Loading HF format state dict...', 'current')
    llava.load_state_dict(state_dict, strict=True, assign=True)
    
    if need_resize:
        ori_emb_shape = llava.get_input_embeddings().weight.shape
        llava.resize_token_embeddings(len(tokenizer))
        new_emb_shape = llava.get_input_embeddings().weight.shape
        print_log('Pad the parameters of `embbedings` and `output` from '
                        f'shape {ori_emb_shape} to shape {new_emb_shape}',
                  'current')
    

    # processor
    image_processor = BUILDER.build(cfg.image_processor)
    assert isinstance(image_processor, CLIPImageProcessor),\
        'This conversion format only supports CLIPImageProcessor.'

    processor = LlavaProcessor(
        tokenizer=tokenizer, image_processor=image_processor)
    
    # save
    print_log('Saving HF Llava...', 'current')
    llava.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('save_dir')
    args = parser.parse_args()
    
    cfg = Config.fromfile(args.config)
    convert_to_hf(cfg, args.save_dir)


if __name__ == '__main__':
    main()
