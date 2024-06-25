import argparse
import os.path as osp

import torch
from mmengine.config import Config
from transformers import AutoTokenizer

from xtuner.model.utils import LoadWoInit
from xtuner.registry import BUILDER


def convert_to_official(config, trained_path, save_path):
    cfg = Config.fromfile(config)
    cfg.model.pretrained_pth = trained_path
    cfg.model.quantization_vit = False
    cfg.model.quantization_llm = False

    with LoadWoInit():
        model = BUILDER.build(cfg.model)
    model.to(torch.bfloat16)

    if model.use_visual_encoder_lora:
        vision_model = model.model.vision_model.merge_and_unload()
        model.model.vision_model = vision_model

    if model.use_llm_lora:
        language_model = model.model.language_model.merge_and_unload()
        model.model.language_model = language_model

    model.model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)

    print(model)


def main():
    parser = argparse.ArgumentParser(
        description='Convert the pth model to HuggingFace model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('trained_model_pth', help='The trained model path.')
    parser.add_argument(
        'save_path', help='The path to save the converted model.')
    args = parser.parse_args()

    if osp.realpath(args.trained_model_pth) == osp.realpath(args.save_path):
        raise ValueError(
            'The trained path and save path should not be the same.')

    convert_to_official(args.config, args.trained_model_pth, args.save_path)


if __name__ == '__main__':
    main()
