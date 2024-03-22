import base64
import os
from collections import OrderedDict
from io import BytesIO
from typing import List, Literal, Optional, Union

import requests
import torch
from accelerate import load_checkpoint_in_model
from peft import LoraConfig, PeftModel
from PIL import Image
from torch import nn
from transformers import AutoModel, CLIPImageProcessor, CLIPVisionModel

from xtuner.dataset.utils import expand2square
from xtuner.utils.config import build_from_cfg_or_obj
from ..modules import ProjectorConfig, ProjectorModel
from ..utils import (LoadWoInit, get_peft_model_state_dict,
                     prepare_for_vision_lora)
from .base import BaseEncoder, _ImageType


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_url: str) -> Image.Image:
    """load image from url, local path or openai GPT4V."""

    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    if image_url.startswith('http'):
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()

        # Open the image using PIL
        img = Image.open(BytesIO(response.content))
    elif image_url.startswith('data:image'):
        img = load_image_from_base64(image_url.split(',')[1])
    else:
        img = Image.open(image_url)

    return img


class LlavaEncoderWrapper(BaseEncoder):

    def __init__(self,
                 model_name_or_path: str,
                 lora=None,
                 select_layer: int = -2,
                 freeze_clip: bool = True):

        super().__init__()

        assert not (lora is not None and freeze_clip)
        self._projector = None
        self.proj_inited = False
        self.freeze_clip = freeze_clip
        self.select_layer = select_layer

        _res = self.build_processor_and_encoder(model_name_or_path)
        self._processor, self._encoder = _res

        if self.freeze_clip:
            self._encoder.requires_grad_(False)

        if lora:
            self.with_lora = True
            lora_conf = build_from_cfg_or_obj(lora, accept=LoraConfig)
            self._encoder = prepare_for_vision_lora(self._encoder, lora_conf)
        else:
            self.with_lora = False

    def post_init_proj(self, config: ProjectorConfig):
        self._projector = ProjectorModel(config)
        self.proj_inited = True

    def build_processor_and_encoder(self, model_name_or_path: str):
        with LoadWoInit:
            processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
            encoder = CLIPVisionModel.from_pretrained(
                model_name_or_path, torch_dtype=torch.float16)
        return processor, encoder

    @classmethod
    def only_build_processor(self, model_name_or_path: str):
        return CLIPImageProcessor.from_pretrained(model_name_or_path)

    @property
    def encoder(self) -> CLIPVisionModel:
        return self._encoder

    @property
    def processor(self):
        return self._processor

    @property
    def projector(self) -> ProjectorModel:
        if self._projector:
            return self._projector
        else:
            raise RuntimeError('The projector has not been created yet, '
                               'please execute `post_init_proj` first.')

    def gradient_checkpointing_enable(self):
        # For backward compatibility
        if hasattr(self.encoder, 'enable_input_require_grads'):
            self.encoder.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.encoder.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        self.encoder.gradient_checkpointing_enable()

        self.projector.enable_input_require_grads()
        self.projector.gradient_checkpointing_enable()

    def preprocess(self, image: _ImageType) -> List[torch.Tensor]:
        """Preprocess the input image, including expanding to square and
        normalization.

        Args:
            image (Image.Image): The input image need to be preprocessed.
        Returns:
            torch.Tensor: The preprocessed image tensor.
        """

        if isinstance(image, str):
            image = load_image(image)

        if not isinstance(image, Image.Image):
            raise TypeError(f"Don't support {type(image).__name__}, "
                            'the image type must be `PIL.Image`.')

        processor = self.processor
        image_mean = processor.image_mean

        background_color = tuple(int(x * 255) for x in image_mean)
        squared_img = expand2square(image, background_color)

        processed = processor.preprocess(squared_img, return_tensors='pt')
        img_tensor = processed['pixel_values'][0]  # shape: 3, h, w

        # before this line, `img_tensor` is on cpu.
        img_tensor = img_tensor.to(self.device).to(self.dtype)
        return img_tensor

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        embeddings = self.projector(
            outputs.hidden_states[self.select_layer][:, 1:])
        return embeddings

    @torch.no_grad()
    def batch_infer(self, images: List[_ImageType]) -> List[torch.Tensor]:
        """Obtain the corresponding embeddings based on the images.

        Args:
            images (List[Image.Image]): The input images. The data layout
                for each image is (c, h, w).
        Returns:
            List[torch.Tensor]: The list of extracted features from images.
                The data layout for each tensor should be (tokens, dims).
        """

        num_imgs = len(images)

        img_tensors = [self.process_img(img) for img in images]

        # Determine if all image sizes are consistent.
        # TODO (pppppM): Confirm when the image size will be inconsistent
        shape_consistant = all(x.shape == img_tensors[0].shape
                               for x in img_tensors)

        from transformers.modeling_outputs import BaseModelOutputWithPooling

        if shape_consistant:
            # Batch inference when all image sizes are consistent.
            # img_tensors[0] shape: (3, h, w)
            # tensor shape: (num_imgs, 3, h, w)
            tensor = torch.stack(img_tensors, dim=0)

            enc_out = self.visual_encoder(tensor, output_hidden_states=True)
            enc_out: BaseModelOutputWithPooling

            # feat shape: (num_imgs, tokens, dims)
            feat = self.projector(enc_out.hidden_states[self.select_layer][:,
                                                                           1:])

            # Split along the batch dimension
            # The feature of each image corresponds to a tensor.
            # len(features): num_imgs, features[0] shape:(1, tokens, dims)
            features = torch.chunk(feat, num_imgs, dim=0)

            # per image feature's layout should be (tokens, dims)
            features = [x.flatten(0, 1) for x in features]

        else:
            features = []
            for tensor in img_tensors:
                tensor: torch.Tensor
                # The visual encoder requires a data layout of (bs, c, h, w).
                # tensor shape: (3, h, w)   batch_tensor shape: (1, 3, h, w)
                batch_tensor = tensor.unsqueeze(0)
                enc_out = self.visual_encoder(
                    batch_tensor, output_hidden_states=True)
                enc_out: BaseModelOutputWithPooling
                # feat shape: (1, tokens, dims)
                feat = self.projector(
                    enc_out.hidden_states[self.select_layer][:, 1:])
                features.append(feat)

        return features

    def save_checkpoint(self, dir: str):

        if self.with_lora:
            _save_dir = os.path.join(dir, 'visual_encoder_adapter')
            self.encoder.save_pretrained(_save_dir, safe_serialization=False)

        if not self.freeze_clip:
            _save_dir = os.path.join(dir, 'visual_encoder')
            self.encoder.save_pretrained(_save_dir, safe_serialization=False)
            self.processor.save_pretrained(_save_dir)

        _save_dir = os.path.join(dir, 'projector')
        self.projector.save_pretrained(_save_dir)

    def load_checkpoint(self, dir):

        if self.with_lora:
            _ckpt_dir = os.path.join(dir, 'visual_encoder_adapter')
            self.encoder.load_adapter(_ckpt_dir)

        if not self.freeze_clip:
            _ckpt_dir = os.path.join(dir, 'visual_encoder')
            load_checkpoint_in_model(self.encoder, _ckpt_dir)
            load_checkpoint_in_model(self.processor, _ckpt_dir)

        if self.proj_inited:
            _ckpt_dir = os.path.join(dir, 'projector')
            load_checkpoint_in_model(self.projector, _ckpt_dir)
        else:
            ProjectorModel.from_pretrained(_ckpt_dir)

    def state_dict(self, *args, **kwargs):

        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. encoder
        if self.with_lora:
            to_return.update(
                get_peft_model_state_dict(self.encoder, state_dict=state_dict))
        elif not self.freeze_clip:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if '_encoder.' in k})

        # Step 2. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if '_projector.' in k})

        return to_return


# if __name__ == '__main__':
#     img = load_image('llava.jpeg')
#     model = VisionEncoderForDeploy('xtuner/llava-internlm-7b',
#                                    'openai/clip-vit-large-patch14-336')

#     model.cuda()
#     model.eval()
#     outputs = model([img])
