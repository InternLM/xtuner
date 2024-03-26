import base64
import os
from io import BytesIO
from typing import List, Literal, Optional, Union

import requests
import torch
from peft import PeftModel
from PIL import Image
from torch import nn
from transformers import AutoModel, CLIPImageProcessor, CLIPVisionModel

from xtuner.dataset.utils import expand2square


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


ModelHub = Literal['huggingface', 'modelscope']


class VisionEncoderForDeploy(nn.Module):

    def __init__(self,
                 model_name_or_path: str,
                 projector_name_or_path: str,
                 adapter_name_or_path: str = None,
                 select_layer: int = -2,
                 hub: ModelHub = 'huggingface',
                 device='cuda'):

        super().__init__()

        # model_path = self._parse_model_path(xtuner_model_name_or_path, hub)
        # visual_encoder_path = self._parse_visual_encoder_path(
        #     model_path, visual_encoder_name_or_path, hub
        # )
        # projector_path = self._parse_projector_path(model_path)

        # # parse visual encoder adapter path.
        # vis_enc_adapter_path = self._parse_vis_enc_adapter_path(model_path)

        self.select_layer = select_layer
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_name_or_path)
        print(f'Load Image Processor From {model_name_or_path}')

        visual_encoder = CLIPVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16)
        print(f'Load Visual Encoder From {model_name_or_path}')

        # when path is None, means without visual encoder adapter
        if adapter_name_or_path:
            self.visual_encoder = PeftModel.from_pretrained(
                visual_encoder, adapter_name_or_path)
            print(f'Load Visual Encoder Adapter From {adapter_name_or_path}')
        else:
            self.visual_encoder = visual_encoder

        self.projector = AutoModel.from_pretrained(
            projector_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True)
        print(f'Load Projector from {projector_name_or_path}')

        self.dtype = torch.float16
        self.device = device
        self.to(self.device)
        self.to(self.dtype)

    def process_img(self, image: Image.Image) -> List[torch.Tensor]:
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

        processor = self.image_processor
        image_mean = processor.image_mean

        background_color = tuple(int(x * 255) for x in image_mean)
        squared_img = expand2square(image, background_color)

        processed = processor.preprocess(squared_img, return_tensors='pt')
        img_tensor = processed['pixel_values'][0]  # shape: 3, h, w

        # before this line, `img_tensor` is on cpu.
        img_tensor = img_tensor.to(self.device).to(self.dtype)
        return img_tensor

    @torch.no_grad()
    def forward(self, images: List[Union[str,
                                         Image.Image]]) -> List[torch.Tensor]:
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

    def _parse_model_path(self, name_or_path: str, hub: ModelHub) -> str:
        """Parse and get the directory path of the model. It supports load
        model from local directory or download from the hub.

        Args:
            name_or_path (str): The directory path or name of the model.
            hub (str): The hub to download models from.

        Returns:
            str: The local directory path of the model.

        Raises:
            NotImplementedError: If the input hub is not supported currently.
        """

        if os.path.isdir(name_or_path):
            model_path = name_or_path
        else:
            if hub == 'huggingface':
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(repo_id=name_or_path)
            elif hub == 'modelscope':
                from modelscope import snapshot_download
                model_path = snapshot_download(model_id=name_or_path)
            else:
                raise NotImplementedError(
                    'Only supports downloading models from `Huggingface` or '
                    '`Modelscope`.')

        return model_path

    def _parse_visual_encoder_path(self, model_path: str,
                                   visual_encoder_name_or_path: str,
                                   hub: ModelHub) -> str:
        """Parse and get the directory path of the visual encoder. It supports
        load visual encoder from local directory, download from the hub, or
        find it in the XTuner model directory.

        Args:
            model_path (str): The directory path of the model.
            visual_encoder_name_or_path (Optional[str]): The directory path or
                name of the visual encoder.
            hub (str): The hub to download models from.

        Returns:
            str: The local directory path of the visual encoder.

        Raises:
            NotImplementedError: If the input hub is not supported currently.
        """

        if 'visual_encoder' in os.listdir(model_path):
            assert visual_encoder_name_or_path is None
            visual_encoder_path = os.path.join(model_path, 'visual_encoder')
        elif os.path.isdir(visual_encoder_name_or_path):
            visual_encoder_path = visual_encoder_name_or_path
        else:
            if hub == 'huggingface':
                from huggingface_hub import snapshot_download
                visual_encoder_path = snapshot_download(
                    repo_id=visual_encoder_name_or_path)
            elif hub == 'modelscope':
                from modelscope import snapshot_download
                visual_encoder_path = snapshot_download(
                    model_id=visual_encoder_name_or_path)
            else:
                raise NotImplementedError(
                    'Only supports downloading models from `Huggingface` or '
                    '`Modelscope`.')

        return visual_encoder_path

    def _parse_projector_path(self, model_path: str) -> Optional[str]:
        """Parse the path of the `projector` model according to the model path.

        Args:
            model_path (str): The path to the model directory.

        Raises:
            ValueError: If the 'projector' directory is not found in the
                `model_path`.

        Returns:
            Optional[str]: The full path of 'projector' directory if exists,
                        else raises ValueError.
        """
        if 'projector' in os.listdir(model_path):
            projector_path = os.path.join(model_path, 'projector')
        else:
            # Raises exception if 'projector' directory/folder not found
            raise ValueError('Projector directory not found in given path')
        return projector_path

    def _parse_vis_enc_adapter_path(self, model_path: str) -> Optional[str]:
        """Parses the model path and returns the path to
        'visual_encoder_adapter' directory.

        Args:
            model_path (str): The path to the model directory.

        Returns:
            Optional[str]: The full path of 'visual_encoder_adapter' directory if exists,
                        else returns None.
        """
        if 'visual_encoder_adapter' in os.listdir(model_path):
            adapter_path = os.path.join(model_path, 'visual_encoder_adapter')
        else:
            # Returns None if 'visual_encoder_adapter' directory/folder not found
            adapter_path = None
        return adapter_path


if __name__ == '__main__':
    img = load_image('llava.jpeg')
    model = VisionEncoderForDeploy('xtuner/llava-internlm-7b',
                                   'openai/clip-vit-large-patch14-336')

    model.cuda()
    model.eval()
    outputs = model([img])
