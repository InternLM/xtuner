from .llava import LLaVADataset
import torch
from PIL import Image
import os
from .utils import expand2square
import numpy as np


class MiniGeminiDataset(LLaVADataset):
    # siglip 864
    # clip 768
    def __init__(self, *args, image_size_aux=768, **kwargs):
        self.image_size_aux = image_size_aux
        super().__init__(*args, **kwargs)

        self._model_name = type(self.image_processor).__name__

        if self._model_name == 'CLIPImageProcessor':
            self.crop_size_raw = self.image_processor.crop_size.copy()
            self.image_processor.crop_size['height'] = image_size_aux
            self.image_processor.crop_size['width'] = image_size_aux
            self.image_processor.size['shortest_edge'] = image_size_aux
        else:
            self.aux_mean = np.array([0.48145466, 0.4578275, 0.40821073])
            self.aux_std = np.array([0.26862954, 0.26130258, 0.27577711])

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')

            if self._model_name == 'CLIPImageProcessor':
                # clip 和 convnext 均值和方差一样，前处理相同，但是 siglip 不一致
                if self.pad_image_to_square:
                    image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))

                image_aux = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                data_dict['pixel_values_aux'] = image_aux

                image = image_aux.clone()
                image = torch.nn.functional.interpolate(
                    image[None], size=[self.crop_size_raw['height'], self.crop_size_raw['width']], mode='bilinear',
                    align_corners=False
                )[0]
                data_dict['pixel_values'] = image
            else:
                # siglip
                image_aux = image
                if self.pad_image_to_square:
                    image = expand2square(
                        image,
                        tuple(
                            int(x * 255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                data_dict['pixel_values'] = image

                # aux image
                if self.pad_image_to_square:
                    image_aux = expand2square(
                        image_aux,
                        tuple(
                            int(x * 255) for x in self.aux_mean))
                image_aux = image_aux.resize((self.image_size_aux, self.image_size_aux), resample=Image.BILINEAR)
                image_aux = np.array(image_aux)  # H, W, 3
                image_aux = image_aux / 255.0
                image_aux = (image_aux - self.aux_mean) / self.aux_std
                image_aux = torch.tensor(image_aux).permute(2, 0, 1)
                data_dict['pixel_values_aux'] = image_aux
        else:
            data_dict['pixel_values_aux'] = torch.zeros(3, self.image_size_aux, self.image_size_aux)
            if self._model_name == 'CLIPImageProcessor':
                data_dict['pixel_values'] = torch.zeros(3, self.crop_size_raw['height'],
                                                        self.crop_size_raw['width'])
        return data_dict
