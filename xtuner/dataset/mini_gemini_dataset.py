from .llava import LLaVADataset
import torch
from PIL import Image
import os
from .utils import expand2square
import numpy as np


class MiniGeminiDataset(LLaVADataset):
    # siglip 864
    # clip 768
    def __init__(self, *args, image_size_aux=864, **kwargs):
        self.image_size_aux = image_size_aux
        super().__init__(*args, **kwargs)

        self.aux_mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.aux_std = np.array([0.26862954, 0.26130258, 0.27577711])

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
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
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            data_dict['pixel_values_aux'] = torch.zeros(3, self.image_size_aux, self.image_size_aux)
        return data_dict
