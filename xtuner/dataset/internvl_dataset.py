from .llava import LLaVADataset
from mmengine import print_log
from mmengine.fileio import get
import io
from PIL import Image
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from xtuner.utils import IMAGE_TOKEN_INDEX, IGNORE_INDEX


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def total_image_token(orig_size, min_num=1, max_num=12, image_size=448, patch_size=16, use_thumbnail=True):
    orig_width, orig_height = orig_size

    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        max_num >= i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    if use_thumbnail:
        blocks += 1

    return blocks*patch_size*patch_size + 2  # 2 for <img> and </img>


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
qualities = list(range(75, 101))


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade


jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        raise NotImplementedError
    return transform


IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'


class InternVL_V1_5_LLaVADataset(LLaVADataset):
    def __init__(self, path, *args, image_processor=None, **kwargs):
        self.cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self.min_dynamic_patch = self.cfg.min_dynamic_patch
        self.max_dynamic_patch = self.cfg.max_dynamic_patch
        self.downsample_ratio = self.cfg.downsample_ratio
        self.image_size = self.cfg.force_image_size
        self.use_thumbnail = self.cfg.use_thumbnail
        self.transformer = build_transform(True, self.image_size)

        self.max_refetch = 1000
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        super().__init__(*args, tokenizer=self.tokenizer, image_processor=image_processor, **kwargs)

    @property
    def modality_length(self):
        length_list = self.text_data['length']
        print_log('end calculating modality length', logger='current')
        return length_list

    def __getitem__(self, index):
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def get_image(self, path):
        if "s3://" in path:
            img_bytes = get(path)
            with io.BytesIO(img_bytes) as buff:
                img = Image.open(buff).convert('RGB')
            return img
        else:
            return Image.open(path).convert('RGB')

    def prepare_data(self, index):
        data_dict = self.text_data[index]

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            assert len(image_file) == 1
            image_file = image_file[0]

            try:
                image = self.get_image(os.path.join(self.image_folder, image_file))
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None

            images = dynamic_preprocess(image, self.min_dynamic_patch, self.max_dynamic_patch, self.image_size)
            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            data_dict['pixel_values'] = pixel_values

            # TODO: more simple way to replace image token
            num_image_tokens = pixel_values.shape[0] * self.patch_token
            image_token = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_tokens}{IMG_END_TOKEN}'
            image_input_ids = self.tokenizer(image_token, add_special_tokens=False).input_ids

            # replace image token to f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
            input_ids = data_dict['input_ids']  # list
            old_image_token_index = input_ids.index(IMAGE_TOKEN_INDEX)
            pre_list = input_ids[:old_image_token_index]
            post_list = input_ids[old_image_token_index + 1:]
            input_ids = pre_list + image_input_ids + post_list
            data_dict['input_ids'] = input_ids

            labels = data_dict['labels']
            pre_list = labels[:old_image_token_index]
            post_list = labels[old_image_token_index + 1:]
            labels = pre_list + [IGNORE_INDEX] * len(image_input_ids) + post_list
            data_dict['labels'] = labels
        else:
            data_dict['pixel_values'] = torch.zeros(1, 3, self.image_size, self.image_size)
        return data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.text_data))
