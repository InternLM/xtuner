import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor

from xtuner.dataset.utils import expand2square
from .dataset import TextDataset


class LlavaDataset(TextDataset):

    def __init__(self, image_dir: str, pad_img_to_squared: bool, *args,
                 **kwargs):
        super.__init__(*args, **kwargs)
        if self.pack_to_max_length:
            raise NotImplementedError

        self.image_processor = CLIPImageProcessor.from_pretrained(
            'openai/clip-vit-large-patch14-336',
            trust_remote_code=True,
        )
        self.pad_img_to_squared = pad_img_to_squared
        self.image_dir = image_dir

    def tokenize_dataset(self, dataset: List[dict]) -> List[dict]:

        def llava_to_training(data):

            image_token = '<image>'
            conversations = data['conversations']

            if 'image' in data:
                image_url = data['image']
            else:
                image_url = None

            while conversations and conversations[0]['from'] == 'gpt':
                # Skip the first one if it is from gpt
                conversations = conversations[1:]

            input_ids = []
            labels = []

            for convs in conversations:
                if convs['from'] == 'human':
                    pattern = f'({image_token})'
                    chunks = re.split(pattern, convs['value'])

                    for chunk in chunks:
                        if chunk == image_token:
                            assert isinstance(image_url, str), image_url
                            input_ids.append(-200)
                            labels.append(-100)
                        elif len(chunk.strip()):
                            prompt = self.chat_template.decorate_user(
                                chunk.strip())
                            tokens = self.tokenizer.encode(
                                prompt, add_special_tokens=False)
                            input_ids.extend(tokens)
                            labels.extend([-100] * len(tokens))

                elif convs['from'] == 'gpt':
                    prompt = self.chat_template.decorate_assistant(
                        chunk.strip())
                    tokens = self.tokenizer.encode(
                        prompt, add_special_tokens=False)
                    input_ids.extend(tokens)
                    labels.extend(labels)
                else:
                    raise NotImplementedError
            return {
                'input_ids': input_ids,
                'labels': labels,
                'image_url': image_url
            }

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            dataset = list(
                tqdm(
                    executor.map(llava_to_training, dataset),
                    desc='Map Dataset',
                    total=len(dataset)))
        return dataset

    def load_image(self, url):
        image_file = self.image_dir / url
        image = Image.open(image_file).convert('RGB')

        if self.pad_img_to_squared:
            background = tuple(
                int(x * 255) for x in self.image_processor.image_mean)
            image = expand2square(image, background)

        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]

        return image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int) -> Dict[str, List]:

        data = self.dataset[item]

        if data['image_urls']:
            image = self.load_image(data['image_urls'])
            data['pixel_values'] = image

        return data
