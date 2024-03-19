import re
from typing import Callable, Dict, List, Type

from mmengine.config.lazy import LazyObject

from xtuner.types import TrainingHybridChatMessages


def map_protocol(
    input_keys: Dict[str, Type] = {},
    output_keys: Dict[str, Type] = {},
    added_keys: Dict[str, Type] = {},
) -> Callable:

    def decorator(func):

        def wrapper(data, *args, **kwargs):

            for key, _type in input_keys.items():
                assert key in data
                if not isinstance(data[key], _type):
                    breakpoint()

            result = func(data, *args, **kwargs)

            for key, _type in output_keys.items():
                assert key in result
                assert isinstance(result[key], _type)

            return result

        return wrapper

    setattr(decorator, 'input_keys', input_keys)
    setattr(decorator, 'output_keys', output_keys)
    setattr(decorator, 'added_keys', added_keys)

    return decorator


def map_sequential(mappings: List[Callable]):

    if not isinstance(mappings, List):
        mappings = list(mappings)

    for i in range(len(mappings)):
        if isinstance(mappings[i], LazyObject):
            mappings[i] = mappings[i].build()

    def _sequential(item, tokenizer, chat_template):

        for func in mappings:
            item = func(item, tokenizer, chat_template)

        return item

    return _sequential


@map_protocol(
    input_keys=dict(input_ids=list, labels=list, image_urls=list),
    output_keys=dict(
        input_ids=list, labels=list, image_urls=list, image_ranges=list),
)
def insert_img_pad_tokens(data, tokenizer, chat_template) -> Dict:

    image_urls = data['image_urls']
    if len(image_urls) == 0:
        data['image_ranges'] = []
        return data

    input_ids = data['input_ids']
    labels = data['labels']

    img_token = chat_template.image_token_index
    img_token_inds = [i for i, t in enumerate(input_ids) if t == img_token]
    assert len(img_token_inds) == len(
        image_urls), f'{img_token_inds} {image_urls}'

    for url, ind in zip(image_urls, img_token_inds):
        # image = self.load_image(url)
        h, w = 336 // 14, 336 // 14

        pad_tokens = [tokenizer.pad_token_id] * (h * w)
        pad_labels = [labels[ind]] * (h * w)

        input_ids[ind] = pad_tokens
        labels[ind] = pad_labels

    new_ids = []
    new_labels = []
    assert len(input_ids) == len(labels)

    img_ranges = []
    for i, _ in enumerate(zip(input_ids, labels)):
        if isinstance(input_ids[i], list):
            assert isinstance(labels[i], list)
            assert len(input_ids[i]) == len(labels[i])

            img_begin = len(new_ids)
            img_end = img_begin + len(input_ids[i])
            img_ranges.append([img_begin, img_end])

            new_ids.extend(input_ids[i])
            new_labels.extend(labels[i])

        else:
            new_ids.append(input_ids[i])
            new_labels.append(labels[i])

    data['input_ids'] = new_ids
    data['labels'] = new_labels
    data['image_ranges'] = img_ranges

    return data


@map_protocol(
    input_keys=dict(messages=list),
    output_keys=dict(input_ids=list, labels=list, image_urls=list),
)
def openai_to_raw_training(item: dict, tokenizer, chat_template) -> Dict:

    data = TrainingHybridChatMessages.from_dict(item)
    data = data.tokenize(tokenizer, chat_template)

    return data


@map_protocol(
    input_keys=dict(conversations=list),
    output_keys=dict(messages=list),
)
def llava_to_openai(data, tokenizer=None, chat_template=None):

    image_token = '<image>'
    conversations = data['conversations']
    messages = []

    if 'image' in data:
        image_url = data['image']
    else:
        image_url = None

    while conversations and conversations[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        conversations = conversations[1:]

    for convs in conversations:
        if convs['from'] == 'human':
            pattern = f'({image_token})'
            chunks = re.split(pattern, convs['value'])

            content = []
            for chunk in chunks:
                if chunk == image_token:
                    assert isinstance(image_url, str), image_url
                    item = dict(type='image_url', image_url=image_url)
                    content.append(item)
                elif len(chunk.strip()):
                    item = dict(type='text', text=chunk.strip())
                    content.append(item)

            msg = {'role': 'user', 'content': content}
            messages.append(msg)

        elif convs['from'] == 'gpt':
            msg = {'role': 'assistant', 'content': convs['value']}
            messages.append(msg)
        else:
            raise NotImplementedError
    return {'messages': messages}
