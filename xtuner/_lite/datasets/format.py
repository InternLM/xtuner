import re


class Alpaca2Openai():

    @classmethod
    def source_format(cls):
        data = {
            'instruction': 'INSTRUCTION',
            'input': 'INPUT',
            'output': 'OUTPUT',
        }
        return data

    @classmethod
    def target_format(cls):
        data = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'INSTRUCTION\nINPUT'
                },
                {
                    'role': 'assistant',
                    'content': 'OUTPUT'
                },
            ]
        }
        return data

    @staticmethod
    def convert(data):
        if data.get('output') == '<nooutput>':
            return {'messages': []}
        else:
            return {
                'messages': [
                    {
                        'role': 'user',
                        'content': f"{data['instruction']}\n{data['input']}"
                    },
                    {
                        'role': 'assistant',
                        'content': f"{data['output']}"
                    },
                ]
            }


def llava_to_openai(data):

    image_token = '<image>'
    conversations = data['conversations']
    messages = []

    if 'image' in data:
        image_urls = data['image']
        if isinstance(image_urls, str):
            image_urls = [image_urls]
    else:
        image_urls = None

    while conversations and conversations[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        conversations = conversations[1:]

    image_id = 0
    for convs in conversations:
        if convs['from'] == 'human':
            pattern = f'({image_token})'
            chunks = re.split(pattern, convs['value'])

            text_content = []
            img_content = []

            for chunk in chunks:
                if chunk == image_token:
                    url = image_urls[image_id]
                    if not isinstance(url, str):
                        raise TypeError(data)
                    # assert , image_url
                    item = dict(type='image_url', image_url=url)
                    img_content.append(item)
                    image_id += 1
                elif len(chunk.strip()):
                    item = dict(type='text', text=chunk.strip())
                    text_content.append(item)

            msg = {'role': 'user', 'content': img_content + text_content}
            messages.append(msg)

        elif convs['from'] == 'gpt':
            msg = {'role': 'assistant', 'content': convs['value']}
            messages.append(msg)
        else:
            raise NotImplementedError

    return {'messages': messages}


def llava_to_openai_interleave(data):

    image_token = '<image>'
    conversations = data['conversations']
    messages = []

    if 'image' in data:
        image_urls = data['image']
        if isinstance(image_urls, str):
            image_urls = [image_urls]
    else:
        image_urls = None

    while conversations and conversations[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        conversations = conversations[1:]

    image_id = 0
    for convs in conversations:
        if convs['from'] == 'human':
            pattern = f'({image_token})'
            chunks = re.split(pattern, convs['value'])

            content = []

            for chunk in chunks:
                if chunk == image_token:
                    url = image_urls[image_id]
                    if not isinstance(url, str):
                        raise TypeError(data)
                    # assert , image_url
                    item = dict(type='image_url', image_url=url)
                    content.append(item)
                    image_id += 1
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


OPENAI_FORMAT_MAP = {
    'llava': llava_to_openai,
    'llava_interleave': llava_to_openai_interleave,
    'alpaca': Alpaca2Openai.convert,
    'openai': lambda x: x,
}
