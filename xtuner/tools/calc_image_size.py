from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
from PIL import Image
import os

data_root = '/mnt/petrelfs/share_data/huanghaian/llava_data/'
data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
image_folder = data_root + 'llava_images'


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def calc_fn(data_dict):
    size = {'width': 0, 'height': 0, 'image': 'None'}
    if data_dict.get('image', None) is not None:
        image_file = data_dict['image']
        image = Image.open(os.path.join(image_folder,
                                        image_file))
        size['image'] = image_file
        size['width'] = image.size[0]
        size['height'] = image.size[1]
    return size


if __name__ == '__main__':
    print('start calculating modality length')
    if data_path.endswith('.json'):
        json_data = json.load(open(data_path))
    elif data_path.endswith('.jsonl'):
        json_data = load_jsonl(data_path)
    else:
        raise NotImplementedError

    with ThreadPoolExecutor(max_workers=16) as executor:
        length_list = list(
            tqdm(
                executor.map(calc_fn, json_data),
                desc='Calculating modality length',
                total=len(json_data)))
    print('end calculating modality length')

    new_output_dict = {}
    for i in range(len(length_list)):
        if length_list[i]['image'] != 'None':
            new_output_dict[length_list[i]['image']] = [length_list[i]['width'], length_list[i]['height']]

    with open('llava_v1_5_mix665k_image_size.json', 'w') as f:
        json.dump(new_output_dict, f)
