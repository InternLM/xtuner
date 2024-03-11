# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

from xtuner.dataset.refcoco_json import RefCOCOJsonDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann-path',
        default='data/refcoco_annotations',
        help='Refcoco annotation path',
    )
    parser.add_argument(
        '--image-path',
        default='data/llava_data/llava_images/coco/train2017',
        help='COCO image path',
    )
    parser.add_argument(
        '--save-path', default='./', help='The folder to save converted data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    data_info = [
        ('refcoco', 'unc'),
        ('refcoco+', 'unc'),
        ('refcocog', 'umd'),
    ]
    all_data = []
    for dataset, split in data_info:
        data = RefCOCOJsonDataset.get_data_json(
            ann_path=args.ann_path,
            image_path=args.image_path,
            dataset=dataset,
            splitBy=split,
        )[0]
        all_data.extend(data)
    save_path = args.save_path + '/train.json'
    with open(save_path, 'w') as f:
        print(f'save to {save_path} with {len(all_data)} items.')
        print(all_data[0])
        json.dump(all_data, f, indent=4)
