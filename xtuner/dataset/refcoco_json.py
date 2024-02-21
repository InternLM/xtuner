import copy
import itertools
import json
import os
import pickle
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from matplotlib.patches import Polygon, Rectangle
from mmengine.config import Config, ConfigDict
from PIL import Image

from xtuner.registry import BUILDER
from ..registry import BUILDER
from .huggingface import process_hf_dataset
from .llava import LLaVADataset
from .utils import expand2square


class RefCOCOJsonDataset(LLaVADataset):
    instruction_pool = [
        '[refer] {}',
        '[refer] give me the location of {}',
        '[refer] where is {} ?',
        '[refer] from this image, tell me the location of {}',
        '[refer] the location of {} is',
        '[refer] could you tell me the location for {} ?',
        '[refer] where can I locate the {} ?',
    ]

    def __init__(
        self,
        data_path,
        image_folder,
        tokenizer,
        image_processor,
        max_dataset_length=None,
        dataset_map_fn=None,
        template_map_fn=None,
        max_length=2048,
        pad_image_to_square=False,
    ):
        json_data = json.load(open(data_path))

        ######################################################
        # Only this part is different from LLaVADataset.__init__
        json_data = self.reformat_data(json_data)
        ######################################################

        for idx in range(len(json_data)):
            if isinstance(json_data[idx]['id'], int):
                json_data[idx]['id'] = str(json_data[idx]['id'])
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = process_hf_dataset(
            dataset=json_data,
            tokenizer=tokenizer,
            max_length=max_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            split='train',
            max_dataset_length=max_dataset_length,
            remove_unused_columns=False,
            pack_to_max_length=False,
            with_image_token=True)

        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square

    def reformat_data(self, json_data):
        new_json_data = []
        for sample in json_data:
            for instruction_template in self.instruction_pool:
                sample['conversations'] = self.gen_refcoco_conversations(
                    sample, instruction_template)
                new_json_data.append(copy.deepcopy(sample))
        return new_json_data

    @classmethod
    def gen_refcoco_conversations(cls, data, instruction_template='{}'):
        """build conversition data from refcoco json data as below.

        "id": "xxx",
        "image": "xxx.jpg",
        "conversations": [
        {
            "from": "human",
            "value": "xxxx"
        },
        {
            "from": "gpt",
            "value": "xxx"
        }
        """

        conversation = [
            {
                'from': 'human',
                'value': ''
            },
            {
                'from': 'gpt',
                'value': ''
            },
        ]

        instruction = instruction_template.format(data['sents'])
        bbox = cls.normalize_bbox(data['bbox'], data['height'], data['width'])
        answer = '{{<{}><{}><{}><{}>}}'.format(bbox[0], bbox[1], bbox[2],
                                               bbox[3])
        conversation[0]['value'] = instruction + '\n<image>'
        conversation[1]['value'] = answer
        return conversation

    @classmethod
    def get_data_json(
        cls,
        ann_path,
        image_path,
        dataset='refcoco',
        splitBy='unc',
    ):
        refer = REFER(ann_path, image_path, dataset, splitBy)
        ref_ids = refer.getRefIds(split='train')

        data = {}
        duplicate_data = defaultdict(list)

        for ref_id in ref_ids:
            ref = refer.loadRefs(ref_id)[0]

            image_id = '{:0>12}'.format(ref['image_id'])
            sents = [sent['raw'] for sent in ref['sentences']]
            bbox = refer.getRefBox(ref['ref_id'])

            image = Image.open(image_path + '/' + image_id + '.jpg')

            for sent in sents:
                sent_id = '_'.join(sent.split(' '))
                data_id = f'{dataset}-{splitBy}-{image_id}-{sent_id}'
                data_item = {
                    'id': data_id,
                    'image': 'coco/train2017/' + image_id + '.jpg',
                    'sents': sent,
                    'bbox': bbox,
                    'height': image.height,
                    'width': image.width
                }
                if data_id in data:
                    duplicate_data[data_id].append(data_item)
                else:
                    data[data_id] = data_item

        return list(data.values()), list(duplicate_data.values())

    @classmethod
    def normalize_bbox(cls, bbox, height, width):
        x, y, w, h = bbox

        bbox = [x / width, y / height, (x + w) / width, (y + h) / height]
        bbox = [int(x * 100) for x in bbox]
        return bbox


class RefCOCOJsonEvalDataset(RefCOCOJsonDataset):
    instruction_pool = ['[refer] give me the location of {}']

    def reformat_data(self, json_data):
        for sample in json_data:
            # reformat img_id
            img_id = sample['img_id'].split('_')[-2]
            sample['image'] = 'coco/train2017/' + img_id + '.jpg'
            sample['id'] = f"{img_id}-{sample['sents']}"
        return super().reformat_data(json_data)


class InvRefCOCOJsonDataset(RefCOCOJsonDataset):
    instruction_pool = [
        '[identify] {}',
        '[identify] what object is in this location {}',
        '[identify] identify the object present at this location {}',
        '[identify] what is it in {}',
        '[identify] describe this object in {}',
        '[identify] this {} is',
        '[identify] the object in {} is',
    ]

    @classmethod
    def gen_refcoco_conversations(cls, data, instruction_template='{}'):
        """build conversition data from refcoco json data as below.

        "id": "xxx",
        "image": "xxx.jpg",
        "conversations": [
        {
            "from": "human",
            "value": "xxxx"
        },
        {
            "from": "gpt",
            "value": "xxx"
        }
        """

        conversation = [
            {
                'from': 'human',
                'value': ''
            },
            {
                'from': 'gpt',
                'value': ''
            },
        ]
        bbox = cls.normalize_bbox(data['bbox'], data['height'], data['width'])
        bbox_str = '{{<{}><{}><{}><{}>}}'.format(bbox[0], bbox[1], bbox[2],
                                                 bbox[3])
        instruction = instruction_template.format(bbox_str)
        answer = data['sents']

        conversation[0]['value'] = instruction + '\n<image>'
        conversation[1]['value'] = answer
        return conversation


# flake8: noqa
# Refer


class REFER:

    def __init__(self, data_root, vis_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        # inv dataset is stored in the same path as normal dataset
        dataset = dataset.split('inv')[-1]
        print('loading dataset %s into memory...' % dataset)
        self.ann_dir = os.path.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.vis_root = vis_root
        elif dataset == 'refclef':
            raise 'No RefClef image data'
        else:
            raise 'No refer dataset is called [%s]' % dataset

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = os.path.join(self.ann_dir, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = pickle.load(open(ref_file, 'rb'))

        # load annotations from data/dataset/instances.json
        instances_file = os.path.join(self.ann_dir, 'instances.json')
        instances = json.load(open(instances_file))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'],
                                                       []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']
                            ]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    # rarely used I guess...
                    refs = [ref for ref in refs if ref['split'] == split]
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    raise 'No such split [%s]' % split
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [
                    self.imgToAnns[image_id] for image_id in image_ids
                    if image_id in self.imgToAnns
                ]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(
                    {self.Refs[ref_id]['ann_id']
                     for ref_id in ref_ids})
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(
                {self.Refs[ref_id]['image_id']
                 for ref_id in ref_ids})
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='box'):
        from matplotlib.collectns import PatchCollection

        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(os.path.join(self.vis_root, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('{}. {}'.format(sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(
                    polygons,
                    facecolors=color,
                    edgecolors=(1, 1, 0, 0),
                    linewidths=3,
                    alpha=1,
                )
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(
                    polygons,
                    facecolors=color,
                    edgecolors=(1, 0, 0, 0),
                    linewidths=1,
                    alpha=1,
                )
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                raise NotImplementedError('RefClef is not downloaded')
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                fill=False,
                edgecolor='green',
                linewidth=3,
            )
            ax.add_patch(box_plot)
