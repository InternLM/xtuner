import os
import os.path as osp
import re

from .base_eval_dataset import BaseEvalDataset

from xtuner.registry import BUILDER
import json
from mmengine.dist import (master_only)
from .textvqa_utils import TextVQAAccuracyEvaluator
from mmengine.logging import print_log
from ..llava_proxy_eval_dataset import LLaVAProxyEvalDataset


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


class TextVQADataset(BaseEvalDataset):
    METAINFO: dict = dict(name='textvqa')

    def __init__(self, data_file, ann_file, image_folder, prompt_template, image_processor, tokenizer,
                 pad_image_to_square=True, use_system=False, metainfo=None,
                 proxy_eval_dataset=dict(type=LLaVAProxyEvalDataset)):
        super().__init__(metainfo)
        self.use_system = use_system
        self.data_file = data_file
        self.ann_file = ann_file
        self.image_folder = image_folder

        template = prompt_template
        self.template = template

        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square
        self.name = os.path.splitext(os.path.basename(data_file))[0]
        self.results_path = os.path.splitext(os.path.basename(data_file))[0] + '-results.jsonl'
        self.data = self.load_data_list()

        proxy_eval_dataset['eval_dataset'] = self
        self.proxy_eval_dataset = BUILDER.build(proxy_eval_dataset)

    def load_data_list(self):
        data = [json.loads(q) for q in open(os.path.expanduser(self.data_file), "r")]
        for i, d in enumerate(data):
            d['img_id'] = i
            d['image_path'] = d['image']
            d['question'] = d['text']
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = self.proxy_eval_dataset.getitem(idx, data)
        return data_dict

    @master_only
    def evaluate(self, result, work_dir, show=True):
        answers_file = osp.join(work_dir, self.results_path)
        ans_file = open(answers_file, "w")

        for pred_dict in result:
            idx = pred_dict["img_id"]
            gt_data = self.data[idx]

            ans_file.write(json.dumps({"question_id": gt_data['question_id'],
                                       "prompt": gt_data['text'],
                                       "text": pred_dict['prediction'],
                                       "metadata": {}}) + "\n")
        ans_file.close()

        annotations = json.load(open(self.ann_file))['data']
        annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in
                       annotations}
        results = [json.loads(line) for line in open(answers_file)]

        pred_list = []
        for result in results:
            annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
            pred_list.append({
                "pred_answer": result['text'],
                "gt_answers": annotation['answers'],
            })

        evaluator = TextVQAAccuracyEvaluator()
        acc = 100. * evaluator.eval_pred_list(pred_list)
        print_log('============================================', 'current')
        print_log('Samples: {}, Accuracy: {:.2f}%'.format(len(pred_list), acc), 'current')
        print_log('============================================', 'current')
        print_log(f'TextVQA successfully finished evaluating', 'current')
        return {'acc': acc}
