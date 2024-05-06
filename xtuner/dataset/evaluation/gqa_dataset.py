import os
import os.path as osp
import json
from mmengine.dist import master_only
from xtuner.dataset.evaluation.base_eval_dataset import BaseEvalDataset

from xtuner.registry import BUILDER
from mmengine.logging import print_log
from xtuner.dataset.llava_proxy_eval_dataset import LLaVAProxyEvalDataset
from .gqa_eval_utils import eval_gqa


class GQADataset(BaseEvalDataset):
    METAINFO: dict = dict(name='gqa')

    def __init__(
            self,
            data_file,
            ann_file,
            image_folder,
            prompt_template,
            image_processor,
            tokenizer,
            pad_image_to_square=True,
            use_system=False,
            for_llava_prompt=False,
            metainfo=None,
            proxy_eval_dataset=dict(type=LLaVAProxyEvalDataset),
    ):
        super().__init__(metainfo)
        self.data_file = data_file
        self.ann_file = ann_file
        # Save detailed information for easy viewing
        self.answer_file = 'answer_gqa_results.jsonl'
        # solely for evaluation purposes
        self.prediction_file = 'pred_gqa_results.jsonl'

        self.image_folder = image_folder
        self.use_system = use_system
        self.for_llava_prompt = for_llava_prompt
        template = prompt_template
        self.template = template

        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square
        self.data = self.load_data_list()

        proxy_eval_dataset['eval_dataset'] = self
        self.proxy_eval_dataset = BUILDER.build(proxy_eval_dataset)

    def load_data_list(self):
        question_data = [json.loads(q) for q in open(os.path.expanduser(self.data_file), "r")]
        data_list = []
        for idx in range(len(question_data)):
            sample = question_data[idx]
            index = sample['question_id']
            image_path = sample['image']
            question = sample['text']
            category = sample['category']

            data = {
                'img_id': idx,
                'index': index,
                'image_path': image_path,
                'question': question,
                'category': category,
            }
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = self.proxy_eval_dataset.getitem(idx, data)
        return data_dict

    @master_only
    def evaluate(self, results, work_dir):
        answers_file = osp.join(work_dir, self.answer_file)
        ans_file = open(answers_file, "w")

        for pred_dict in results:
            idx = pred_dict["img_id"]
            gt_data = self.data[idx]

            ans_file.write(
                json.dumps(
                    {
                        "question_id": gt_data['index'],
                        "prompt": gt_data['question'],
                        "text": pred_dict['prediction'],
                        "metadata": {},
                    }
                )
                + "\n"
            )
        ans_file.close()

        all_preds = []
        for line_idx, line in enumerate(open(answers_file)):
            res = json.loads(line)
            question_id = res['question_id']
            text = res['text'].rstrip('.').lower()
            all_preds.append({"questionId": question_id, "prediction": text})

        prediction_file = osp.join(work_dir, self.prediction_file)
        with open(prediction_file, 'w') as f:
            json.dump(all_preds, f)

        evaluator = eval_gqa(questions=self.ann_file, predictions=prediction_file)
        print_log('============================================', 'current')
        scores = evaluator.forward()
        print_log('============================================', 'current')
        print_log(f'GQA successfully finished evaluating', 'current')
        return scores
