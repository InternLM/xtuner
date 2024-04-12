import os
import os.path as osp
import json
from mmengine.dist import master_only
from xtuner.dataset.evaluation.base_eval_dataset import BaseEvalDataset

from xtuner.registry import BUILDER
from mmengine.logging import print_log
from xtuner.dataset.llava_proxy_eval_dataset import LLaVAProxyEvalDataset
from .vqav2_utils import EvalAIAnswerProcessor


class VQAv2Dataset(BaseEvalDataset):

    METAINFO: dict = dict(name='vqa_v2')

    def __init__(
        self,
        data_file,
        test_file,
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
        self.test_file = test_file
        self.image_folder = image_folder
        # Save detailed information for easy viewing
        self.answer_file = 'answer_vqav2_results.jsonl'
        # solely for evaluation purposes
        self.prediction_file = 'pred_vqav2_results.jsonl'
        self.answer_processor = EvalAIAnswerProcessor()

        self.use_system = use_system
        self.for_llava_prompt = for_llava_prompt
        self.template = prompt_template
        self.pad_image_to_square = pad_image_to_square

        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)

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

        results = []
        error_line = 0
        for line_idx, line in enumerate(open(answers_file)):
            try:
                results.append(json.loads(line))
            except:
                error_line += 1

        results = {x['question_id']: x['text'] for x in results}
        test_split = [json.loads(line) for line in open(self.test_file)]

        all_answers = []

        for x in test_split:
            if x['question_id'] not in results:
                all_answers.append({
                    'question_id': x['question_id'],
                    'answer': ''
                })
            else:
                all_answers.append({
                    'question_id': x['question_id'],
                    'answer': self.answer_processor(results[x['question_id']])
                })

        prediction_file = osp.join(work_dir, self.prediction_file)
        with open(prediction_file, 'w') as f:
            json.dump(all_answers, f)

        print_log('============================================', 'current')
        print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')
        print_log(f'Please submit the generated {prediction_file} file to the official server for evaluation.',
                  'current')
        print_log('============================================', 'current')
        return {'acc': 0}
