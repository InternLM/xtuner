import os
import os.path as osp
from typing import Optional
import json
from mmengine.dist import master_only
from xtuner.dataset.evaluation.base_eval_dataset import BaseEvalDataset

from xtuner.registry import BUILDER
from mmengine.logging import print_log
from xtuner.dataset.llava_proxy_eval_dataset import LLaVAProxyEvalDataset


def relaxed_correctness(prediction: str,
                        target: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      prediction: Predicted string.
      target: Target string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['label'], str):
            elem['label'] = [elem['label']]
        score = max([
            relaxed_correctness(elem['prediction'].strip(), ann)
            for ann in elem['label']
        ])
        scores.append(score)
    return scores, sum(scores) / len(scores)


class ChartQADataset(BaseEvalDataset):
    METAINFO: dict = dict(name='chartqa')

    def __init__(
            self,
            data_file,
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
        self.use_system=use_system
        self.for_llava_prompt = for_llava_prompt

        if isinstance(data_file, str):
            data_file = [data_file]
        self.raw_data = [json.load(open(f)) for f in data_file]
        # test_human, test_augmented
        self.name = [
            os.path.splitext(os.path.basename(f))[0] for f in data_file
        ]
        self.name_map = {name: i for i, name in enumerate(self.name)}
        self.revert_name_map = {i: name for i, name in enumerate(self.name)}

        template = prompt_template
        self.template = template

        self.image_folder = image_folder
        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square
        self.data = self.load_data_list()

        proxy_eval_dataset['eval_dataset'] = self
        self.proxy_eval_dataset = BUILDER.build(proxy_eval_dataset)

    def load_data_list(self):
        data_list = []
        idx = 0

        for data_idx in range(len(self.raw_data)):
            for sample_idx in range(len(self.raw_data[data_idx])):
                sample = self.raw_data[data_idx][sample_idx]
                image_path = sample['imgname']
                question = sample['query']
                answer = sample['label']
                category = self.name[data_idx]
                data = {
                    'img_id': idx,
                    'image_path': image_path,
                    'question': question,
                    'answer': answer,
                    'category': category
                }
                data_list.append(data)
                idx += 1
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = self.proxy_eval_dataset.getitem(idx, data)
        return data_dict

    @master_only
    def evaluate(self, result, work_dir):
        orig_index = [x['img_id'] for x in self.data]
        results = [[] for _ in range(len(self.name))]
        for pred_dict in result:
            index = pred_dict['img_id']
            new_index = orig_index.index(index)
            filtered_rows = self.data[new_index]
            cur_result = {}
            cur_result['query'] = filtered_rows.get('question')
            cur_result['prediction'] = pred_dict['prediction']
            cur_result['label'] = filtered_rows.get('answer')

            index = self.name_map[filtered_rows['category']]
            results[index].append(cur_result)

        print_log('============================================', 'current')
        acc_list = []
        for i, result in enumerate(results):
            scores, _accuracy = evaluate_relaxed_accuracy(result)

            for res, score in zip(result, scores):
                res['score'] = score
            prediction_file = osp.join(work_dir, self.revert_name_map[i] + '.json')
            with open(prediction_file, 'w') as f:
                json.dump(result, f)

            print_log('Acc: {}, Category: {}, # samples: {}'.format(_accuracy, self.revert_name_map[i],
                                                                    len(result)), 'current')
            acc_list.append(_accuracy)

        print_log('============================================', 'current')
        acc = sum(acc_list) / len(acc_list)
        print_log('Overall Acc: {}'.format(acc), 'current')
        print_log('============================================', 'current')
        print_log('ChartQA successfully finished evaluating', 'current')

        return {'Acc': acc}
