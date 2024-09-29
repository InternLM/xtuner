# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from typing import Any, Sequence

import evaluate
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from rich.console import Console
from rich.table import Table

import xtuner
from xtuner.registry import BUILDER


class SacreBLEUMetric(BaseMetric):

    def __init__(self,
                 tokenizer,
                 dump_dir='./',
                 epoch_num='',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        tokenizer = BUILDER.build(tokenizer)
        xtuner_path = os.path.dirname(xtuner.__file__)
        self.sacrebleu = evaluate.load(
            os.path.join(xtuner_path, 'evaluation/metrics/sacrebleu.py'))
        self.dump_dir = dump_dir
        self.epoch_num = epoch_num

    @staticmethod
    def accuracy(preds, gts):
        """Computes the accuracy for preds and gts."""
        correct = [1 if pred == gt else 0 for pred, gt in zip(preds, gts)]
        acc = np.mean(correct) * 100
        return acc

    def process(self, data_batch: Any, generations: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        for predict, data_sample in zip(generations,
                                        data_batch['data_samples']):
            predict = predict
            golden_text = data_sample['tgt_lang_text']

            print('---' * 10 + 'Start' + '---' * 10)
            print('Prediction:\n{}\nGolden:\n{}\n'.format(
                predict, golden_text))
            print('---' * 10 + 'End' + '---' * 10)
            self.results.append(
                dict(predictions=predict, references=golden_text))
            tmp_file_path = os.path.join(
                self.dump_dir, f'tmp_inference_results{self.epoch_num}.json')
            with open(tmp_file_path, 'w') as file:
                json.dump(self.results, file, indent=4)
            # import pdb;pdb.set_trace()

    def compute_metrics(self, results: list, tokenize='13a') -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        if 'en_zh' in self.epoch_num:
            tokenize = 'zh'
        elif 'en_ja' in self.epoch_num:
            tokenize = 'ja-mecab'

        file_path = os.path.join(self.dump_dir,
                                 f'inference_results{self.epoch_num}.json')

        with open(file_path, 'w') as file:
            json.dump(results, file, indent=4)

        print(f'Total Length of the Prediction:{len(results)}')
        predictions = [item['predictions'] for item in results]
        references = [item['references'] for item in results]

        # convert to lower
        predictions = [item.lower() for item in predictions]
        references = [item.lower() for item in references]

        metric_results = self.sacrebleu.compute(
            predictions=predictions, references=references, tokenize=tokenize)
        self._print_results(metric_results)
        return metric_results

    def _print_results(self, table_metrics: dict) -> None:
        table_title = ' SacreBLEU Metric '
        table = Table(title=table_title)
        console = Console()
        table.add_column('Metric', justify='left')
        table.add_column('Value', justify='right')
        for key, value in table_metrics.items():
            table.add_row(key, str(value))
        with console.capture() as capture:
            console.print(table, end='')
        print_log('\n' + capture.get(), 'current')
