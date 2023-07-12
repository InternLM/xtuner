from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmchat.registry import METRICS, TOKENIZER

@METRICS.register_module()
class MMLUMetric(BaseMetric):
    METAINFO = {
        'subjects':
        ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
         'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
         'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
         'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
         'global_facts', 'high_school_biology', 'high_school_chemistry',
         'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
         'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
         'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
         'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
         'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies',
         'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
         'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
         'professional_accounting', 'professional_law', 'professional_medicine',
         'professional_psychology', 'public_relations', 'security_studies', 'sociology',
         'us_foreign_policy', 'virology', 'world_religions')
    }

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokenizer = TOKENIZER.build(tokenizer)
        self.abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]

    @staticmethod
    def ABCD_to_0123(abcd):
        func = lambda x: {'A': 0, 'B': 1, 'C': 2, 'D': 3}[x]
        return func(abcd)

    @staticmethod
    def accuracy(preds, gts):
        """Computes the accuracy for preds and gts"""
        correct = [1 for pred, gt in zip(preds, gts) if pred == gt]
        acc = sum(correct) / len(preds) * 100
        return acc

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        subjects = data_batch['subject']
        gts = [self.ABCD_to_0123(gt) for gt in data_batch['output']]
        preds = []
        for sample, subject, gt in zip(data_samples, subjects, gts):
            pred_logits = sample['logits']
            labels = sample['labels']
            labels_non_zero_id = (labels != -100).nonzero()[0][0]
            pred_logtis_abcd = pred_logits[labels_non_zero_id-1, self.abcd_idx]
            pred = torch.argmax(pred_logtis_abcd).item()
            preds.append(pred)
            self.results.append((subject, pred, gt))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        subject_results = {subject: {'preds': [], 'gts': []} for subject in self.METAINFO['subjects']}
        for subject, pred, gt in results:
            subject_results[subject]['preds'].append(pred)
            subject_results[subject]['gts'].append(gt)
        metrics = dict()
        for subject in self.METAINFO['subjects']:
            if len(subject_results[subject]['preds']) == 0:
                logger.info(f'Skip subject {subject} for mmlu')
            else:
                score = self.accuracy(subject_results[subject]['preds'], subject_results[subject]['gts'])
                metrics[f'{subject}'] = score
        metrics['average'] = np.mean(list(metrics.values()))
        return metrics
