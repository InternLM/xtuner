# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from rich.console import Console
from rich.table import Table

from xtuner.registry import BUILDER


class MMLUMetric(BaseMetric):
    METAINFO = {
        'subcategories': {
            'abstract_algebra': ['math'],
            'anatomy': ['health'],
            'astronomy': ['physics'],
            'business_ethics': ['business'],
            'clinical_knowledge': ['health'],
            'college_biology': ['biology'],
            'college_chemistry': ['chemistry'],
            'college_computer_science': ['computer science'],
            'college_mathematics': ['math'],
            'college_medicine': ['health'],
            'college_physics': ['physics'],
            'computer_security': ['computer science'],
            'conceptual_physics': ['physics'],
            'econometrics': ['economics'],
            'electrical_engineering': ['engineering'],
            'elementary_mathematics': ['math'],
            'formal_logic': ['philosophy'],
            'global_facts': ['other'],
            'high_school_biology': ['biology'],
            'high_school_chemistry': ['chemistry'],
            'high_school_computer_science': ['computer science'],
            'high_school_european_history': ['history'],
            'high_school_geography': ['geography'],
            'high_school_government_and_politics': ['politics'],
            'high_school_macroeconomics': ['economics'],
            'high_school_mathematics': ['math'],
            'high_school_microeconomics': ['economics'],
            'high_school_physics': ['physics'],
            'high_school_psychology': ['psychology'],
            'high_school_statistics': ['math'],
            'high_school_us_history': ['history'],
            'high_school_world_history': ['history'],
            'human_aging': ['health'],
            'human_sexuality': ['culture'],
            'international_law': ['law'],
            'jurisprudence': ['law'],
            'logical_fallacies': ['philosophy'],
            'machine_learning': ['computer science'],
            'management': ['business'],
            'marketing': ['business'],
            'medical_genetics': ['health'],
            'miscellaneous': ['other'],
            'moral_disputes': ['philosophy'],
            'moral_scenarios': ['philosophy'],
            'nutrition': ['health'],
            'philosophy': ['philosophy'],
            'prehistory': ['history'],
            'professional_accounting': ['other'],
            'professional_law': ['law'],
            'professional_medicine': ['health'],
            'professional_psychology': ['psychology'],
            'public_relations': ['politics'],
            'security_studies': ['politics'],
            'sociology': ['culture'],
            'us_foreign_policy': ['politics'],
            'virology': ['health'],
            'world_religions': ['philosophy'],
        },
        'categories': {
            'STEM': [
                'physics', 'chemistry', 'biology', 'computer science', 'math',
                'engineering'
            ],
            'humanities': ['history', 'philosophy', 'law'],
            'social sciences':
            ['politics', 'culture', 'economics', 'geography', 'psychology'],
            'other (business, health, misc.)': ['other', 'business', 'health'],
        },
    }
    METAINFO['subcategories_list'] = list({
        subcat
        for subcats in METAINFO['subcategories'].values() for subcat in subcats
    })

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokenizer = BUILDER.build(tokenizer)
        self.abcd_idx = [
            tokenizer.encode('A', add_special_tokens=False)[0],
            tokenizer.encode('B', add_special_tokens=False)[0],
            tokenizer.encode('C', add_special_tokens=False)[0],
            tokenizer.encode('D', add_special_tokens=False)[0],
        ]

    @staticmethod
    def ABCD_to_0123(abcd):
        return {'A': 0, 'B': 1, 'C': 2, 'D': 3}[abcd]

    @staticmethod
    def find_first_zero_index(tensor):
        indices = torch.nonzero(tensor == 0)
        if indices.numel() > 0:
            return indices[0].item()
        else:
            return None

    @staticmethod
    def accuracy(preds, gts):
        """Computes the accuracy for preds and gts."""
        correct = [1 if pred == gt else 0 for pred, gt in zip(preds, gts)]
        acc = np.mean(correct) * 100
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
        subjects = data_batch['data_samples']['subjects']
        gts = [
            self.ABCD_to_0123(gt)
            for gt in data_batch['data_samples']['labels']
        ]
        preds = []
        for sample, attn_mask, subject, gt in zip(
                data_samples, data_batch['data']['attention_mask'], subjects,
                gts):
            pred_logits = sample['logits']
            first_zero_idx = self.find_first_zero_index(attn_mask)
            pred_idx = -1 if first_zero_idx is None else first_zero_idx - 1
            pred_logtis_abcd = pred_logits[pred_idx, self.abcd_idx]
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
        subjects_results = {
            subject: {
                'preds': [],
                'gts': []
            }
            for subject in self.METAINFO['subcategories'].keys()
        }
        subcats_results = {
            subcat: {
                'preds': [],
                'gts': []
            }
            for subcat in self.METAINFO['subcategories_list']
        }
        cats_results = {
            cat: {
                'preds': [],
                'gts': []
            }
            for cat in self.METAINFO['categories'].keys()
        }
        for subject, pred, gt in results:
            subjects_results[subject]['preds'].append(pred)
            subjects_results[subject]['gts'].append(gt)
            subcats = self.METAINFO['subcategories'][subject]
            for subcat in subcats:
                subcats_results[subcat]['preds'].append(pred)
                subcats_results[subcat]['gts'].append(gt)
        for cat, subcats in self.METAINFO['categories'].items():
            for subcat in subcats:
                if subcat in subcats_results:
                    cats_results[cat]['preds'].extend(
                        subcats_results[subcat]['preds'])
                    cats_results[cat]['gts'].extend(
                        subcats_results[subcat]['gts'])

        subjects_metrics = dict()
        subcats_metrics = dict()
        cats_metrics = dict()
        for subject in self.METAINFO['subcategories'].keys():
            assert len(subjects_results[subject]['preds']) == len(
                subjects_results[subject]['gts'])
            if len(subjects_results[subject]['preds']) == 0:
                print_log(f'Skip subject {subject} for mmlu', 'current')
            else:
                score = self.accuracy(subjects_results[subject]['preds'],
                                      subjects_results[subject]['gts'])
                subjects_metrics[f'{subject}'] = score
        for subcat in self.METAINFO['subcategories_list']:
            assert len(subcats_results[subcat]['preds']) == len(
                subcats_results[subcat]['gts'])
            if len(subcats_results[subcat]['preds']) == 0:
                print_log(f'Skip subcategory {subcat} for mmlu', 'current')
            else:
                score = self.accuracy(subcats_results[subcat]['preds'],
                                      subcats_results[subcat]['gts'])
                subcats_metrics[f'{subcat}'] = score
        for cat in self.METAINFO['categories'].keys():
            assert len(cats_results[cat]['preds']) == len(
                cats_results[cat]['gts'])
            if len(cats_results[cat]['preds']) == 0:
                print_log(f'Skip category {cat} for mmlu', 'current')
            else:
                score = self.accuracy(cats_results[cat]['preds'],
                                      cats_results[cat]['gts'])
                cats_metrics[f'{cat}'] = score

        metrics = dict()
        metrics.update(subjects_metrics)
        metrics.update(subcats_metrics)
        metrics.update(cats_metrics)
        metrics['average'] = np.mean(list(subjects_metrics.values()))

        table_metrics = dict()
        table_metrics.update(cats_metrics)
        table_metrics['average'] = np.mean(list(subjects_metrics.values()))
        self._print_results(table_metrics)
        return metrics

    def _print_results(self, table_metrics: dict) -> None:
        table_title = ' MMLU Benchmark '
        table = Table(title=table_title)
        console = Console()
        table.add_column('Categories', justify='left')
        table.add_column('Accuracy (%)', justify='right')
        for cat, acc in table_metrics.items():
            table.add_row(cat, f'{acc:.1f}')
        with console.capture() as capture:
            console.print(table, end='')
        print_log('\n' + capture.get(), 'current')
