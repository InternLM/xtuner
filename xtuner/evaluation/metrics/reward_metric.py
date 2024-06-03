import itertools
from collections import defaultdict
from typing import List, Optional, Sequence

import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from rich.console import Console
from rich.table import Table


class RewardMetric(BaseMetric):
    r"""Reward model evaluation metric.
    """
    default_prefix: Optional[str] = ''

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        logits = torch.cat(
            [sample['logits'].unsqueeze(0) for sample in data_samples], dim=0)
        labels = data_batch['data']['labels']
        ds_names = data_batch['data_samples']['ds_names']
        chosen_idx = torch.where(labels == 0)
        rejected_idx = torch.where(labels == 1)
        chosen_logits = logits[chosen_idx].cpu()
        rejected_logits = logits[rejected_idx].cpu()

        correct = (chosen_logits > rejected_logits).cpu()
        self.results.append({
            'chosen_logits': chosen_logits,
            'rejected_logits': rejected_logits,
            'correct': correct,
            'ds_names': ds_names
        })

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        correct = torch.cat([res['correct'] for res in results])
        chosen_logits = torch.cat([res['chosen_logits'] for res in results])
        rejected_logits = torch.cat(
            [res['rejected_logits'] for res in results])
        ds_names = list(itertools.chain(*[res['ds_names'] for res in results]))

        # group by ds_names
        grouped_correct = defaultdict(list)
        grouped_chosen_logits = defaultdict(list)
        grouped_rejected_logits = defaultdict(list)
        for i, ds_name in enumerate(ds_names):
            grouped_correct[ds_name].append(correct[i])
            grouped_chosen_logits[ds_name].append(chosen_logits[i])
            grouped_rejected_logits[ds_name].append(rejected_logits[i])

        # print metrics in a rich table
        table = Table(title='Reward Metrics')
        table.add_column('Dataset Name')
        table.add_column('Accuracy')
        table.add_column('Chosen Score')
        table.add_column('Rejected Score')

        for ds_name in grouped_correct.keys():
            correct = torch.stack(grouped_correct[ds_name])
            chosen_logits = torch.stack(grouped_chosen_logits[ds_name])
            rejected_logits = torch.stack(grouped_rejected_logits[ds_name])

            acc = correct.float().mean()
            metrics[f'accuracy/{ds_name}'] = acc.item()
            metrics[f'chosen_score/{ds_name}'] = chosen_logits.mean().item()
            metrics[f'rejected_score{ds_name}'] = rejected_logits.mean().item()

            table.add_row(ds_name, f'{acc:.4f}', f'{chosen_logits.mean():.4f}',
                          f'{rejected_logits.mean():.4f}')

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')
        print_log('\n' + capture.get(), 'current')

        return metrics
