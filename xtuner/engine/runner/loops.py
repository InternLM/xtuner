# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmengine.runner import IterBasedTrainLoop
from mmengine.runner import ValLoop as MMENGINE_ValLoop
from mmengine.runner import TestLoop as MMENGINE_TestLoop
from torch.utils.data import DataLoader
from typing import Sequence
from mmengine.dist import broadcast_object_list, is_main_process, get_world_size, get_rank,barrier, collect_results
from xtuner.registry import BUILDER
import math
from tqdm import tqdm
import torch
from mmengine.runner.amp import autocast


class TrainLoop(IterBasedTrainLoop):

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_iters: Optional[int] = None,
                 max_epochs: Union[int, float] = None,
                 **kwargs) -> None:

        if max_iters is None and max_epochs is None:
            raise RuntimeError('Please specify the `max_iters` or '
                               '`max_epochs` in `train_cfg`.')
        elif max_iters is not None and max_epochs is not None:
            raise RuntimeError('Only one of `max_iters` or `max_epochs` can '
                               'exist in `train_cfg`.')
        else:
            if max_iters is not None:
                iters = int(max_iters)
                assert iters == max_iters, ('`max_iters` should be a integer '
                                            f'number, but get {max_iters}')
            elif max_epochs is not None:
                if isinstance(dataloader, dict):
                    diff_rank_seed = runner._randomness_cfg.get(
                        'diff_rank_seed', False)
                    dataloader = runner.build_dataloader(
                        dataloader,
                        seed=runner.seed,
                        diff_rank_seed=diff_rank_seed)
                iters = max_epochs * len(dataloader)
            else:
                raise NotImplementedError
        super().__init__(
            runner=runner, dataloader=dataloader, max_iters=iters, **kwargs)


class ValLoop(MMENGINE_ValLoop):
    def __init__(self, runner, dataloader=None, evaluator=None, fp16: bool = False, select_metric='first') -> None:
        # must be concatset
        super(MMENGINE_ValLoop, self).__init__(runner, dataloader)
        self._runner = runner
        self.fp16 = fp16
        self.select_metric = select_metric

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.gradient_checkpointing_disable()
        self.runner.model.eval()

        rank = get_rank()
        metrics = []
        for _, dataset in enumerate(self.dataloader.datasets):
            self.runner.model.preparing_for_generation(dataset.metainfo)

            results = []
            n_samples = len(dataset)
            per_rank_samples = math.ceil(n_samples / get_world_size())
            per_rank_ids = range(per_rank_samples * rank,
                                 min(n_samples, per_rank_samples * (rank + 1)))
            for idx in tqdm(per_rank_ids, desc=f'Rank {rank}'):
                data_batch = dataset[idx]
                self.run_iter(idx, data_batch, results)

            barrier()
            results = collect_results(results, len(dataset))

            if is_main_process():
                metric = dataset.evaluate(results, self.runner.work_dir)
                objects = [metric]
            else:
                objects = [None]
            broadcast_object_list(objects)
            metric = objects[0]
            metrics.append(metric)
            del dataset

        # select metrics
        if self.select_metric == 'first':
            metrics = metrics[0]
        else:
            raise NotImplementedError

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        self.runner.model.gradient_checkpointing_enable()
        self.runner.model.train()
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], results: list):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        assert 'img_id' in data_batch, 'img_id is required in data_batch. ' \
                                       'The __getitem__ function in the dataset must ' \
                                       'return a dictionary with the img_id.'
        prediction = {'img_id': data_batch['img_id']}

        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)

        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step({'data': data_batch})
        prediction['prediction'] = outputs['prediction']
        results.append(prediction)

        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


class TestLoop(ValLoop):
    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.gradient_checkpointing_disable()
        self.runner.model.eval()

        rank = get_rank()
        metrics = []
        for _, dataset_cfg in enumerate(self.datasets):
            dataset = self._build_dataset(dataset_cfg)
            assert len(dataset) > 0, 'The dataset is empty'

            self.runner.model.preparing_for_generation(dataset.metainfo)

            results = []
            n_samples = len(dataset)
            per_rank_samples = math.ceil(n_samples / get_world_size())
            per_rank_ids = range(per_rank_samples * rank,
                                 min(n_samples, per_rank_samples * (rank + 1)))
            for idx in tqdm(per_rank_ids, desc=f'Rank {rank}'):
                data_batch = dataset[idx]
                self.run_iter(idx, data_batch, results)

            barrier()
            results = collect_results(results, len(dataset))

            if is_main_process():
                metric = dataset.evaluate(results, self.runner.work_dir)
                objects = [metric]
            else:
                objects = [None]
            broadcast_object_list(objects)
            metric = objects[0]
            metrics.append(metric)
            del dataset

        # select metrics
        if self.select_metric == 'first':
            metrics = metrics[0]
        else:
            raise NotImplementedError
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')

        self.runner.model.gradient_checkpointing_enable()
        self.runner.model.train()
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], results: list):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        assert 'img_id' in data_batch, 'img_id is required in data_batch. ' \
                                       'The __getitem__ function in the dataset must ' \
                                       'return a dictionary with the img_id.'
        prediction = {'img_id': data_batch['img_id']}

        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step({'data': data_batch})
        prediction.update(outputs)
        results.append(prediction)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
