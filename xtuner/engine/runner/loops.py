# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmengine.runner import IterBasedTrainLoop
from mmengine.runner import ValLoop as MMENGINE_ValLoop
from torch.utils.data import DataLoader
from typing import Sequence
from mmengine.dist import broadcast_object_list, is_main_process, get_world_size, get_rank, barrier, collect_results
import math
import torch
from mmengine.model import is_model_wrapper

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


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
    def __init__(self, runner, dataloader, evaluator=None, torch_dtype='fp16', select_metric='first') -> None:
        # must be concatset
        super(MMENGINE_ValLoop, self).__init__(runner, dataloader)
        self.collate_fn = self.dataloader.collate_fn
        self._runner = runner
        self.torch_dtype = torch_dtype
        if torch_dtype is not None:
            self.torch_dtype = TORCH_DTYPE_MAP[torch_dtype]
        self.select_metric = select_metric

    def run(self) -> dict:
        """Launch validation."""
        self.runner.logger.info('==================== Start val loop ===================')
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')

        if is_model_wrapper(self.runner.model):
            model = self.runner.model.module
        else:
            model = self.runner.model

        model.gradient_checkpointing_disable()
        model.eval()

        rank = get_rank()
        metrics = []
        # Ensure that eta and log are displayed correctly.
        current_run_total_ids = 0
        for _, dataset in enumerate(self.dataloader.dataset.datasets):
            model.preparing_for_generation(dataset.metainfo)

            results = []
            n_samples = len(dataset)
            per_rank_samples = math.ceil(n_samples / get_world_size())
            per_rank_ids = range(per_rank_samples * rank,
                                 min(n_samples, per_rank_samples * (rank + 1)))
            for idx in per_rank_ids:
                data_batch = dataset[idx]
                # TODO: Only bs=1 is currently supported temporarily
                data_batch = self.collate_fn([data_batch])
                self.run_iter(current_run_total_ids, data_batch, results)
                current_run_total_ids += 1

            barrier()
            self.runner.logger.info('==================== Start collect results ===================')
            results = collect_results(results, len(dataset))
            self.runner.logger.info('========= Starting the evaluation of a data ===========')
            if is_main_process():
                metric = dataset.evaluate(results, self.runner.work_dir)
                objects = [metric]
            else:
                objects = [None]
            broadcast_object_list(objects)
            metric = objects[0]
            metrics.append(metric)

        # select metrics
        if self.select_metric == 'first':
            metrics = metrics[0]
        else:
            raise NotImplementedError

        self.runner.logger.info('================ Ending val loop ================')
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        model.gradient_checkpointing_enable()
        model.train()
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], results: list):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        assert 'img_id' in data_batch['data'], 'img_id is required in data_batch. ' \
                                               'The __getitem__ function in the dataset must ' \
                                               'return a dictionary with the img_id.'
        prediction = {'img_id': data_batch['data']['img_id'][0]}

        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)

        # outputs should be sequence of BaseDataElement
        outputs = self.runner.model.val_step(data_batch)
        prediction.update(outputs)
        results.append(prediction)

        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


class TestLoop(ValLoop):
    def run(self) -> dict:
        """Launch validation."""
        self.runner.logger.info('==================== Start test loop ===================')
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')

        if is_model_wrapper(self.runner.model):
            model = self.runner.model.module
        else:
            model = self.runner.model

        model.gradient_checkpointing_disable()
        model.eval()

        if self.torch_dtype is not None:
            self.runner.logger.info(f'Convert model dtype to {self.torch_dtype}')
            model.to(self.torch_dtype)

        rank = get_rank()
        metrics = []
        # Ensure that eta and log are displayed correctly.
        current_run_total_ids = 0
        for _, dataset in enumerate(self.dataloader.dataset.datasets):
            model.preparing_for_generation(dataset.metainfo)

            results = []
            n_samples = len(dataset)
            per_rank_samples = math.ceil(n_samples / get_world_size())
            per_rank_ids = range(per_rank_samples * rank,
                                 min(n_samples, per_rank_samples * (rank + 1)))
            for idx in per_rank_ids:
                data_batch = dataset[idx]
                # TODO: Only bs=1 is currently supported temporarily
                data_batch = self.collate_fn([data_batch])
                self.run_iter(current_run_total_ids, data_batch, results)
                current_run_total_ids += 1

            barrier()
            self.runner.logger.info('==================== Start collect results ===================')
            results = collect_results(results, len(dataset))
            self.runner.logger.info('========= Starting the evaluation of a data ===========')

            if is_main_process():
                metric = dataset.evaluate(results, self.runner.work_dir)
                objects = [metric]
            else:
                objects = [None]
            broadcast_object_list(objects)
            metric = objects[0]
            metrics.append(metric)

        # select metrics
        if self.select_metric == 'first':
            metrics = metrics[0]
        else:
            raise NotImplementedError
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        self.runner.logger.info('================ Ending test loop ================')
        # model.gradient_checkpointing_enable()
        # model.train()
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], results: list):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        assert 'img_id' in data_batch['data'], 'img_id is required in data_batch. ' \
                                               'The __getitem__ function in the dataset must ' \
                                               'return a dictionary with the img_id.'
        prediction = {'img_id': data_batch['data']['img_id'][0]}

        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        # outputs should be sequence of BaseDataElement
        outputs = self.runner.model.val_step(data_batch)
        prediction.update(outputs)
        results.append(prediction)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
