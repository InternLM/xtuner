# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook

from xtuner.registry import BUILDER


class DatasetInfoHook(Hook):

    def __init__(self, tokenizer):
        self.tokenizer = BUILDER.build(tokenizer)

    def log(self, runner, dataset, mode='train'):
        runner.logger.info(f'Num {mode} samples {len(dataset)}')
        runner.logger.info(f'{mode} example:')
        input_ids = [abs(x) for x in dataset[0]['input_ids']]
        runner.logger.info(self.tokenizer.decode(input_ids))

    def before_train(self, runner) -> None:
        do_train = runner.train_loop is not None
        do_eval = runner.val_loop is not None
        do_test = runner.test_loop is not None
        if do_train:
            train_dataset = runner.train_dataloader.dataset
            self.log(runner, train_dataset, mode='train')
        if do_eval:
            eval_dataset = runner.val_dataloader.dataset
            self.log(runner, eval_dataset, mode='eval')
        if do_test:
            test_dataset = runner.test_dataloader.dataset
            self.log(runner, test_dataset, mode='test')

    def before_val(self, runner) -> None:
        eval_dataset = runner.val_dataloader.dataset
        self.log(runner, eval_dataset, mode='eval')

    def before_test(self, runner) -> None:
        test_dataset = runner.test_dataloader.dataset
        self.log(runner, test_dataset, mode='test')
