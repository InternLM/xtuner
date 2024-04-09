# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import FlexibleRunner, Runner

from xtuner.types import ChatBackendProtocol


class ChatHook(Hook):

    priority = 'LOW'

    def __init__(self,
                 prompts: Union[str, List[str]],
                 every_n_iters: Optional[int] = None,
                 sample_params: Optional[dict] = None):
        self.evaluation_inputs = prompts
        if isinstance(self.evaluation_inputs, str):
            self.evaluation_inputs = [self.evaluation_inputs]

        self.every_n_iters = every_n_iters
        self.sample_params = sample_params

    def batch_infer(self, runner: Union[Runner, FlexibleRunner],
                    position: str):

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model: ChatBackendProtocol

        responses = model.batch_infer(self.evaluation_inputs,
                                      self.sample_params)

        runner.logger.info('ChatHook Generating...')
        for question, answer in zip(self.evaluation_inputs, responses):
            runner.logger.info(f'(ChatHook {position}){question}')
            runner.logger.info(f'(ChatHook {position}){answer}')

    def before_train(self, runner: Union[Runner, FlexibleRunner]):
        runner.logger.info('before_train in EvaluateChatHook.')
        self.batch_infer(runner, 'before train')

    def after_train_iter(self,
                         runner: Union[Runner, FlexibleRunner],
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        if self.every_n_iters is None:
            return

        if self.every_n_train_iters(runner, self.every_n_iters):
            self.batch_infer(runner, f'after {runner.iter} iter')

    def after_train(self, runner):
        self.batch_infer(runner, 'after train')

    def after_val(self, runner) -> None:
        if self.every_n_iters is not None:
            return
        self.batch_infer(runner, 'after val')
