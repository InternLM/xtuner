from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from transformers import StoppingCriteriaList

from mmchat.registry import HOOKS, TOKENIZER
from mmchat.utils import StopWordStoppingCriteria


@HOOKS.register_module()
class SampleGenerateHook(Hook):

    def __init__(self,
                 tokenizer,
                 sample_inputs,
                 instruction,
                 every_n_iters=None,
                 max_new_tokens=600,
                 stop_word=None):
        self.sample_inputs = sample_inputs
        self.instruction = instruction
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        self.tokenizer = TOKENIZER.build(tokenizer)
        self.stop_criteria = StoppingCriteriaList()
        if stop_word is not None:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, stop_word))

    def _generate_samples(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        for sample_input in self.sample_inputs:
            inputs = self.instruction.format(input=sample_input, **runner.cfg)
            input_ids = self.tokenizer(
                inputs, return_tensors='pt')['input_ids']
            input_ids = input_ids.to(device)
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=self.stop_criteria)
            runner.logger.info(
                f'Sample output:\n'
                f'{self.tokenizer.decode(generation_output[0])}\n')

    def before_train(self, runner):
        runner.logger.info('before_train in SampleGenerateHook.')
        self._generate_samples(runner)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        if self.every_n_iters is None or (batch_idx +
                                          1) % self.every_n_iters != 0:
            return
        runner.logger.info('after_train_iter in SampleGenerateHook.')
        self._generate_samples(runner)

    def after_val(self, runner) -> None:
        if self.every_n_iters is not None:
            return
        runner.logger.info('after_val in SampleGenerateHook.')
        self._generate_samples(runner)
