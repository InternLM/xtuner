from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmchat.registry import HOOKS


@HOOKS.register_module()
class SampleGenerateHook(Hook):

    def __init__(self, sample_inputs, every_n_iters=None):
        self.sample_inputs = sample_inputs
        self.every_n_iters = every_n_iters

    def _generate_samples(self, runner):
        runner.logger.info('after_train_iter in SampleGenerateHook.')
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        tokenizer = model.data_preprocessor.tokenizer

        for sample_input in self.sample_inputs:
            inputs = 'Below is an instruction that describes a task. ' \
                'Write a response that appropriately completes the request.' \
                '\n\n ### Instruction:\n{sample_input}\n\n' \
                '### Response: '.format(sample_input=sample_input)
            runner.logger.info(f'sample input: {inputs}')
            input_ids = tokenizer(inputs, return_tensors='pt')['input_ids']
            input_ids = input_ids.to('cuda')
            generation_output = model.llm.generate(
                input_ids=input_ids,
                max_new_tokens=370,
            )
            runner.logger.info(
                f'sample output: {tokenizer.decode(generation_output[0])}')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        if self.every_n_iters is None or (batch_idx +
                                          1) % self.every_n_iters != 0:
            return
        self._generate_samples(runner)

    def after_val(self, runner) -> None:
        if self.every_n_iters is not None:
            return
        self._generate_samples(runner)
