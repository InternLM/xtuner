from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmchat.registry import HOOKS, TOKENIZER


@HOOKS.register_module()
class SampleGenerateHook(Hook):

    def __init__(self,
                 tokenizer,
                 sample_inputs,
                 instruction,
                 every_n_iters=None,
                 max_new_tokens=600):
        self.sample_inputs = sample_inputs
        self.instruction = instruction
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        self.tokenizer = TOKENIZER.build(tokenizer)

    def _generate_samples(self, runner):
        runner.logger.info('after_train_iter in SampleGenerateHook.')
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        for sample_input in self.sample_inputs:
            inputs = self.instruction.format(input=sample_input)
            input_ids = self.tokenizer(
                inputs, return_tensors='pt')['input_ids']
            input_ids = input_ids.to(device)
            generation_output = model.llm.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
            )
            runner.logger.info(
                f'Sample output:\n'
                f'{self.tokenizer.decode(generation_output[0])}\n')

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
