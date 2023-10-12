# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria


class EvaluateChatHook(Hook):

    def __init__(self,
                 tokenizer,
                 evaluation_inputs,
                 system='',
                 prompt_template=None,
                 every_n_iters=None,
                 max_new_tokens=600,
                 stop_word=None):
        self.evaluation_inputs = evaluation_inputs
        if isinstance(self.evaluation_inputs, str):
            self.evaluation_inputs = [self.evaluation_inputs]
        if prompt_template is None:
            instruction = '{input}'
        else:
            instruction = prompt_template.get('INSTRUCTION', '{input}')
            if system != '':
                system = prompt_template.get(
                    'SYSTEM', '{system}\n').format(system=system)
        self.instruction = instruction
        self.system = system
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        self.tokenizer = BUILDER.build(tokenizer)
        self.stop_criteria = StoppingCriteriaList()
        # default generation config
        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )
        if stop_word is not None:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, stop_word))

    def _generate_samples(self, runner, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        is_checkpointing = model.llm.is_gradient_checkpointing
        use_cache = model.llm.config.use_cache

        # Cast to inference mode
        model.llm.gradient_checkpointing_disable()
        model.llm.config.use_cache = True
        model.eval()

        for sample_input in self.evaluation_inputs:
            inputs = (self.system + self.instruction).format(
                input=sample_input, round=1, **runner.cfg)
            input_ids = self.tokenizer(
                inputs, return_tensors='pt')['input_ids']
            input_ids = input_ids.to(device)
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                generation_config=self.gen_config,
                stopping_criteria=self.stop_criteria)
            runner.logger.info(
                f'Sample output:\n'
                f'{self.tokenizer.decode(generation_output[0])}\n')

        # Cast to training mode
        if is_checkpointing:
            model.llm.gradient_checkpointing_enable()
        model.llm.config.use_cache = use_cache
        model.train()

    def before_train(self, runner):
        runner.logger.info('before_train in EvaluateChatHook.')
        self._generate_samples(runner, max_new_tokens=50)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        if self.every_n_iters is None or (batch_idx +
                                          1) % self.every_n_iters != 0:
            return
        runner.logger.info('after_train_iter in EvaluateChatHook.')
        self._generate_samples(runner)

    def after_train(self, runner):
        runner.logger.info('after_train in EvaluateChatHook.')
        self._generate_samples(runner)

    def after_val(self, runner) -> None:
        if self.every_n_iters is not None:
            return
        runner.logger.info('after_val in EvaluateChatHook.')
        self._generate_samples(runner)
