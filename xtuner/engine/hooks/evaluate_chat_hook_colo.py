# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria

import torch
import torch.distributed as dist
from typing import Any, Callable, Optional
from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers import PreTrainedTokenizer
import copy


def _prepare_logits_processor(
    top_k: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        processor_list.append(TopKLogitsWarper(top_k))
    if top_p is not None and top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    return processor_list


def _is_sequence_finished(unfinished_sequences: torch.Tensor) -> bool:
    if dist.is_initialized() and dist.get_world_size() > 1:
        # consider DP
        unfinished_sequences = unfinished_sequences.clone()
        # dist.broadcast(unfinished_sequences, 0)
        dist.all_reduce(unfinished_sequences)
    return unfinished_sequences.max() == 0


def _sample(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    # early_stopping: bool = False,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
    update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
    **model_kwargs,
) -> torch.Tensor:
    max_length = input_ids.size(1) + max_new_tokens

    logits_processor = _prepare_logits_processor(top_k, top_p, temperature)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    for _ in range(input_ids.size(1), max_length):
        model_inputs = (
            prepare_inputs_fn(input_ids, **model_kwargs) if prepare_inputs_fn is not None else {"input_ids": input_ids}
        )
        outputs = model(data=model_inputs, mode='tensor')
        # outputs = model(**model_inputs)

        # NOTE: this is correct only in left padding mode
        next_token_logits = outputs["logits"][:, -1, :]
        next_token_logits = logits_processor(input_ids, next_token_logits)
        # sample
        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if update_model_kwargs_fn is not None:
            model_kwargs = update_model_kwargs_fn(outputs, model_kwargs)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        if _is_sequence_finished(unfinished_sequences):
            break

    return input_ids


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
    update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
    **kwargs,
) -> torch.Tensor:
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    generation_config.validate()
    is_greedy_gen_mode = (generation_config.num_beams == 1) and generation_config.do_sample is False
    is_sample_gen_mode = (generation_config.num_beams == 1) and generation_config.do_sample is True
    is_beam_gen_mode = (generation_config.num_beams > 1) and generation_config.do_sample is False
    if is_greedy_gen_mode:
        # run greedy search
        raise NotImplementedError
    elif is_sample_gen_mode:
        # run sample
        return _sample(
            model,
            input_ids,
            generation_config.max_new_tokens,
            early_stopping=generation_config.early_stopping,
            eos_token_id=generation_config.eos_token_id,
            pad_token_id=generation_config.pad_token_id,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
            prepare_inputs_fn=prepare_inputs_fn,
            update_model_kwargs_fn=update_model_kwargs_fn,
            **model_kwargs,
        )
    elif is_beam_gen_mode:
        raise NotImplementedError
    else:
        raise ValueError("Unsupported generation mode")


class EvaluateChatHookColossalAI(Hook):

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
            # generation_output = generate(
            #     runner.model.model_wrapper, input_ids=input_ids, generation_config=self.gen_config, max_new_tokens=max_new_tokens
            # )
            generation_output = generate(
                runner.model, input_ids=input_ids, generation_config=self.gen_config, max_new_tokens=max_new_tokens
            )
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
