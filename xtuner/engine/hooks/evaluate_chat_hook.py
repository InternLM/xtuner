# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.utils.misc import get_object_from_string
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)


class EvaluateChatHook(Hook):

    def __init__(self,
                 tokenizer,
                 evaluation_inputs,
                 evaluation_images=None,
                 image_processor=None,
                 system='',
                 prompt_template=None,
                 every_n_iters=None,
                 max_new_tokens=600,
                 stop_word=None):
        self.evaluation_inputs = evaluation_inputs
        if isinstance(self.evaluation_inputs, str):
            self.evaluation_inputs = [self.evaluation_inputs]
        self.evaluation_images = evaluation_images
        if isinstance(self.evaluation_images, str):
            self.evaluation_images = [self.evaluation_images]
        if self.evaluation_images is not None:
            assert len(
                self.evaluation_images) in [1, len(self.evaluation_inputs)]
            if len(self.evaluation_images) == 1:
                self.evaluation_images = [self.evaluation_images[0]] * len(
                    self.evaluation_inputs)
            self.evaluation_images = [
                load_image(img) for img in self.evaluation_images
            ]
        if prompt_template is None:
            instruction = '{input}'
        else:
            if isinstance(prompt_template, str):  # for resume
                prompt_template = get_object_from_string(prompt_template)
            instruction = prompt_template.get('INSTRUCTION', '{input}')
            if system != '':
                system = prompt_template.get(
                    'SYSTEM', '{system}\n').format(system=system)
        self.instruction = instruction
        self.system = system
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        self.tokenizer = BUILDER.build(tokenizer)
        if image_processor is not None:
            self.image_processor = BUILDER.build(image_processor)
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
        model.activation_checkpointing_disable()
        model.llm.config.use_cache = True
        model.eval()
        if self.evaluation_images is not None:
            for sample_image, sample_input in zip(self.evaluation_images,
                                                  self.evaluation_inputs):
                image = expand2square(
                    sample_image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                image = image.to(device)
                sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
                inputs = (self.system + self.instruction).format(
                    input=sample_input, round=1, **runner.cfg)
                chunk_encode = []
                for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                    if idx == 0:
                        cur_encode = self.tokenizer(chunk)
                    else:
                        cur_encode = self.tokenizer(
                            chunk, add_special_tokens=False)
                    chunk_encode.append(cur_encode)
                assert len(chunk_encode) == 2
                input_ids = []
                for idx, cur_chunk_encode in enumerate(chunk_encode):
                    input_ids.extend(cur_chunk_encode['input_ids'])
                    if idx != len(chunk_encode) - 1:
                        input_ids.append(IMAGE_TOKEN_INDEX)
                input_ids = torch.tensor(input_ids).to(device)
                visual_outputs = model.visual_encoder(
                    image.unsqueeze(0), output_hidden_states=True)
                pixel_values = model.projector(visual_outputs.hidden_states[
                    model.visual_select_layer][:, 1:])

                mm_inputs = prepare_inputs_labels_for_multimodal(
                    llm=model.llm,
                    input_ids=input_ids.unsqueeze(0),
                    pixel_values=pixel_values)

                generation_output = model.generate(
                    **mm_inputs,
                    max_new_tokens=max_new_tokens,
                    generation_config=self.gen_config,
                    bos_token_id=self.tokenizer.bos_token_id,
                    stopping_criteria=self.stop_criteria)
                runner.logger.info(
                    f'Sample output:\n'
                    f'{inputs + self.tokenizer.decode(generation_output[0])}\n'
                )
        else:
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
            model.activation_checkpointing_enable()
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
