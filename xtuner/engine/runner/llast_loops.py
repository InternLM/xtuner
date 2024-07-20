# Copyright (c) LLaST. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence

import torch
from mmengine.evaluator import Evaluator
from mmengine.model import is_model_wrapper
from mmengine.registry import LOOPS
from mmengine.runner import TestLoop, autocast
from mmengine.utils.misc import get_object_from_string
from torch.utils.data import DataLoader
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.dataset.llast import prepare_inputs_labels_for_llast
from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria


def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria


@LOOPS.register_module()
class LLaSTTestLoop(TestLoop):

    def __init__(self,
                 runner,
                 tokenizer,
                 dataloader: DataLoader | Dict,
                 evaluator: Evaluator | Dict | List,
                 fp16: bool = False,
                 system='',
                 prompt_template=None,
                 max_new_tokens=256,
                 num_beams=1,
                 do_sample=True,
                 stop_word=None,
                 end_str=None):
        super().__init__(runner, dataloader, evaluator, fp16)
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
        self.max_new_tokens = max_new_tokens
        self.tokenizer = BUILDER.build(tokenizer)
        # default generation config
        self.gen_config = GenerationConfig(
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )
        if stop_word is not None:
            self.stop_criteria = get_stop_criteria(
                tokenizer=self.tokenizer, stop_words=stop_word)
        else:
            self.stop_criteria = StoppingCriteriaList()
        self.end_str = end_str

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        if self.fp16:
            if is_model_wrapper(self.runner.model):
                self.runner.model.module = self.runner.model.module.to(
                    torch.float16)
            else:
                self.runner.model = self.runner.model.to(torch.float16)

        with autocast(enabled=self.fp16, dtype=torch.float16):
            if is_model_wrapper(self.runner.model):
                data_preprocessor = self.runner.model.module.data_preprocessor  # noqa: E501
                audio_data_dtype = self.runner.model.module.speech_encoder.encoder.conv1.weight.dtype  # noqa: E501
                llm_data_dtype = self.runner.model.module.projector.model[
                    0].weight.dtype
                projector = self.runner.model.module.projector
                speech_encoder = self.runner.model.module.speech_encoder
                llm = self.runner.model.module.llm
                generate_fn = self.runner.model.module.generate
                decoder_start_token_id = self.runner.model.module.speech_encoder.config.decoder_start_token_id  # noqa: E501
            else:
                data_preprocessor = self.runner.model.data_preprocessor
                audio_data_dtype = self.runner.model.speech_encoder.encoder.conv1.weight.dtype  # noqa: E501
                llm_data_dtype = self.runner.model.projector.model[
                    0].weight.dtype
                speech_encoder = self.runner.model.speech_encoder
                projector = self.runner.model.projector
                llm = self.runner.model.llm
                generate_fn = self.runner.model.generate
                decoder_start_token_id = self.runner.model.speech_encoder.config.decoder_start_token_id  # noqa: E501

            data_batch = data_preprocessor(data_batch, False)

            data = data_batch['data']
            data['audio_tokens'] = data['audio_tokens'].to(audio_data_dtype)
            batch_size = data['audio_tokens'].shape[0]
            decoder_input_ids = torch.tensor([[1] * batch_size
                                              ]) * decoder_start_token_id
            audio_outputs = speech_encoder(
                data['audio_tokens'],
                decoder_input_ids=decoder_input_ids.to(
                    data['audio_tokens'].device),
                output_hidden_states=True).encoder_last_hidden_state

            audio_outputs = audio_outputs.to(llm_data_dtype)
            audio_outputs = audio_outputs[:, :max(data['audio_lens']), :]

            audio_tokens = projector(audio_outputs)
            data['audio_tokens'] = audio_tokens

            mm_inputs = prepare_inputs_labels_for_llast(
                llm=llm,
                input_ids=data['input_ids'],
                audio_lens=data['audio_lens'],
                audio_tokens=audio_tokens)

            # dtype = self.runner.model.llm.dtype
            mm_inputs['inputs_embeds'] = mm_inputs['inputs_embeds'].to(
                llm_data_dtype)
            generation_output = generate_fn(
                **mm_inputs,
                max_new_tokens=self.max_new_tokens,
                generation_config=self.gen_config,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria)

            generations = self.tokenizer.batch_decode(
                generation_output, skip_special_tokens=True)

            if self.end_str:
                generations = [
                    item.split(self.end_str)[0] for item in generations
                ]

        self.evaluator.process(data_samples=generations, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=generations)
