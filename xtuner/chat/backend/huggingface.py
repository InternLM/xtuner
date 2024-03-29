from typing import List, Optional

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers import GenerationConfig as HFGenerationConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from xtuner.chat.streamer import HFTextIteratorStreamer, HFTextStreamer
from xtuner.model.utils import LoadWoInit
from xtuner.tools.utils import get_stop_criteria
from xtuner.types import ChatMessages, ChatTemplate, SampleParams
from .base import BaseBackend


class HFBackend(BaseBackend):

    def __init__(
        self,
        chat_template: ChatTemplate,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()

        self.llm = llm
        self.llm.cuda()
        self.tokenizer = tokenizer

        self._chat_template = chat_template

    @property
    def chat_template(self) -> ChatTemplate:
        return self._chat_template

    @property
    def eos_token_id(self):
        if self.tokenizer.pad_token_id:
            return self.tokenizer.eos_token_id
        else:
            return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def build_llm_and_tokenizer(self,
                                model_name_or_path,
                                adapter=None,
                                bits=None):

        if bits is None:
            quantization_config = None
            load_in_8bit = False
        elif bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
            load_in_8bit = False
        elif bits == 8:
            quantization_config = None
            load_in_8bit = True

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            encode_special_tokens=True)

        with LoadWoInit():
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map='auto',
                load_in_8bit=load_in_8bit,
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16)

        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapter)

        model.eval()
        return model, tokenizer

    def create_streamer(self, iterable=False):
        if iterable:
            return HFTextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                chat_template=self.chat_template)
        else:
            return HFTextStreamer(
                self.tokenizer,
                skip_prompt=True,
                chat_template=self.chat_template)

    def parse_sample_params(self, params: SampleParams) -> HFGenerationConfig:

        if params is None:
            params = SampleParams()

        hf_gen_config = HFGenerationConfig(
            max_new_tokens=params.max_new_tokens,
            do_sample=params.temperature > 0,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            seed=params.seed,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id)

        stop_words = params.stop_words
        stop_words.extend(self.chat_template.stop_words)

        return hf_gen_config, stop_words

    def chat(self,
             messages: ChatMessages,
             streamer=None,
             sample_params: Optional[SampleParams] = None):

        prompt = messages.get_prompt(self.chat_template)
        ids = self.tokenizer.encode(prompt, return_tensors='pt')

        hf_gen_config, stop_words = self.parse_sample_params(sample_params)

        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)

        generate_output = self.llm.generate(
            inputs=ids.cuda(),
            streamer=streamer,
            generation_config=hf_gen_config,
            stopping_criteria=stop_criteria)

        output = self.tokenizer.decode(
            generate_output[0][len(ids[0]):], skip_special_tokens=True)

        for word in stop_words:
            output = output.rstrip(word)

        return output

    def batch_infer(self,
                    messages: List[ChatMessages],
                    sample_params: SampleParams | None = None):
        raise NotImplementedError
