from typing import Optional

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers import GenerationConfig as HFGenerationConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from xtuner.chat.streamer import HFTextIteratorStreamer, HFTextStreamer
from xtuner.model.utils import LoadWoInit
from xtuner.tools.utils import get_stop_criteria
from xtuner.types import HybridChatMessages, HybridChatTemplate, SampleParams
from .base import BaseBackend


class _HFBackend(BaseBackend):

    def __init__(
        self,
        chat_template: HybridChatTemplate,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()

        self.llm = llm
        self.llm.cuda()
        self.tokenizer = tokenizer

        self._chat_template = chat_template

    @property
    def chat_template(self) -> HybridChatTemplate:
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

    def response_with_code_interpreter(self, response: str):
        return False

    def response_with_function_call(self, response: str):
        return False

    def create_streamer(self, chat_template=None, iterable=False):
        if iterable:
            return HFTextIteratorStreamer(
                self.tokenizer, skip_prompt=True, chat_template=chat_template)
        else:
            return HFTextStreamer(
                self.tokenizer, skip_prompt=True, chat_template=chat_template)

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
             messages: HybridChatMessages,
             streamer=None,
             sample_params: Optional[SampleParams] = None):

        prompt = messages.apply_chat_template(self.chat_template)
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


class HFBackend(_HFBackend):

    def __init__(
        self,
        chat_template: HybridChatTemplate,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        vision_tower: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(chat_template, llm, tokenizer)

        if vision_tower:
            self.vision_tower = vision_tower
            self.vision_tower.cuda()
            self.vision_tower.eval()
        else:
            self.vision_tower = None

    def chat(self,
             messages: HybridChatMessages,
             streamer=None,
             sample_params=None):

        img_urls = messages.collect_img_urls()

        if self.vision_tower is None or len(img_urls) == 0:
            return super().chat(messages, streamer, sample_params)

        prompt = messages.apply_chat_template(self.chat_template)

        img_features = self.vision_tower(img_urls)

        # prompt, img_ranges = _insert_img_pad_tokens(
        #     prompt, self.chat_template.image_token, img_features,
        #     self.tokenizer.pad_token
        # )

        chunks = prompt.split(self.chat_template.image_token)
        assert len(chunks) - 1 == len(img_urls)
        chunk_embeddings = []
        for i in range(len(chunks)):

            chunk_ids = self.tokenizer.encode(chunks[i], return_tensors='pt')
            chunk_ids = chunk_ids.to(self.llm.device)
            chunk_emb = self.llm.get_input_embeddings()(chunk_ids)
            chunk_embeddings.append(chunk_emb)

            if i < len(chunks) - 1:
                chunk_embeddings.append(img_features[i].unsqueeze(0))

        embeddings = torch.cat(chunk_embeddings, dim=1)

        hf_gen_config, stop_words = self.parse_sample_params(sample_params)

        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)

        generate_output = self.llm.generate(
            input_ids=None,
            inputs_embeds=embeddings,
            streamer=streamer,
            generation_config=hf_gen_config,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=stop_criteria)

        output = self.tokenizer.decode(
            generate_output[0], skip_special_tokens=True)

        for word in stop_words:
            output = output.rstrip(word)

        return output
