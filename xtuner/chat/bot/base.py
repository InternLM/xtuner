from abc import abstractmethod

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.chat.utils import GenerationConfig
from xtuner.model.utils import LoadWoInit
from xtuner.tools.utils import get_stop_criteria


class BaseBot():

    @property
    def generation_config(self):
        pass

    @abstractmethod
    def generate(self, inputs, generation_config=None):
        pass

    @abstractmethod
    def predict(self, inputs, generation_config=None, repeat=1):
        pass


class HFBot(BaseBot):

    def __init__(
        self,
        model_name_or_path,
        adapter=None,
        bits=None,
    ) -> None:
        super().__init__()

        self.llm, self.tokenizer = self.build_components(
            model_name_or_path, adapter, bits)

        from transformers import GenerationConfig as HFGenerationConfig
        self._generation_config = HFGenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )

    def build_components(self, model_name_or_path, adapter=None, bits=None):

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

    @property
    def generation_config(self):
        return self._generation_config

    def generate(self, text, gen_config: GenerationConfig = None):

        ids = self.tokenizer.encode(text, return_tensors='pt')

        from transformers import GenerationConfig as HFGenerationConfig
        hf_gen_config = HFGenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            do_sample=gen_config.temperature > 0,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
            seed=gen_config.seed,
            eos_token_id=self.generation_config.eos_token_id,
            pad_token_id=self.generation_config.pad_token_id)
        stop_words = gen_config.stop_words
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)

        generate_output = self.llm.generate(
            inputs=ids.cuda(),
            generation_config=hf_gen_config,
            stopping_criteria=stop_criteria)

        output = self.tokenizer.decode(generate_output[0][len(ids[0]):])

        return output

    def predict(self,
                texts,
                generation_config: GenerationConfig = None,
                repeat=1):

        outputs = []
        for text in tqdm(texts):
            item = []
            for r in range(repeat):
                item.append(self.generate(text, generation_config))
            outputs.append(item[0])

        return outputs
