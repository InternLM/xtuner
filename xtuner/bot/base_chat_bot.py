from abc import abstractmethod

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)

from xtuner.model.utils import LoadWoInit
from xtuner.tools.utils import get_stop_criteria


class ChatInstance():

    def __init__(self, bot, bot_name, chat_template, system_template) -> None:

        self.chat_template = chat_template
        self.system_template = system_template
        self.bot_name = bot_name
        self.bot = bot

        self.num_turns = 0
        self.history_text = ''

    def reset_history(self):

        self.num_turns = 0
        self.history_text = ''

    def apply_template(self, text, system=None):

        prompt_text = ''

        if 'SYSTEM' in self.chat_template and self.num_turns == 0:
            system_text = None
            if self.system_template is not None:
                system_text = self.system_template.format(
                    round=self.num_turns + 1, bot_name=self.bot_name)
            elif system is not None:
                system_text = system
            if system_text is not None:
                prompt_text += self.system_template.format(
                    system=system_text,
                    round=self.num_turns + 1,
                    bot_name=self.bot_name)
                prompt_text += self.chat_template['INSTRUCTION'].format(
                    input=text,
                    round=self.num_turns + 1,
                    bot_name=self.bot_name)

        prompt_text += self.chat_template['INSTRUCTION'].format(
            input=text, round=self.num_turns + 1, bot_name=self.bot_name)
        return prompt_text

    def chat(self, text, system=None, generation_config=None):

        templated_text = self.apply_template(text, system)
        self.history_text += templated_text

        output = self.bot.generate(self.history_text, generation_config)
        self.history_text += output

        self.history_text += self.chat_template.get('SEP', '')
        self.num_turns += 1
        return output

    def predict(self, texts, system=None, generation_config=None):

        templated_texts = [self.apply_template(t, system) for t in texts]
        outputs = self.bot.predict(templated_texts, generation_config)
        return outputs


class BaseChatBot():

    def __init__(self, bot_name, chat_template, system_template) -> None:

        self.default_bot_name = bot_name
        self.default_chat_template = chat_template
        self.default_system_template = system_template

    def create_instance(self,
                        bot_name=None,
                        chat_template=None,
                        system_template=None):

        if bot_name is None:
            bot_name = self.default_bot_name

        if chat_template is None:
            chat_template = self.default_chat_template

        if system_template is None:
            system_template = self.default_system_template

        return ChatInstance(self, bot_name, chat_template, system_template)

    @property
    def generation_config(self):
        pass

    @abstractmethod
    def generate(self, inputs, generation_config=None):
        pass

    @abstractmethod
    def predict(self, inputs, generation_config=None, repeat=1):
        pass


class HFChatBot(BaseChatBot):

    def __init__(
        self,
        bot_name,
        model_name_or_path,
        adapter=None,
        bits=None,
        chat_template=None,
        system_template=None,
        max_new_tokens=2048,
        temperature=0.1,
        top_k=40,
        top_p=0.75,
        repetition_penalty=1.0,
        stop_words=[],
    ) -> None:
        super().__init__(bot_name, chat_template, system_template)

        self.llm, self.tokenizer = self.build_components(
            model_name_or_path, adapter, bits)

        self._generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )

        self._stop_words = stop_words + chat_template.get('STOP_WORDS', [])
        self.stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=self._stop_words)

        self.num_history_tokens = 0

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

    def generate(self, text, generation_config=None):

        ids = self.tokenizer.encode(text, return_tensors='pt')

        if generation_config is None:
            generation_config = self.generation_config

        generate_output = self.llm.generate(
            inputs=ids.cuda(),
            generation_config=generation_config,
            stopping_criteria=self.stop_criteria)

        output = self.tokenizer.decode(generate_output[0][len(ids[0]):])

        return output

    def predict(self, texts, generation_config=None, repeat=1):

        outputs = []
        for text in tqdm(texts):
            item = []
            for r in range(repeat):
                item.append(self.generate(text, generation_config))
            outputs.append(item)

        return outputs


class LMDeployChatBot():
    # TODO support tp
    def __init__(
        self,
        bot_name,
        model_name_or_path,
        chat_template=None,
        system_template=None,
        max_new_tokens=2048,
        temperature=0.1,
        top_k=40,
        top_p=0.75,
        repetition_penalty=1.0,
        stop_words=[],
        seed=None,
        use_logn_attn=False,
        use_dynamic_ntk=False,
        rope_scaling_factor=0.0,
        max_batch_size=1,
    ) -> None:
        super().__init__(bot_name, chat_template, system_template)

        stop_words += chat_template.get('STOP_WORDS', [])

        from lmdeploy import pipeline
        from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
        self._generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_words=stop_words,
        )

        backend_config = TurbomindEngineConfig(
            session_len=max_new_tokens,
            max_batch_size=max_batch_size,
            rope_scaling_factor=rope_scaling_factor,
            use_dynamic_ntk=use_dynamic_ntk,
            use_logn_attn=use_logn_attn)
        self.pipeline = pipeline(model_name_or_path, backend_config)

    @property
    def generation_config(self):
        return self._generation_config

    def generate(self, text, generation_config=None):

        if generation_config is None:
            generation_config = self.generation_config

        output = self.pipeline([text], gen_config=generation_config)

        return output

    def predict(self, texts, generation_config=None, repeat=1):

        if generation_config is None:
            generation_config = self.generation_config

        # TODO support repeat > 1
        outputs = self.pipeline(texts, gen_config=generation_config)

        return outputs
