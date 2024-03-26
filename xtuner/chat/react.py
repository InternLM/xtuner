import os

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig


class HFReActBot():

    def __init__(
        self,
        model_name_or_path,
        adapter=None,
        bits=None,
    ) -> None:

        from lagent.actions import ActionExecutor, GoogleSearch
        from lagent.agents import (CALL_PROTOCOL_CN, FORCE_STOP_PROMPT_CN,
                                   ReAct, ReActProtocol)

        try:
            SERPER_API_KEY = os.environ['SERPER_API_KEY']
        except Exception:
            print('Please obtain the `SERPER_API_KEY` from https://serper.dev '
                  'and set it using `export SERPER_API_KEY=xxx`.')
        search_tool = GoogleSearch(api_key=SERPER_API_KEY)

        llm = self.build_llm(model_name_or_path, adapter, bits)

        self.react = ReAct(
            llm=llm,
            action_executor=ActionExecutor(actions=[search_tool]),
            protocol=ReActProtocol(
                call_protocol=CALL_PROTOCOL_CN,
                force_stop=FORCE_STOP_PROMPT_CN))

    def build_llm(self, model_name_or_path, adapter=None, bits=None):

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

        model_kwargs = dict(
            device_map='auto',
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16)

        from lagent.llms import HFTransformerCasualLM
        model = HFTransformerCasualLM(model_name_or_path, model_kwargs)

        if adapter is not None:
            model.model = PeftModel.from_pretrained(model.model, adapter)

        return model

    def chat(self, text, system=None, generation_config=None):

        return self.react.chat(text).response
