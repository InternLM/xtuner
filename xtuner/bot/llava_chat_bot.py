import os
import os.path as osp

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from .base_chat_bot import BaseChatBot, ChatInstance


class LlavaInstance(ChatInstance):

    def __init__(self, bot, bot_name, chat_template, system_template,
                 image) -> None:
        super().__init__(bot, bot_name, chat_template, system_template)

        self.pixel_values = self.bot.process_img(image)
        self.history_text += DEFAULT_IMAGE_TOKEN + '\n'

    def reset_history(self):

        self.num_turns = 0
        self.history_text = DEFAULT_IMAGE_TOKEN + '\n'

    def chat(self, text, system=None, generation_config=None):

        templated_text = self.apply_template(text, system)
        self.history_text += templated_text

        inputs = self.prepare_inputs(self.history_text, self.pixel_values,
                                     self.num_turns)
        output = self.bot.generate(inputs, generation_config)
        self.history_text += output


class HFLlavaBot(BaseChatBot):

    def __init__(
        self,
        bot_name,
        model_name_or_path,
        llava_name_or_path,
        visual_select_layer=-2,
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

        components = self.build_components(model_name_or_path,
                                           llava_name_or_path, bits)
        llm, tokenizer, visual_encoder, image_processor = components
        self.llm = llm
        self.tokenizer = tokenizer
        self.visual_encoder = visual_encoder
        self.image_processor = image_processor
        self.visual_select_layer = visual_select_layer

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

        return LlavaInstance(self, bot_name, chat_template, system_template)

    @property
    def generation_config(self):
        return self._generation_config

    def build_components(self, model_name_or_path, llava_name_or_path, bits):

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

        llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16)

        llava_path = snapshot_download(
            repo_id=llava_name_or_path
        ) if not osp.isdir(llava_name_or_path) else llava_name_or_path

        assert 'visual_encoder' in os.listdir(llava_path)
        visual_encoder_path = osp.join(llava_path, 'visual_encoder')
        visual_encoder = CLIPVisionModel.from_pretrained(
            visual_encoder_path, torch_dtype=torch.float16)
        image_processor = CLIPImageProcessor.from_pretrained(
            visual_encoder_path)

        # load adapter
        if 'llm_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'llm_adapter')
            llm = PeftModel.from_pretrained(llm, adapter_path)
            print(f'Load LLM adapter from {llava_name_or_path}')
        if 'visual_encoder_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
            visual_encoder = PeftModel.from_pretrained(visual_encoder,
                                                       adapter_path)
            print(f'Load visual_encoder adapter from {llava_name_or_path}')

        return llm, tokenizer, visual_encoder, image_processor

    def process_img(self, image):
        image = load_image(image)
        image = expand2square(
            image,
            tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]

        image = image.cuda().unsqueeze(0)

        visual_outputs = self.visual_encoder(image, output_hidden_states=True)
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

        return pixel_values

    def prepare_inputs(self, text, pixel_values, n_turn):
        chunk_encode = []
        for idx, chunk in enumerate(text.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0 and n_turn == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(
                    chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=ids, pixel_values=pixel_values)

        return mm_inputs

    def generate(self, inputs, generation_config=None):

        if generation_config is None:
            generation_config = self.generation_config

        generate_output = self.llm.generate(
            **inputs,
            generation_config=generation_config,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)

        output = self.tokenizer.decode(generate_output[0])

        return output
