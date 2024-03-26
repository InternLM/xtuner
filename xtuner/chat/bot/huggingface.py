import os
import os.path as osp

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)

from xtuner.chat.streamer import HFTextIteratorStreamer, HFTextStreamer
from xtuner.chat.utils import GenerationConfig
from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from .base import BaseBot


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

    def create_streamer(self, chat_template=None, iterable=False):
        if iterable:
            return HFTextIteratorStreamer(
                self.tokenizer, skip_prompt=True, chat_template=chat_template)
        else:
            return HFTextStreamer(
                self.tokenizer, skip_prompt=True, chat_template=chat_template)

    @property
    def generation_config(self):
        return self._generation_config

    def generate(self,
                 text,
                 streamer=None,
                 gen_config: GenerationConfig = None):

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
            streamer=streamer,
            generation_config=hf_gen_config,
            stopping_criteria=stop_criteria)

        output = self.tokenizer.decode(
            generate_output[0][len(ids[0]):], skip_special_tokens=True)

        for word in stop_words:
            output = output.rstrip(word)

        return output

    def predict(self,
                texts,
                generation_config: GenerationConfig = None,
                repeat=1):

        outputs = []
        for text in tqdm(texts):
            item = []
            for r in range(repeat):
                item.append(self.generate(text, None, generation_config))
            outputs.append(item[0])

        return outputs


class HFLlavaBot(BaseBot):

    def __init__(
        self,
        model_name_or_path,
        llava_name_or_path,
        visual_encoder_name_or_path=None,
        visual_select_layer=-2,
        bits=None,
    ) -> None:

        components = self.build_components(model_name_or_path,
                                           llava_name_or_path,
                                           visual_encoder_name_or_path, bits)
        llm, tokenizer, visual_encoder, image_processor, projector = components
        self.llm = llm
        self.tokenizer = tokenizer
        self.visual_encoder = visual_encoder
        self.image_processor = image_processor
        self.visual_select_layer = visual_select_layer
        self.projector = projector

        from transformers import GenerationConfig as HFGenerationConfig
        self._generation_config = HFGenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )

    def create_streamer(self, chat_template=None, iterable=False):
        if iterable:
            return HFTextIteratorStreamer(
                self.tokenizer, skip_prompt=True, chat_template=chat_template)
        else:
            return HFTextStreamer(
                self.tokenizer, skip_prompt=True, chat_template=chat_template)

    @property
    def generation_config(self):
        return self._generation_config

    def build_components(self, model_name_or_path, llava_name_or_path,
                         visual_encoder_name_or_path, bits):

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

        if 'visual_encoder' in os.listdir(llava_path):
            assert visual_encoder_name_or_path is None
            visual_encoder_path = osp.join(llava_path, 'visual_encoder')
        else:
            visual_encoder_path = visual_encoder_name_or_path

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

        # build projector
        projector_path = osp.join(llava_path, 'projector')
        projector = AutoModel.from_pretrained(
            projector_path, torch_dtype=torch.float16, trust_remote_code=True)
        print(f'Load projector from {llava_name_or_path}')

        projector.cuda()
        projector.eval()
        visual_encoder.cuda()
        visual_encoder.eval()

        if bits is None:
            llm.cuda()

        llm.eval()

        return llm, tokenizer, visual_encoder, image_processor, projector

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

    def generate(self, inputs, streamer=None, gen_config=None):

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
            **inputs,
            streamer=streamer,
            generation_config=hf_gen_config,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=stop_criteria)

        output = self.tokenizer.decode(
            generate_output[0], skip_special_tokens=True)

        for word in stop_words:
            output = output.rstrip(word)

        return output
