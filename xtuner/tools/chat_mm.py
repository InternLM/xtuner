# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.dataset.utils import expand2square, load_image
from xtuner.model import MLPProjector
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_chat_utils
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')
    parser.add_argument('--adapter', default=None, help='adapter name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--projector-type', default=None, help='projector type')
    parser.add_argument('--llm-weight', default=None, help='llm weight')
    parser.add_argument(
        '--projector-weight', default=None, help='projector weight')
    parser.add_argument('--image', default=None, help='image')

    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=PROMPT_TEMPLATE.default,
        help='Specify a prompt template')

    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument('--command-stop-word', default=None, help='Stop key')
    parser.add_argument('--answer-stop-word', default=None, help='Stop key')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args


def get_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    while result is None:
        print(('\ndouble enter to end input (EXIT: exit chat, '
               'RESET: reset history) >>> '),
              end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')
    return result


# def prepare_inputs_for_generation(
#     self,
#     input_ids,
#     past_key_values=None,
#     attention_mask=None,
#     inputs_embeds=None,
#     **kwargs
# ):
#     if past_key_values:
#         input_ids = input_ids[:, -1:]

#     position_ids = kwargs.get("position_ids", None)
#     if attention_mask is not None and position_ids is None:
#         # create position_ids on the fly for batch generation
#         position_ids = attention_mask.long().cumsum(-1) - 1
#         position_ids.masked_fill_(attention_mask == 0, 1)
#         if past_key_values:
#             position_ids = position_ids[:, -1].unsqueeze(-1)

#     # if `inputs_embeds` are passed, we only want to use them in
#     # the 1st generation step
#     if inputs_embeds is not None and past_key_values is None:
#         model_inputs = {"inputs_embeds": inputs_embeds}
#     else:
#         model_inputs = {"input_ids": input_ids}

#     model_inputs.update(
#         {
#             "position_ids": position_ids,
#             "past_key_values": past_key_values,
#             "use_cache": kwargs.get("use_cache"),
#             "attention_mask": attention_mask,
#             "images": kwargs.get("images", None)
#         }
#     )
#     return model_inputs

# def forward(self, *args, **kwargs):
#     if kwargs.get('pixel_values', None) is not None:
#         visual_outputs = self.visual_encoder(
#             kwargs['pixel_values'], output_hidden_states=True)
#         pixel_values = self.projector(
#             visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
#         kwargs['pixel_values'] = pixel_values
#         kwargs = self.prepare_inputs_labels_for_multimodal(**kwargs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # model_kwargs
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True
    }
    # build model
    llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                               **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)
    if args.adapter is not None:
        llm = PeftModel.from_pretrained(
            llm, args.adapter, offload_folder=args.offload_folder)
        print(f'Load adapter from {args.adapter}')
    visual_encoder = CLIPVisionModel.from_pretrained(args.visual_encoder)
    processor = CLIPImageProcessor.from_pretrained(args.visual_encoder)
    # TODO
    projector = MLPProjector(visual_encoder.config.hidden_size,
                             llm.config.hidden_size)
    print(f'Load projector weight from {args.projector_weight}!')
    projector_state_dict = torch.load(args.projector_weight)
    projector.load_state_dict(projector_state_dict)
    if args.llm_weight is not None:
        print(f'Load projector weight from {args.llm_weight}!')
        llm_state_dict = torch.load(args.llm_weight)
        llm.load_state_dict(llm_state_dict)
    llm.cuda()
    visual_encoder.cuda()
    projector.cuda()
    llm.eval()
    visual_encoder.eval()
    projector.eval()

    # llm.prepare_inputs_for_generation = prepare_inputs_for_generation

    if args.image is not None:
        image = load_image(args.image)
        image = expand2square(
            image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda()

    Streamer, stop_criteria = get_chat_utils(llm)
    if args.no_streamer:
        Streamer = None

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    n_turn = 0
    inputs = ''
    while True:
        text = get_input()
        while text.strip() == 'RESET':
            print('Log: History responses have been removed!')
            n_turn = 0
            inputs = ''
            text = get_input()
        if text.strip() == 'EXIT':
            print('Log: Exit!')
            exit(0)

        if args.image is not None and n_turn == 0:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        template = PROMPT_TEMPLATE[args.prompt_template]
        prompt_text = ''
        if 'SYSTEM' in template and n_turn == 0:
            system_text = None
            if args.system_template is not None:
                system_text = SYSTEM_TEMPLATE[args.system_template].format(
                    round=n_turn + 1)
            elif args.system is not None:
                system_text = args.system
            if system_text is not None:
                prompt_text += template['SYSTEM'].format(
                    system=system_text, round=n_turn + 1)
        prompt_text += template['INSTRUCTION'].format(
            input=text, round=n_turn + 1)
        inputs += prompt_text
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = tokenizer(chunk)
            else:
                cur_encode = tokenizer(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode['input_ids'])
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda()

        visual_outputs = visual_encoder(
            image.unsqueeze(0), output_hidden_states=True)
        pixel_values = projector(visual_outputs.hidden_states[-2][:, 1:])

        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=llm, input_ids=ids.unsqueeze(0), pixel_values=pixel_values)

        streamer = Streamer(tokenizer) if Streamer is not None else None
        generate_output = llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=streamer,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria)
        if streamer is None:
            output_text = tokenizer.decode(generate_output[0][len(ids[0]):])
            end = '' if output_text[-1] == '\n' else '\n'
            print(output_text, end=end)
        inputs += tokenizer.decode(generate_output[0])
        n_turn += 1
        if len(generate_output[0]) >= args.max_new_tokens:
            print('Remove the memory of history responses, since '
                  f'it exceeds the length limitation {args.max_new_tokens}.')
            n_turn = 0
            inputs = ''


if __name__ == '__main__':
    main()
