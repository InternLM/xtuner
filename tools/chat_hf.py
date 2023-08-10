# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import re

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)
from utils import get_chat_utils, update_stop_criteria

from mmchat.utils import PROMPT_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMChat chat with a pretrained HF model')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')
    parser.add_argument('--adapter', default=None, help='adapter name or path')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins', action='store_true', help='Whether to with plugins')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument('--command-stop-word', default=None, help='Stop key')
    parser.add_argument('--answer-stop-word', default=None, help='Stop key')
    parser.add_argument(
        '--prompt',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt option')
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

    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main():
    args = parse_args()

    if args.with_plugins:
        from plugins import plugins_api

    torch.manual_seed(args.seed)

    # build model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    Streamer, stop_criteria = get_chat_utils(model)
    if args.no_streamer:
        Streamer = None

    command_stop_cr, answer_stop_cr = update_stop_criteria(
        base=stop_criteria,
        tokenizer=tokenizer,
        command_stop_word=args.command_stop_word,
        answer_stop_word=args.answer_stop_word)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    encode_kwargs = {}
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        encode_kwargs['disallowed_special'] = ()

    n_turn = 0
    inputs = ''
    while True:
        text = get_input()

        if text == 'exit':
            exit(0)
        if args.prompt is not None:
            template = PROMPT_TEMPLATE[args.prompt]
            if 'INSTRUCTION_START' in template and n_turn == 0:
                prompt_text = template['INSTRUCTION_START'].format(
                    input=text, bot_name=args.bot_name)
            else:
                prompt_text = template['INSTRUCTION'].format(
                    input=text, bot_name=args.bot_name)
            inputs += prompt_text
        else:
            inputs += text
        ids = tokenizer.encode(
            inputs,
            return_tensors='pt',
            add_special_tokens=n_turn == 0,
            **encode_kwargs)
        streamer = Streamer(tokenizer) if Streamer is not None else None
        if args.with_plugins:
            generate_output = model.generate(
                inputs=ids.cuda(),
                generation_config=gen_config,
                streamer=streamer,
                stopping_criteria=command_stop_cr).cpu()
            generate_output_text = tokenizer.decode(
                generate_output[0][len(ids[0]):])
            if streamer is None:
                print(generate_output_text, end='')
            try:
                calculate_open = re.findall(r'- Calculator: (.+)\.',
                                            inputs)[0] == 'enabled'
                solve_open = re.findall(r'- Equation solver: (.+)\.',
                                        inputs)[0] == 'enabled'
                search_open = re.findall(r'- Web search: (.+)\.',
                                         inputs)[0] == 'enabled'
            except Exception:
                print(f'Wrong prompt:\n{inputs}')
            pattern = r'<\|Commands\|>:(.*?)<eoc>'
            command_text = ', '.join(re.findall(pattern, generate_output_text))
            extent_text = plugins_api(
                command_text,
                calculate_open=calculate_open,
                solve_open=solve_open,
                search_open=search_open)
            print(extent_text, end='')
            extent_text_ids = tokenizer.encode(
                extent_text,
                return_tensors='pt',
                add_special_tokens=False,
                **encode_kwargs)
            new_ids = torch.cat((generate_output, extent_text_ids), dim=1)
            new_streamer = Streamer(
                tokenizer) if Streamer is not None else None
            generate_output = model.generate(
                inputs=new_ids.cuda(),
                generation_config=gen_config,
                streamer=new_streamer,
                stopping_criteria=answer_stop_cr)
            if streamer is None:
                print(
                    tokenizer.decode(generate_output[0][len(new_ids[0]):]),
                    end='')
        else:
            generate_output = model.generate(
                inputs=ids.cuda(),
                generation_config=gen_config,
                streamer=streamer,
                stopping_criteria=answer_stop_cr)
            if streamer is None:
                print(
                    tokenizer.decode(generate_output[0][len(ids[0]):]), end='')
        inputs = tokenizer.decode(generate_output[0]) + '\n'
        n_turn += 1
        if len(generate_output[0]) >= args.max_new_tokens:
            print('Remove the memory for history responses, since '
                  f'it exceeds the length limitation {args.max_new_tokens}.')
            n_turn = 0
            inputs = ''


if __name__ == '__main__':
    main()
