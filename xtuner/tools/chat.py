# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re
import sys

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from transformers.generation.streamers import TextStreamer

from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


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
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument(
        '--adapter', default=None, help='adapter name or path')
    adapter_group.add_argument(
        '--llava', default=None, help='llava name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument('--image', default=None, help='image')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
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
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
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
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # build llm
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
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }
    if args.lagent:
        from lagent.actions import ActionExecutor, GoogleSearch
        from lagent.agents import (CALL_PROTOCOL_CN, FORCE_STOP_PROMPT_CN,
                                   ReAct, ReActProtocol)
        from lagent.llms import HFTransformerCasualLM

        try:
            SERPER_API_KEY = os.environ['SERPER_API_KEY']
        except Exception:
            print('Please obtain the `SERPER_API_KEY` from https://serper.dev '
                  'and set it using `export SERPER_API_KEY=xxx`.')
            sys.exit(1)

        model_kwargs.pop('trust_remote_code')
        llm = HFTransformerCasualLM(
            args.model_name_or_path, model_kwargs=model_kwargs)
        if args.adapter is not None:
            print(f'Loading adapter from {args.adapter}...')
            llm.model = PeftModel.from_pretrained(
                llm.model,
                args.adapter,
                offload_folder=args.offload_folder,
                trust_remote_code=True)
        search_tool = GoogleSearch(api_key=SERPER_API_KEY)
        chatbot = ReAct(
            llm=llm,
            action_executor=ActionExecutor(actions=[search_tool]),
            protocol=ReActProtocol(
                call_protocol=CALL_PROTOCOL_CN,
                force_stop=FORCE_STOP_PROMPT_CN))
        while True:
            text = get_input()
            while text.strip() == 'RESET':
                print('Log: History responses have been removed!')
                chatbot._session_history = []
                inputs = ''
                text = get_input()
            if text.strip() == 'EXIT':
                print('Log: Exit!')
                exit(0)
            response = chatbot.chat(text)
            print(response.response)
    else:
        if args.with_plugins is None:
            inner_thoughts_open = False
            calculate_open = False
            solve_open = False
            search_open = False
        else:
            assert args.prompt_template == args.system_template == 'moss_sft'
            from plugins import plugins_api
            inner_thoughts_open = True
            calculate_open = 'calculate' in args.with_plugins
            solve_open = 'solve' in args.with_plugins
            search_open = 'search' in args.with_plugins
            # pre-import for api and model preparation
            if calculate_open:
                from plugins import calculate  # noqa: F401
            if solve_open:
                from plugins import solve  # noqa: F401
            if search_open:
                from plugins import search  # noqa: F401
        # build llm
        llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                   **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            encode_special_tokens=True)
        print(f'Load LLM from {args.model_name_or_path}')
        if args.adapter is not None:
            llm = PeftModel.from_pretrained(
                llm,
                args.adapter,
                offload_folder=args.offload_folder,
                trust_remote_code=True)
            print(f'Load adapter from {args.adapter}')
        if args.llava is not None:
            llava_path = snapshot_download(
                repo_id=args.llava) if not osp.isdir(
                    args.llava) else args.llava

            # build visual_encoder
            if 'visual_encoder' in os.listdir(llava_path):
                assert args.visual_encoder is None, (
                    "Please don't specify the `--visual-encoder` since passed "
                    '`--llava` contains a visual encoder!')
                visual_encoder_path = osp.join(llava_path, 'visual_encoder')
            else:
                assert args.visual_encoder is not None, (
                    'Please specify the `--visual-encoder`!')
                visual_encoder_path = args.visual_encoder
            visual_encoder = CLIPVisionModel.from_pretrained(
                visual_encoder_path,
                torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
            image_processor = CLIPImageProcessor.from_pretrained(
                visual_encoder_path)
            print(f'Load visual_encoder from {visual_encoder_path}')

            # load adapter
            if 'llm_adapter' in os.listdir(llava_path):
                adapter_path = osp.join(llava_path, 'llm_adapter')
                llm = PeftModel.from_pretrained(
                    llm,
                    adapter_path,
                    offload_folder=args.offload_folder,
                    trust_remote_code=True)
                print(f'Load LLM adapter from {args.llava}')
            if 'visual_encoder_adapter' in os.listdir(llava_path):
                adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
                visual_encoder = PeftModel.from_pretrained(
                    visual_encoder,
                    adapter_path,
                    offload_folder=args.offload_folder)
                print(f'Load visual_encoder adapter from {args.llava}')

            # build projector
            projector_path = osp.join(llava_path, 'projector')
            projector = AutoModel.from_pretrained(
                projector_path,
                torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
                trust_remote_code=True)
            print(f'Load projector from {args.llava}')

            projector.cuda()
            projector.eval()
            visual_encoder.cuda()
            visual_encoder.eval()

        llm.eval()

        if args.image is not None:
            image = load_image(args.image)
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            image = image.cuda().unsqueeze(0).to(visual_encoder.dtype)
            visual_outputs = visual_encoder(image, output_hidden_states=True)
            pixel_values = projector(
                visual_outputs.hidden_states[args.visual_select_layer][:, 1:])

        stop_words = args.stop_words
        sep = ''
        if args.prompt_template:
            template = PROMPT_TEMPLATE[args.prompt_template]
            stop_words += template.get('STOP_WORDS', [])
            sep = template.get('SEP', '')
        stop_criteria = get_stop_criteria(
            tokenizer=tokenizer, stop_words=stop_words)

        if args.no_streamer:
            streamer = None
        else:
            streamer = TextStreamer(tokenizer, skip_prompt=True)

        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
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

            if args.prompt_template:
                prompt_text = ''
                template = PROMPT_TEMPLATE[args.prompt_template]
                if 'SYSTEM' in template and n_turn == 0:
                    system_text = None
                    if args.system_template is not None:
                        system_text = SYSTEM_TEMPLATE[
                            args.system_template].format(
                                round=n_turn + 1, bot_name=args.bot_name)
                    elif args.system is not None:
                        system_text = args.system
                    if system_text is not None:
                        prompt_text += template['SYSTEM'].format(
                            system=system_text,
                            round=n_turn + 1,
                            bot_name=args.bot_name)
                prompt_text += template['INSTRUCTION'].format(
                    input=text, round=n_turn + 1, bot_name=args.bot_name)
                if args.prompt_template == args.system_template == 'moss_sft':
                    if not inner_thoughts_open:
                        prompt_text.replace('- Inner thoughts: enabled.',
                                            '- Inner thoughts: disabled.')
                    if not calculate_open:
                        prompt_text.replace(('- Calculator: enabled. API: '
                                             'Calculate(expression)'),
                                            '- Calculator: disabled.')
                    if not solve_open:
                        prompt_text.replace(
                            '- Equation solver: enabled. API: Solve(equation)',
                            '- Equation solver: disabled.')
                    if not search_open:
                        prompt_text.replace(
                            '- Web search: enabled. API: Search(query)',
                            '- Web search: disabled.')
            else:
                prompt_text = text
            inputs += prompt_text
            if args.image is None:
                if n_turn == 0:
                    ids = tokenizer.encode(inputs, return_tensors='pt')
                else:
                    ids = tokenizer.encode(
                        inputs, return_tensors='pt', add_special_tokens=False)

                if args.with_plugins is not None:
                    generate_output = llm.generate(
                        inputs=ids.cuda(),
                        generation_config=gen_config,
                        streamer=streamer,
                        stopping_criteria=stop_criteria).cpu()
                    generate_output_text = tokenizer.decode(
                        generate_output[0][len(ids[0]):])
                    if streamer is None:
                        end = '' if generate_output_text[-1] == '\n' else '\n'
                        print(generate_output_text, end=end)
                    pattern = r'<\|Commands\|>:(.*?)<eoc>'
                    command_text = ', '.join(
                        re.findall(pattern, generate_output_text))
                    extent_text = plugins_api(
                        command_text,
                        calculate_open=calculate_open,
                        solve_open=solve_open,
                        search_open=search_open)
                    end = '' if extent_text[-1] == '\n' else '\n'
                    print(extent_text, end=end)
                    extent_text_ids = tokenizer.encode(
                        extent_text,
                        return_tensors='pt',
                        add_special_tokens=False)
                    new_ids = torch.cat((generate_output, extent_text_ids),
                                        dim=1)

                    generate_output = llm.generate(
                        inputs=new_ids.cuda(),
                        generation_config=gen_config,
                        streamer=streamer,
                        stopping_criteria=stop_criteria)
                    if streamer is None:
                        output_text = tokenizer.decode(
                            generate_output[0][len(new_ids[0]):])
                        end = '' if output_text[-1] == '\n' else '\n'
                        print(output_text, end=end)
                else:
                    generate_output = llm.generate(
                        inputs=ids.cuda(),
                        generation_config=gen_config,
                        streamer=streamer,
                        stopping_criteria=stop_criteria)
                    if streamer is None:
                        output_text = tokenizer.decode(
                            generate_output[0][len(ids[0]):])
                        end = '' if output_text[-1] == '\n' else '\n'
                        print(output_text, end=end)
                inputs = tokenizer.decode(generate_output[0])
            else:
                chunk_encode = []
                for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                    if idx == 0 and n_turn == 0:
                        cur_encode = tokenizer.encode(chunk)
                    else:
                        cur_encode = tokenizer.encode(
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
                    llm=llm, input_ids=ids, pixel_values=pixel_values)

                generate_output = llm.generate(
                    **mm_inputs,
                    generation_config=gen_config,
                    streamer=streamer,
                    bos_token_id=tokenizer.bos_token_id,
                    stopping_criteria=stop_criteria)
                if streamer is None:
                    output_text = tokenizer.decode(generate_output[0])
                    end = '' if output_text[-1] == '\n' else '\n'
                    print(output_text, end=end)
                inputs += tokenizer.decode(generate_output[0])
            n_turn += 1
            inputs += sep
            if len(generate_output[0]) >= args.max_new_tokens:
                print(
                    'Remove the memory of history responses, since '
                    f'it exceeds the length limitation {args.max_new_tokens}.')
                n_turn = 0
                inputs = ''


if __name__ == '__main__':
    main()
