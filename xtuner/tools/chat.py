# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import re

import torch
from mmengine.config import Config, DictAction
from transformers import GenerationConfig

from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
from xtuner.tools.utils import get_chat_utils, update_stop_criteria
from xtuner.utils import PROMPT_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser(
        description='Chat with a pretrained model')
    parser.add_argument(
        'config',
        help='config file name or path. Note: Please use the original '
        'configs, instead of the automatically saved log configs.')
    parser.add_argument('--adapter', default=None, help='adapter model')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt option')
    parser.add_argument(
        '--is-deepspeed',
        action='store_true',
        help='whether the adapter is saved from deepspeed')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument('--command-stop-word', default=None, help='Stop key')
    parser.add_argument('--answer-stop-word', default=None, help='Stop key')
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
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def get_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    while result is None:
        print('\ndouble enter to end input >>> ', end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')
    return result


def main():
    args = parse_args()

    if args.with_plugins is None:
        inner_thoughts_open = False
        calculate_open = False
        solve_open = False
        search_open = False
    else:
        assert args.prompt_template == 'moss_sft'
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

    torch.manual_seed(args.seed)

    # parse config
    if not os.path.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = BUILDER.build(cfg.model)
    # Cast to inference mode
    model.llm.gradient_checkpointing_disable()
    model.llm.config.use_cache = True

    tokenizer = BUILDER.build(cfg.tokenizer)

    if args.adapter is not None:
        adapter = torch.load(args.adapter, map_location='cpu')
        state_dict_key = 'module' if args.is_deepspeed else 'state_dict'
        model.load_state_dict(adapter[state_dict_key], strict=False)
        print(f'Load adapter from {args.adapter}')

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

    n_turn = 0
    inputs = ''
    while True:
        text = get_input()

        if text == 'exit':
            exit(0)
        if args.prompt_template is not None:
            template = PROMPT_TEMPLATE[args.prompt_template]
            if 'INSTRUCTION_START' in template and n_turn == 0:
                prompt_text = template['INSTRUCTION_START'].format(
                    input=text, round=n_turn + 1, **cfg)
            else:
                prompt_text = template['INSTRUCTION'].format(
                    input=text, round=n_turn + 1, **cfg)
            if args.prompt_template == 'moss_sft':
                if not inner_thoughts_open:
                    prompt_text.replace('- Inner thoughts: enabled.',
                                        '- Inner thoughts: disabled.')
                if not calculate_open:
                    prompt_text.replace(
                        '- Calculator: enabled. API: Calculate(expression)',
                        '- Calculator: disabled.')
                if not solve_open:
                    prompt_text.replace(
                        '- Equation solver: enabled. API: Solve(equation)',
                        '- Equation solver: disabled.')
                if not search_open:
                    prompt_text.replace(
                        '- Web search: enabled. API: Search(query)',
                        '- Web search: disabled.')

            inputs += prompt_text
        else:
            inputs += text
        ids = tokenizer.encode(inputs, return_tensors='pt')
        streamer = Streamer(tokenizer) if Streamer is not None else None
        if args.with_plugins is not None:
            generate_output = model.generate(
                inputs=ids.cuda(),
                generation_config=gen_config,
                streamer=streamer,
                stopping_criteria=command_stop_cr).cpu()
            generate_output_text = tokenizer.decode(
                generate_output[0][len(ids[0]):])
            if streamer is None:
                end = '' if generate_output_text[-1] == '\n' else '\n'
                print(generate_output_text, end=end)
            pattern = r'<\|Commands\|>:(.*?)<eoc>'
            command_text = ', '.join(re.findall(pattern, generate_output_text))
            extent_text = plugins_api(
                command_text,
                calculate_open=calculate_open,
                solve_open=solve_open,
                search_open=search_open)
            end = '' if extent_text[-1] == '\n' else '\n'
            print(extent_text, end=end)
            extent_text_ids = tokenizer.encode(
                extent_text, return_tensors='pt', add_special_tokens=False)
            new_ids = torch.cat((generate_output, extent_text_ids), dim=1)
            new_streamer = Streamer(
                tokenizer) if Streamer is not None else None
            generate_output = model.generate(
                inputs=new_ids.cuda(),
                generation_config=gen_config,
                streamer=new_streamer,
                stopping_criteria=answer_stop_cr)
            if streamer is None:
                output_text = tokenizer.decode(
                    generate_output[0][len(new_ids[0]):])
                end = '' if output_text[-1] == '\n' else '\n'
                print(output_text, end=end)
        else:
            generate_output = model.generate(
                inputs=ids.cuda(),
                generation_config=gen_config,
                streamer=streamer,
                stopping_criteria=answer_stop_cr)
            if streamer is None:
                output_text = tokenizer.decode(
                    generate_output[0][len(ids[0]):])
                end = '' if output_text[-1] == '\n' else '\n'
                print(output_text, end=end)
        inputs = tokenizer.decode(generate_output[0])
        n_turn += 1
        if len(generate_output[0]) >= args.max_new_tokens:
            print('Remove the memory of history responses, since '
                  f'it exceeds the length limitation {args.max_new_tokens}.')
            n_turn = 0
            inputs = ''


if __name__ == '__main__':
    main()
