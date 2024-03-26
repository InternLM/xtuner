# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch

from xtuner.chat import GenerationConfig
from xtuner.chat.template import CHAT_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')

    parser.add_argument('--adapter', default=None, help='adapter name or path')

    hf_spec_group = parser.add_argument_group()
    hf_spec_group.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='HF LLM bits')

    lmdeploy_spec_group = parser.add_argument_group()
    lmdeploy_spec_group.add_argument(
        '--lmdeploy', action='store_true', help='Whether to use lmdeploy')
    lmdeploy_spec_group.add_argument(
        '--logn-attn', action='store_true', help='Whether to use dynamic NTK')
    lmdeploy_spec_group.add_argument(
        '--dynamic-ntk',
        action='store_true',
        help='Whether to use dynamic NTK')
    lmdeploy_spec_group.add_argument(
        '--rope_scaling_factor', type=float, default=0.0, help='')

    vllm_group = parser.add_argument_group()
    vllm_group.add_argument(
        '--vllm', action='store_true', help='Whether to use vLLM')

    llava_spec_group = parser.add_argument_group()
    llava_spec_group.add_argument(
        '--llava', default=None, help='llava name or path')
    llava_spec_group.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    llava_spec_group.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    llava_spec_group.add_argument('--image', type=str, help='image')

    lagent_spec_group = parser.add_argument_group()
    lagent_spec_group.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')

    moss_spec_group = parser.add_argument_group()
    moss_spec_group.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    moss_spec_group.add_argument(
        '--moss-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')

    bot_spec_group = parser.add_argument_group()
    bot_spec_group.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    bot_spec_group.add_argument(
        '--prompt-template',
        choices=CHAT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    bot_spec_group.add_argument(
        '--chat-template',
        choices=CHAT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    bot_spec_group.add_argument(
        '--system', default=None, help='Specify the system text')

    gen_group = parser.add_argument_group()
    gen_group.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    gen_group.add_argument(
        '--max-new-tokens',
        type=int,
        default=512,
        help='Maximum number of new tokens allowed in generated text')
    gen_group.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    gen_group.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    gen_group.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    gen_group.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    gen_group.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')

    parser.add_argument(
        '--serve', action='store_true', help='Whether to serve')
    parser.add_argument('--port', type=str)

    parser.add_argument('--openai-api-key', type=str)
    parser.add_argument('--predict', type=str)
    parser.add_argument('--results', type=str, default='results.xlsx')

    parser.add_argument('--predict-repeat', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)

    args = parser.parse_args()

    if args.prompt_template and args.chat_template:
        # TODO depercated warning
        pass

    if args.moss_plugins and args.with_plugins:
        # TODO depercated warning
        pass

    if args.lmdeploy and args.adapter:
        raise NotImplementedError

    if args.lmdeploy and args.bits:
        raise NotImplementedError

    if not args.lmdeploy and args.batch_size > 1:
        raise NotImplementedError

    if args.lmdeploy and args.predict and args.batch_size == 1:
        # TODO adjust batch size warning
        pass

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


def build_bot(args):

    use_lmdeploy = args.lmdeploy
    use_vllm = args.vllm
    use_openai_api = ':' in args.model_name_or_path

    if use_vllm + use_lmdeploy > 1:
        raise RuntimeError

    if use_lmdeploy:
        from xtuner.chat import LMDeployBot
        return LMDeployBot(args.model_name_or_path, args.batch_size,
                           args.max_length, args.logn_attn,
                           args.rope_scaling_factor)

    elif use_vllm:
        from xtuner.chat import VllmBot
        return VllmBot(args.model_name_or_path, args.max_length,
                       args.logn_attn, args.rope_scaling_factor)
    elif use_openai_api:
        from xtuner.chat import OpenaiBot
        return OpenaiBot(args.model_name_or_path, args.openai_api_key)

    else:
        if args.llava and args.image:
            from xtuner.chat import HFLlavaBot
            return HFLlavaBot(
                args.model_name_or_path,
                args.llava,
                args.visual_encoder,
                bits=args.bits)

        else:
            from xtuner.chat import HFBot
            return HFBot(args.model_name_or_path, args.adapter, args.bits)


def build_chat_instance(bot, args):
    use_lagent = args.lagent
    use_llava = args.llava is not None
    use_moss = args.moss_plugins is not None

    chat_template = CHAT_TEMPLATE[args.prompt_template]

    if use_lagent + use_moss + use_llava > 1:
        raise RuntimeError

    if use_lagent:
        # TODO
        pass
    elif use_moss:
        # TODO
        pass
    elif use_llava:
        from xtuner.chat import LlavaChat
        return LlavaChat(bot, args.image, args.bot_name, chat_template)
    else:
        from xtuner.chat import BaseChat
        return BaseChat(bot, args.bot_name, chat_template)


def interactive_chat(bot, system, streamer, gen_config):

    while True:
        text = get_input()
        while text.strip() == 'RESET':
            print('Log: History responses have been removed!')
            bot.reset_history()
            text = get_input()

        if text.strip() == 'EXIT':
            print('Log: Exit!')
            exit(0)

        _ = bot.chat(text, system, streamer, gen_config)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.predict:
        bot = build_bot(args)
        instance = build_chat_instance(bot, args)

        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_words=[],
            seed=args.seed,
        )

        from datasets import load_dataset
        dataset = load_dataset('text', data_files=args.predict)['train']
        texts = dataset['text']

        for i in range(args.predict_repeat):
            preds = instance.predict(texts, args.system, gen_config)
            dataset = dataset.add_column(f'response_{i}', preds)

        df = dataset.to_pandas()

        if args.lmdeploy:
            sheet_name = 'lmdeploy'
        elif args.vllm:
            sheet_name = 'vllm'
        else:
            sheet_name = 'huggingface'

        df.to_excel(args.results, sheet_name)
        print(f'Results saved in {args.results}')

    elif args.serve:

        if args.lmdeploy:
            from xtuner.chat import run_lmdeploy_server
            run_lmdeploy_server(args.model_name_or_path, args.batch_size,
                                args.max_length)
        elif args.vllm:
            from xtuner.chat import run_vllm_server
            run_vllm_server(args.model_name_or_path)
    else:
        bot = build_bot(args)

        instance = build_chat_instance(bot, args)

        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_words=[],
            seed=args.seed,
        )
        streamer = instance.create_streamer()
        interactive_chat(instance, args.system, streamer, gen_config)


if __name__ == '__main__':
    main()
