# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch

from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE


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
    llava_spec_group.add_argument('--image', default=None, help='image')

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
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    bot_spec_group.add_argument(
        '--chat-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    bot_spec_group.add_argument(
        '--system', default=None, help='Specify the system text')
    bot_spec_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')

    gen_group = parser.add_argument_group()
    gen_group.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
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

    parser.add_argument('--predict', type=str)
    parser.add_argument('--results', type=str, default='results.xlsx')

    parser.add_argument('--predict-repeat', type=int)
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

    use_lagent = args.lagent
    use_llava = args.llava is not None
    use_moss = args.moss_plugins is not None
    use_lmdeploy = args.lmdeploy
    use_vllm = args.vllm

    if use_lagent + use_moss + use_llava > 1:
        raise RuntimeError

    if use_vllm + use_lmdeploy > 1:
        raise RuntimeError

    chat_template = PROMPT_TEMPLATE[args.prompt_template]
    # system_template = SYSTEM_TEMPLATE[args.system_template]
    system_template = None

    if use_lmdeploy:
        if use_lagent:
            raise NotImplementedError
        elif use_moss:
            from xtuner.bot import LMDeployMossBot
            return LMDeployMossBot(args.bot_name, args.model_name_or_path,
                                   chat_template, system_template,
                                   args.max_length, args.max_new_tokens,
                                   args.temperature, args.top_k, args.top_p,
                                   args.repetition_penalty, args.stop_words,
                                   args.seed, args.logn_attn,
                                   args.rope_scaling_factor, args.moss_plugins)
        elif use_llava:
            raise NotImplementedError
        else:
            from xtuner.bot import LMDeployChatBot
            return LMDeployChatBot(args.bot_name, args.model_name_or_path,
                                   chat_template, system_template,
                                   args.max_length, args.max_new_tokens,
                                   args.temperature, args.top_k, args.top_p,
                                   args.repetition_penalty, args.stop_words,
                                   args.seed, args.logn_attn,
                                   args.rope_scaling_factor)
    elif use_vllm:
        if use_lagent:
            raise NotImplementedError
        elif use_moss:
            # TODO
            pass
        elif use_llava:
            raise NotImplementedError
        else:
            from xtuner.bot import vLLMChatBot
            return vLLMChatBot(args.bot_name, args.model_name_or_path,
                               chat_template, system_template, args.max_length,
                               args.max_new_tokens, args.temperature,
                               args.top_k, args.top_p, args.repetition_penalty,
                               args.stop_words, args.seed, args.logn_attn,
                               args.rope_scaling_factor)
    else:
        if use_lagent:
            from xtuner.bot import HFReActBot
            return HFReActBot(args.model_name_or_path, args.adapter, args.bits)
        elif use_moss:
            from xtuner.bot import HFMossBot
            return HFMossBot(args.bot_name, args.model_name_or_path,
                             args.adapter, args.bits, chat_template,
                             system_template, args.max_length,
                             args.max_new_tokens, args.temperature, args.top_k,
                             args.top_p, args.repetition_penalty,
                             args.stop_words, args.moss_plugins)
        elif use_llava:
            raise NotImplementedError
        else:
            from xtuner.bot import HFChatBot
            return HFChatBot(args.bot_name, args.model_name_or_path,
                             args.adapter, args.bits, chat_template,
                             system_template, args.max_length,
                             args.max_new_tokens, args.temperature, args.top_k,
                             args.top_p, args.repetition_penalty,
                             args.stop_words)


def interactive_chat(bot, system):

    while True:
        text = get_input()
        while text.strip() == 'RESET':
            print('Log: History responses have been removed!')
            bot.reset_history()
            text = get_input()

        if text.strip() == 'EXIT':
            print('Log: Exit!')
            exit(0)

        response = bot.chat(text, system)
        print(response, system)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    bot = build_bot(args)

    if args.predict:
        from datasets import load_dataset
        dataset = load_dataset('text', data_files=args.predict)['train']
        texts = dataset['text']
        chat_instance = bot.create_instance()

        for i in range(args.predict_repeat):
            preds = chat_instance.predict(texts, args.system)
            dataset = dataset.add_column(f'response_{i}', preds)

        df = dataset.to_pandas()
        sheet_name = 'lmdeploy' if args.lmdeploy else 'huggingface'
        df.to_excel(args.results, sheet_name)
        print(f'Results saved in {args.results}')
    else:
        chat_instance = bot.create_instance()
        interactive_chat(chat_instance, args.system)


if __name__ == '__main__':
    main()
