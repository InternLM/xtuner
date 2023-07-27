import argparse

import torch
from mmengine.config import Config, DictAction
from transformers import GenerationConfig
from utils import PROMPT_TEMPLATE, get_chat_utils

from mmchat.registry import MODELS, TOKENIZER


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMChat chat with a pretrained model')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--prompt',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt option')
    parser.add_argument(
        '--adapter-checkpoint', default=None, help='adapter checkpoint file')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=1,
        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.')
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

    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = MODELS.build(cfg.model)

    Decorator, Streamer, stop_criteria = get_chat_utils(model)

    if args.adapter_checkpoint is not None:
        adapter_checkpoint = torch.load(
            args.adapter_checkpoint, map_location='cpu')
        model.load_state_dict(adapter_checkpoint['state_dict'], strict=False)
        print(f'Load adapter from {args.adapter_checkpoint}')

    tokenizer = TOKENIZER.build(cfg.tokenizer)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # warmup
    warmup_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    model.llm.generate(
        inputs=torch.tensor([[1]]).cuda(), generation_config=warmup_config)

    while True:
        streamer = Streamer(tokenizer)
        text = get_input()

        if text == 'exit':
            exit(0)
        if args.prompt is not None:
            text = PROMPT_TEMPLATE[args.prompt].format(input=text)
        text = Decorator.decorate(text)
        ids = tokenizer.encode(text, return_tensors='pt')

        model.llm.generate(
            inputs=ids.cuda(),
            generation_config=gen_config,
            streamer=streamer,
            stopping_criteria=stop_criteria)


if __name__ == '__main__':
    main()
