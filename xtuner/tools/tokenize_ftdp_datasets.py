import argparse
import json
import os
import os.path as osp
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
from mmengine import list_dir_or_file, track_progress_rich
from transformers import AutoTokenizer

SEPCIAL_TOKENS = [
    '<|plugin|>', '<|interpreter|>', '<|action_end|>', '<|action_start|>',
    '<|im_end|>', '<|im_start|>'
]

CHATML_LLAMAV13_32K_TOKEN_CFG = dict(
    role_cfg=dict(
        system=dict(
            begin=dict(
                with_name='<|im_start|>system name={name}\n',
                without_name='<|im_start|>system\n',
                name={
                    'interpreter': '<|interpreter|>',
                    'plugin': '<|plugin|>',
                }),
            end='<|im_end|>\n',
            loss=dict(
                meta=False,
                icl=False,
                current=False,
                prefix=False,
            )),
        user=dict(
            begin=dict(
                with_name='<|im_start|>user name={name}\n',
                without_name='<|im_start|>user\n',
            ),
            end='<|im_end|>\n',
            loss=dict(
                icl=False,
                current=False,
                prefix=False,
            )),
        assistant=dict(
            begin=dict(
                with_name='<|im_start|>assistant name={name}\n',
                without_name='<|im_start|>assistant\n',
                name={
                    'interpreter': '<|interpreter|>',
                    'plugin': '<|plugin|>',
                }),
            end='<|im_end|>\n',
            loss=dict(
                icl=True,
                current=True,
                prefix=False,
                end=True,
            )),
        environment=dict(
            begin=dict(
                with_name='<|im_start|>environment name={name}\n',
                without_name='<|im_start|>environment\n',
                name={
                    'interpreter': '<|interpreter|>',
                    'plugin': '<|plugin|>',
                }),
            end='<|im_end|>\n',
            loss=dict(
                icl=False,
                current=False,
                prefix=False,
            )),
        tool=dict(
            begin=dict(
                with_name='<|action_start|>{name}\n',
                name={
                    'interpreter': '<|interpreter|>',
                    'plugin': '<|plugin|>',
                }),
            end='<|action_end|>\n',
            belong='assistant',
        ),
        thought=dict(
            begin=dict(without_name=''),
            end='',
            belong='assistant',
        ),
    ),
    max_len=32 * 1024,
)


def chatml_format(
    processed_data,
    tokenizer,
    role_cfg,
    max_len=2048,
    encode_json=True,
):
    """
    ```python
        dict(
            role='',
            content='',
            name='', -> Begin 扩增
            type='',
            )
    ```
    ```python
        dict(
            system=dict(
                begin=dict(
                    with_name='<TOKENS_UNUSED_140>system name={name}\n',
                    without_name='<TOKENS_UNUSED_140>system\n',
                    name={
                        'interpreter': '<TOKENS_UNUSED_136>',
                        'plugin': '<TOKENS_UNUSED_135>',
                    }),
                end='<TOKENS_UNUSED_139>\n',
                loss=dict(
                    meta=False,
                    icl=False,
                    current=False,
                    prefix=False,
                )),
            user=dict(
                begin=dict(
                    with_name='<TOKENS_UNUSED_140>user name={name}\n',
                    without_name='<TOKENS_UNUSED_140>user\n',
                ),
                end='<TOKENS_UNUSED_139>\n',
                loss=dict(
                    icl=False,
                    current=False,
                    prefix=False,
                )),
            assistant=dict(
                begin=dict(
                    with_name='<TOKENS_UNUSED_140>assistant name={name}\n',
                    without_name='<TOKENS_UNUSED_140>assistant\n',
                    name={
                        'interpreter': '<TOKENS_UNUSED_136>',
                        'plugin': '<TOKENS_UNUSED_135>',
                    }),
                end='<TOKENS_UNUSED_139>\n',
                loss=dict(
                    icl=True,
                    current=True,
                    prefix=False,
                    end=True,
                )),
            environment=dict(
                begin=dict(
                    with_name='<TOKENS_UNUSED_140>environment name={name}\n',
                    without_name='<TOKENS_UNUSED_140>environment\n',
                    name={
                        'interpreter': '<TOKENS_UNUSED_136>',
                        'plugin': '<TOKENS_UNUSED_135>',
                    }),
                end='<TOKENS_UNUSED_139>\n',
                loss=dict(
                    icl=False,
                    current=False,
                    prefix=False,
                )),
            tool=dict(
                begin=dict(
                    with_name='<TOKENS_UNUSED_138>{name}\n',
                    name={
                        'interpreter': '<TOKENS_UNUSED_136>',
                        'plugin': '<TOKENS_UNUSED_135>',
                    }),
                end='<TOKENS_UNUSED_137>\n',
                belong='assistant',
            ),
            thought=dict(
                begin='',
                end='',
                belong='assistant',
        ),
    ```
    """

    def format_begin(role_cfg, message):
        name = message.get('name', None)
        if name is not None:
            begin = role_cfg['begin'].get('with_name', '')
            if name in role_cfg['begin'].get('name', {}):
                begin = begin.format(name=role_cfg['begin']['name'][name])
            else:
                begin = begin.format(name=name)
        else:
            begin = role_cfg['begin'].get('without_name', '')
        return begin

    def format_sub_role(messages: List[Dict], roles_cfg) -> List[Dict]:
        new_message = list()
        for message in messages:
            if message['role'] in [
                    'assistant', 'user', 'system', 'environment'
            ]:
                new_message.append(message)
                continue
            role_cfg = roles_cfg[message['role']]
            begin = format_begin(role_cfg, message)
            new_content = begin + message['content'] + role_cfg['end']
            if role_cfg.get('fallback_role'):
                new_message.append(
                    dict(role=role_cfg['fallback_role'], content=new_content))
            elif role_cfg.get('belong'):
                if new_message[-1]['role'] != role_cfg.get('belong'):
                    new_message.append(
                        dict(role=role_cfg.get('belong'), content=new_content))
                else:
                    new_message[-1]['content'] += new_content
            else:
                new_message.append(
                    dict(role=message['role'], content=new_content))

        return new_message

    token_ids = []
    _processed_data = format_sub_role(processed_data, role_cfg)

    for dialog_item in _processed_data:
        role = dialog_item['role']
        content = dialog_item['content']
        # TODO: is strip necessary? or use lstrip? 避免开始有\n\n的情况
        # content = content.lstrip()
        begin = format_begin(role_cfg[role], dialog_item)
        end = role_cfg[role]['end']
        begin_token = tokenizer.encode(begin, add_special_tokens=False)
        if not role_cfg[role]['loss'].get('beigin', False):
            begin_token = [-token_id for token_id in begin_token]
        end_token = tokenizer.encode(
            role_cfg[role]['end'], add_special_tokens=False)
        # breakpoint()
        if not role_cfg[role]['loss'].get('end', False):
            end_token = [-token_id for token_id in end_token]

        content_token = tokenizer.encode(
            begin + content + end, add_special_tokens=False)
        content_token = content_token[len(begin_token):-len(end_token)]

        if dialog_item.get('loss', True):
            loss_cfg = role_cfg[role]['loss']
        else:
            loss_cfg = dict(icl=False, current=False, meta=False)
        if not loss_cfg[dialog_item.get('type', 'current')]:
            content_token = [-token_id for token_id in content_token]

        if begin == '':
            tokens = content_token
        else:
            tokens = begin_token + content_token
        if end != '':
            tokens = tokens + end_token

        token_ids += tokens

    token_ids = [tokenizer.bos_token_id] + token_ids
    token_ids = token_ids[:max_len]
    if encode_json:
        line = str.encode(json.dumps({'tokens': token_ids}) + '\n')
        return line, len(token_ids)
    return token_ids, len(token_ids)


def write_bin_meta_bin(path, dataset_name, filename, samples):
    train_path = osp.join(path, f'train/cn/{dataset_name}')
    valid_path = osp.join(path, f'valid/cn/{dataset_name}')
    train_dir = Path(train_path)
    valid_dir = Path(valid_path)
    train_dir.mkdir(exist_ok=True, parents=True)
    valid_dir.mkdir(exist_ok=True, parents=True)
    train_f = open(train_dir.joinpath(f'{filename}.bin'), 'wb')
    valid_f_path = valid_dir.joinpath(f'{filename}.bin')
    valid_f = open(valid_f_path, 'wb')
    print(train_dir)
    print(valid_dir)
    train_tokens = 0
    valid_tokens = 0
    last_train_position = 0
    last_valid_position = 0
    train_samples = 0
    valid_samples = 0
    train_meta = []
    valid_meta = []
    for line, token_num in samples:
        train_tokens += token_num
        train_f.write(line)
        train_meta.append((last_train_position, token_num))
        last_train_position += len(line)
        train_samples += 1
        if (train_samples) % 100 == 0:  # ?
            valid_tokens += token_num
            valid_f.write(line)
            valid_meta.append((last_valid_position, token_num))
            last_valid_position += len(line)
            valid_samples += 1
    train_f.close()
    valid_f.close()
    np.save(open(train_dir.joinpath(f'{filename}.bin.meta'), 'wb'), train_meta)

    # remove the length of `valid_samples` is less than 500
    # 500 is a magic number, you can change it to any number you want
    # the number must bigger the DP.
    if valid_samples > 500:
        np.save(
            open(valid_dir.joinpath(f'{filename}.bin.meta'), 'wb'), valid_meta)
    else:
        print(f'{valid_f_path} is removed because the number of',
              f'`valid_samples`({valid_samples}) is less than 500')
        os.remove(valid_f_path)
    return train_tokens, valid_tokens, train_samples, valid_samples


def tokenize_and_save(tokenizer, processed_dir, tokenized_dir):
    tokenized_save_dir = osp.join(tokenized_dir, 'chatml_llamav13_32k')
    data_dir = processed_dir
    all_train_tokens = 0
    all_valid_tokens = 0
    all_train_samples = 0
    all_valid_samples = 0

    for filename in list_dir_or_file(data_dir, recursive=True, list_dir=False):
        file_path = os.path.join(data_dir, filename)
        if '/processed/' not in file_path:
            continue
        assert '.jsonl' in filename

        # dataset name such as char_x10_chat_format
        dataset_name = filename.split(os.sep)[0]

        # Hardcode here to skip tokenizing the file if it already exists
        # (Refactor the `write_bin_meta_bin`!).
        train_f = osp.join(tokenized_save_dir, 'train', 'cn', dataset_name,
                           f'{osp.splitext(osp.basename(filename))[0]}.bin')
        if osp.isfile(train_f):
            print(f'{train_f} already exists, skip it')
            continue

        tokenize_fun = partial(
            chatml_format,
            tokenizer=tokenizer,
            **CHATML_LLAMAV13_32K_TOKEN_CFG)
        samples = []
        with open(file_path) as f:
            dataset = f.readlines()
        task_num = len(dataset)
        dataset = map(lambda x: (json.loads(x), ), dataset)

        for sample in track_progress_rich(
                tokenize_fun,
                dataset,
                nproc=32,
                task_num=task_num,
                chunksize=32,
                description=f'{os.path.basename(file_path)}...'):
            samples.append(sample)

        train_tokens, valid_tokens, train_samples, valid_samples = write_bin_meta_bin(  # noqa E501
            path=tokenized_save_dir,
            dataset_name=dataset_name,
            samples=samples,
            filename=osp.splitext(osp.basename(filename))[0])
        if train_tokens is None:
            print(f'{osp.splitext(osp.basename(filename))[0]} already '
                  'exists, skip it')
            continue

        print(f'train_tokens {train_tokens}', flush=True)
        print(f'train_samples {train_samples}')
        print(f'valid tokens {valid_tokens}')
        print(f'valid_samples {valid_samples}')
        all_train_tokens += train_tokens
        all_valid_tokens += valid_tokens
        all_train_samples += train_samples
        all_valid_samples += valid_samples

    print(f'all train tokens {all_train_tokens}')
    print(f'all train samples {all_train_samples}')
    print(f'all valid tokens {all_valid_tokens}')
    print(f'all valid samples {all_valid_samples}')


def tokenizer_add_special_tokens(tokenizer):
    print(f'Before adding special tokens, Vocabulary Size: {len(tokenizer)}')
    for special_token in SEPCIAL_TOKENS:
        if special_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([special_token], special_tokens=True)
    print(f'After adding special tokens, Vocabulary Size: {len(tokenizer)}')


def save_new_tokenizer(tokenizer, save_dir):
    tokenizer.save_pretrained(save_dir)
    print(f'save new tokenizer to {save_dir}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--processed-dir', help='The folder to save untokenized data.')
    parser.add_argument(
        '--tokenized-dir', help='The folder to save tokenized data.')
    parser.add_argument(
        '--tokenizer-path', help='The path to the hf tokenizer.')
    parser.add_argument(
        '--tokenizer-w-special-tokens-save-dir',
        default=None,
        help='We have to add special tokens to the vocabulary of '
        'the given tokenizer, and save the new tokenizer to this folder.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, padding_side='right')

    ori_vocab_size = len(tokenizer)
    tokenizer_add_special_tokens(tokenizer)
    if len(tokenizer) != ori_vocab_size:
        save_new_tokenizer(tokenizer, args.tokenizer_w_special_tokens_save_dir)

    tokenize_and_save(tokenizer, args.processed_dir, args.tokenized_dir)


if __name__ == '__main__':
    main()
