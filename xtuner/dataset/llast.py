# Copyright (c) LLaST. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import csv
import logging
import os
from typing import List, Optional

import torch
import whisper
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel

from xtuner.dataset.huggingface import process_hf_dataset
from xtuner.utils import IGNORE_INDEX, LLAST_AUDIO_TOKEN_INDEX

# Language Mapping
LANG_DICT = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'Chinese': 'zh-CN',
    'German': 'de',
    'Japanese': 'ja',
    'Catalan': 'ca',
    'Italian': 'it',
    'Russian': 'ru',
    'Portuguese': 'pt',
    'Persian': 'fa',
    'Estonian': 'et',
    'Mongolian': 'mn',
    'Dutch': 'nl',
    'Turkish': 'tr',
    'Arabic': 'ar',
    'Swedish': 'sv-SE',
    'Latvian': 'lv',
    'Slovenian': 'sl',
    'Tamil': 'ta',
    'Indonesian': 'id',
    'Welsh': 'cy'
}

SIM_LANG_DICT = {v: k for k, v in LANG_DICT.items()}

device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'


def convert_data(
    data_path,
    mode='s2t',
    src_lang='French',
    tgt_lang='English',
    check_audio_path=False,
    cv_dir='',
    audio_folder='',
    postfix='',
):
    """
    Args:
        mode (str):
            's2t': speech-to-text translation
            's2s': speech-to-speech translation
            'asr': speech-to-text recognition
        src_lang (str):
        tgt_lang (str):

    Return:
        output_list (List)
        ids_list (List)
    """

    assert src_lang in list(
        LANG_DICT.keys()), 'src_languge: {} is not supported currently.'
    assert tgt_lang in list(
        LANG_DICT.keys()), 'tgt_language: {} is not supported currently.'

    with open(data_path) as f:
        reader = csv.DictReader(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
        )
        raw_data = [dict(e) for e in reader]

    output_list = []
    ids_list = []
    for item in raw_data:
        tgt_lang_text = item['translation']
        src_lang_text = item['sentence']

        if mode == 's2t':
            conv = {
                'input':
                '<audio>\nPlease translate the {} sentence into {}.'.format(
                    src_lang, tgt_lang),
                'output':
                tgt_lang_text
            }
        elif mode == 'asr':
            conv = {
                'input':
                '<audio>\nPlease transcribe the {} sentence into {}.'.format(
                    src_lang, src_lang),
                'output':
                src_lang_text
            }
        item['id'] = item['path'].split('.')[0]
        item_processed = dict(
            conversation=[conv],
            tgt_lang_text=tgt_lang_text,
            src_lang_text=src_lang_text,
            ids=item['id'],
            src_lang=LANG_DICT[src_lang])

        if check_audio_path:
            wav_path = os.path.join(cv_dir, LANG_DICT[src_lang], audio_folder,
                                    item['id'] + postfix)
            if not os.path.exists(wav_path):
                print_log(f'Wav:{wav_path} not exists', logger='current')
                continue

        output_list.append(item_processed)
        ids_list.append(item['id'])

    return output_list, ids_list


def prepare_inputs_labels_for_llast(
        llm: PreTrainedModel,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_tokens: Optional[torch.FloatTensor] = None,
        audio_lens: Optional[torch.FloatTensor] = None,
        audio_mask: Optional[torch.FloatTensor] = None):
    if audio_tokens is None:
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': None,
            'labels': labels
        }
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]

    new_inputs_embeds = []
    new_labels = []
    cur_audio_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_audios = (cur_input_ids == LLAST_AUDIO_TOKEN_INDEX).sum()
        if num_audios == 0:
            cur_audio_tokens = audio_tokens[cur_audio_idx]
            cur_inputs_embeds_1 = llm.get_input_embeddings()(cur_input_ids)
            cur_inputs_embeds = torch.cat(
                [cur_inputs_embeds_1, cur_audio_tokens[0:0]], dim=0)
            new_inputs_embeds.append(cur_inputs_embeds)
            new_labels.append(labels[batch_idx])
            cur_audio_idx += 1
            continue

        audio_token_indices = [-1] + torch.where(
            cur_input_ids == LLAST_AUDIO_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]
            ]
        cur_input_ids_noaudio = []
        cur_labels = labels[batch_idx]
        cur_labels_noaudio = []
        for i in range(len(audio_token_indices) - 1):
            cur_input_ids_noaudio.append(
                cur_input_ids[audio_token_indices[i] +
                              1:audio_token_indices[i + 1]])
            cur_labels_noaudio.append(cur_labels[audio_token_indices[i] +
                                                 1:audio_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noaudio]
        cur_inputs_embeds = llm.get_input_embeddings()(
            torch.cat(cur_input_ids_noaudio))
        cur_inputs_embeds_no_audio = torch.split(
            cur_inputs_embeds, split_sizes, dim=0)

        cur_new_inputs_embeds = []
        cur_new_labels = []

        for i in range(num_audios + 1):
            cur_new_inputs_embeds.append(cur_inputs_embeds_no_audio[i])
            cur_new_labels.append(cur_labels_noaudio[i])
            if i == 0:
                prefix_len = cur_inputs_embeds_no_audio[i].shape[0]
            if i < num_audios:
                cur_audio_tokens = audio_tokens[cur_audio_idx]
                cur_audio_idx += 1
                cur_new_inputs_embeds.append(cur_audio_tokens)
                cur_new_labels.append(
                    torch.full((cur_audio_tokens.shape[0], ),
                               IGNORE_INDEX,
                               device=cur_labels.device,
                               dtype=cur_labels.dtype))

        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_inputs_embeds.append(cur_new_inputs_embeds)
        new_labels.append(cur_new_labels)

    # Combine them
    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    new_inputs_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len),
                                   IGNORE_INDEX,
                                   dtype=new_labels[0].dtype,
                                   device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len),
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len),
                               dtype=position_ids.dtype,
                               device=position_ids.device)

    # remain debug
    if audio_lens is None and audio_tokens.shape[0] == 1:
        audio_lens = [audio_tokens.shape[1]]
    max_audio_len = max(audio_lens)
    for i, (cur_new_embed, cur_new_labels, audio_len) in enumerate(
            zip(new_inputs_embeds, new_labels, audio_lens)):
        cur_len = cur_new_embed.shape[0]
        new_inputs_embeds_padded.append(
            torch.cat((cur_new_embed,
                       torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                   dtype=cur_new_embed.dtype,
                                   device=cur_new_embed.device)),
                      dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            attention_mask[i, (prefix_len + audio_len):(prefix_len +
                                                        max_audio_len)] = False
            position_ids[i, :cur_len] = torch.arange(
                0,
                cur_len,
                dtype=position_ids.dtype,
                device=position_ids.device)

    new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return {
        'input_ids': None,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'past_key_values': past_key_values,
        'inputs_embeds': new_inputs_embeds,
        'labels': new_labels
    }


class LLaSTDataset(Dataset):
    """Dataset for LLaST.

    Args:
        en2x_sample_step (int): sample step of en2x data (1 means, no sampling,
            10 means sampling 1/10 data)
    """

    def __init__(self,
                 tsv_dir,
                 tokenizer,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 input_ids_with_output=True,
                 audio_folder='clips',
                 debug=False,
                 split='train',
                 cv_dir=None,
                 with_asr=True,
                 en2x_sample_step=1,
                 check_audio_path=False,
                 x2en=['fr', 'es'],
                 en2x=['ja', 'de']):

        super().__init__()

        assert offline_processed_text_folder or (tsv_dir and tokenizer)
        if offline_processed_text_folder and tsv_dir:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`tsv_dir` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        self.cv_dir = cv_dir
        self.postfix = '.wav'
        self.audio_folder = audio_folder
        if offline_processed_text_folder is not None and os.path.exists(
                offline_processed_text_folder):
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            raw_data, ids = self.get_data_ids(
                tsv_dir=tsv_dir,
                split=split,
                with_asr=with_asr,
                check_audio_path=check_audio_path,
                cv_dir=cv_dir,
                audio_folder=audio_folder,
                en2x_sample_step=en2x_sample_step,
                x2en=x2en,
                en2x=en2x)

            if debug:
                debug_length = int(len(raw_data) * 0.01)
                raw_data, ids = raw_data[:debug_length], ids[:debug_length]

            raw_data = DatasetDict({split: HFDataset.from_list(raw_data)})

            self.text_data = process_hf_dataset(
                # TODO: need to merge latest xtuner
                # broadcast_from_master=False,
                dataset=raw_data,
                tokenizer=tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split=split,
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_audio_token=True,
                input_ids_with_output=input_ids_with_output)

    def get_data_ids(self, tsv_dir, split, with_asr, check_audio_path, cv_dir,
                     audio_folder, en2x_sample_step, x2en, en2x):

        raw_data = []
        ids = []

        for src in x2en:
            data_path = os.path.join(tsv_dir,
                                     f'covost_v2.{src}_en.{split}.tsv')
            x2en_raw_data, x2en_ids = convert_data(
                data_path,
                src_lang=SIM_LANG_DICT[src],
                tgt_lang=SIM_LANG_DICT['en'],
                mode='s2t',
                check_audio_path=check_audio_path,
                cv_dir=cv_dir,
                audio_folder=audio_folder,
                postfix=self.postfix,
            )
            print_log(f'{src} to en:{len(x2en_raw_data)}', logger='current')
            raw_data += x2en_raw_data
            ids += x2en_ids
            # augement with asr data
            if split == 'train' and with_asr:
                print_log('*' * 30, logger='current')
                print_log(
                    'split is train, augment data with asr data',
                    logger='current')
                raw_data_asr, ids_asr = convert_data(
                    data_path, src_lang=SIM_LANG_DICT[src], mode='asr')
                raw_data += raw_data_asr
                ids += ids_asr

        for tgt in en2x:
            data_path = os.path.join(tsv_dir,
                                     f'covost_v2.en_{tgt}.{split}.tsv')

            en2x_raw_data, en2x_ids = convert_data(
                data_path,
                src_lang=SIM_LANG_DICT['en'],
                tgt_lang=SIM_LANG_DICT[tgt],
                mode='s2t',
                check_audio_path=check_audio_path,
                cv_dir=cv_dir,
                audio_folder=audio_folder,
                postfix=self.postfix)

            print_log(f'en to {tgt}:{len(en2x_raw_data)}', logger='current')
            # use sampling
            print_log(f'sampling step:{en2x_sample_step}', logger='current')
            assert en2x_sample_step < len(en2x_raw_data)
            en2x_raw_data, en2x_ids = en2x_raw_data[::en2x_sample_step], \
                en2x_ids[::en2x_sample_step]
            print_log(
                f'after sampled, en to {tgt}:{len(en2x_raw_data)}',
                logger='current')
            raw_data += en2x_raw_data
            ids += en2x_ids

        # augement with asr data of English
        if split == 'train' and with_asr and len(en2x) > 0:
            print_log(
                'split is train, augment data with asr data', logger='current')
            raw_data_asr, ids_asr = convert_data(
                data_path, src_lang=SIM_LANG_DICT['en'], mode='asr')
            raw_data += raw_data_asr
            ids += ids_asr

        return raw_data, ids

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('audio', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def get_whisper_features(self, cv_dir, ids, postfix='.mp3.wav'):
        self.whisper_feats = {}
        print_log('extract whisper fetures', logger='current')
        for item_id in tqdm(ids):
            item_id = item_id.split('.')[0]
            wav_path = os.path.join(cv_dir, self.audio_folder,
                                    item_id + postfix)
            feats = self.extract_whisper_features(wav_path)
            self.whisper_feats[item_id] = feats

    def __getitem__(self, index):
        data_dict = self.text_data[index]

        wav_path = os.path.join(self.cv_dir, data_dict['src_lang'],
                                self.audio_folder,
                                data_dict['ids'].split('.')[0] + self.postfix)

        audio = whisper.load_audio(str(wav_path))
        audio = torch.from_numpy(audio)
        audio_len = audio.shape[0]
        audio = whisper.pad_or_trim(audio)
        max_audio_len = audio.shape[0]

        # (80, 3000)
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(audio)

        max_mel_len = mel.shape[1]
        mel_features_len = int(audio_len / max_audio_len * max_mel_len)
        features_len = mel_features_len // 2
        data_dict['audio_tokens'] = mel
        data_dict['audio_lens'] = features_len
        return data_dict
