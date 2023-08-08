# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/OpenLMLab/MOSS_WebSearchTool
import heapq
import json
import os
import re
import sys
from collections import Counter
from string import punctuation
from urllib import parse

import cv2  # noqa: F401
import fasttext
import requests
import spacy
from bs4 import BeautifulSoup
from numpy import dot
from numpy.linalg import norm

try:
    SERPER_KEY = os.environ['SERPER_KEY']
except Exception:
    print('Please obtain the `SERPER_KEY` from https://serper.dev and '
          'set it using `export SERPER_KEY=xxx`.')
    sys.exit(1)

fasttext.FastText.eprint = lambda x: None

cwd_path = os.getcwd()
try:
    ft_en = fasttext.load_model(os.path.join(cwd_path, 'models/cc.en.300.bin'))
except Exception:
    print('Please download and gunzip `cc.en.300.bin` from '
          'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/'
          f'cc.en.300.bin.gz, and put it in `{cwd_path}/models/`.')
    sys.exit(1)
try:
    ft_zh = fasttext.load_model(os.path.join(cwd_path, 'models/cc.zh.300.bin'))
except Exception:
    print('Please download and gunzip `cc.zn.300.bin` from '
          'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/'
          f'cc.zh.300.bin.gz, and put it in `{cwd_path}/models/`.')
    sys.exit(1)

try:
    nlp_en = spacy.load('en_core_web_sm')
except Exception:
    print('Download and install en_core_web_sm by '
          'running `python -m spacy download en_core_web_sm`...')
    os.system('python -m spacy download en_core_web_sm')
    nlp_en = spacy.load('en_core_web_sm')

try:
    nlp_zh = spacy.load('zh_core_web_sm')
except Exception:
    print('Download and install zh_core_web_sm by '
          'running `python -m spacy download zh_core_web_sm`...')
    os.system('python -m spacy download zh_core_web_sm')
    nlp_zh = spacy.load('zh_core_web_sm')


def score(key_words, sentence, ft):
    res = 0
    for key_word in key_words.split():
        key_embedding = ft.get_word_vector(key_word)
        vector = ft.get_word_vector(sentence)
        cos_sim = dot(key_embedding, vector) / (
            norm(key_embedding) * norm(vector))
        res += cos_sim
    return res


def score_2(key_words, sentence):
    res = 0
    for key_word in key_words.split():
        res += 1 if sentence.find(key_word) > 0 else 0
    return res


def score_3(key_words, sentence, measure):
    if sentence.split():
        return measure.infer(sentence, key_words)
    else:
        return -10000


def top_sentence(text, limit, nlp):
    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    doc = nlp(text.lower())
    for token in doc:
        if (token.text in nlp.Defaults.stop_words
                or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            keyword.append(token.text)

    freq_word = Counter(keyword)
    max_freq = Counter(keyword).most_common(1)[0][1]
    for w in freq_word:
        freq_word[w] = (freq_word[w] / max_freq)
    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += freq_word[word.text]
                else:
                    sent_strength[sent] = freq_word[word.text]
    summary = []

    sorted_x = sorted(
        sent_strength.items(), key=lambda kv: kv[1], reverse=True)

    counter = 0
    for i in range(len(sorted_x)):
        summary.append(str(sorted_x[i][0]).capitalize())
        counter += 1
        if (counter >= limit):
            break
    return summary


def containenglish(str0):
    return bool(re.search('[a-z]', str0))


def clean_html(html: str) -> str:
    """Remove HTML markup from the given string."""
    # Remove inline JavaScript/CSS, HTML comments, and HTML tags
    cleaned_html = re.sub(
        r'(?is)<(script|style).*?>.*?(</\1>)|<!--(.*?)-->[\n]?|<(?s).*?>', '',
        html.strip())

    # Deal with whitespace and HTML entities
    cleaned_html = re.sub(r'&nbsp;|  |\t|&.*?;[0-9]*&.*?;|&.*?;', '',
                          cleaned_html)

    return cleaned_html.strip()


def get_web_response(url):
    try:
        response = requests.get(url=url, timeout=5)
        response.encoding = 'utf-8'
        return response
    except requests.exceptions.RequestException:
        return None


def extract_description(soup):
    description = soup.find(attrs={'name': 'description'})
    if description:
        content = description.get('content')
        if content:
            return content
    return None


def summ_web(q, url, ft_en, ft_zh, is_eng, nlp_en, nlp_zh, measure_en,
             measure_zh, snippet, title):
    url = parse.unquote(url)

    response = get_web_response(url)
    if response is None:
        return {
            'title': title,
            'url': url,
            'summ': snippet,
            'note': 'fail to get ... use snippet',
            'type': 'snippet'
        }

    soup = BeautifulSoup(response.text, 'html.parser')
    description = extract_description(soup)

    if description:
        if all(key_word in description for key_word in q.split()):
            return {
                'title': title,
                'url': url,
                'summ': description,
                'note': 'use description as summ',
                'type': 'description'
            }

    text = clean_html(response.text)
    sentences = re.split('\n|。|\\.', text)

    ft = ft_en if is_eng else ft_zh
    measure = measure_en if is_eng else measure_zh
    nlp = nlp_en if is_eng else nlp_zh

    scored_sentences = []
    for sentence in sentences:
        if 3 <= len(sentence) <= 200:
            scored_sentence = {
                'ft':
                -1 * score(q, sentence, ft) if ft else None,
                'score_2':
                -1 * score_2(q, sentence),
                'measure':
                -1 *
                score_3(q, sentence, measure=measure) if measure else None,
                'sentence':
                sentence
            }
            scored_sentences.append(scored_sentence)

    top_sentences = heapq.nsmallest(
        5, scored_sentences, key=lambda x: x['ft'] or float('inf')
    ) + heapq.nsmallest(
        10, scored_sentences, key=lambda x: x['score_2']) + heapq.nsmallest(
            5, scored_sentences, key=lambda x: x['measure'] or float('inf'))

    stop_word = '.' if is_eng else '。'
    combined_text = stop_word.join(
        [sentence['sentence'] for sentence in top_sentences])

    if len(combined_text) < 3:
        return {
            'title': title,
            'url': url,
            'summ': snippet,
            'note': 'bad web, fail to summ, use snippet,',
            'type': 'snippet'
        }

    try:
        summary = top_sentence(text=combined_text, limit=3, nlp=nlp)
        summary = ''.join(summary)
    except Exception:
        return {
            'title': title,
            'url': url,
            'summ': snippet,
            'note': 'unknown summ error , use snippet',
            'type': 'snippet'
        }

    if any(key_word in summary for key_word in q.split()):
        return {
            'title': title,
            'url': url,
            'summ': summary,
            'note': 'good summ and use it',
            'type': 'my_summ'
        }

    return {
        'title': title,
        'url': url,
        'summ': snippet,
        'note': 'poor summ , use snippet',
        'type': 'snippet'
    }


def search_api(q, SERPER_KEY):
    url = 'https://google.serper.dev/search'

    if containenglish(q):
        payload = json.dumps({
            'q': q,
        })
    else:
        payload = json.dumps({'q': q})
    headers = {'X-API-KEY': SERPER_KEY, 'Content-Type': 'application/json'}

    response = requests.request('POST', url, headers=headers, data=payload)

    response_dict = json.loads(response.text)

    return response_dict


def filter_urls(urls, snippets, titles, black_list=None, topk=3):
    if black_list is None:
        black_list = ['enoN, youtube.com, bilibili.com', 'zhihu.com']

    filtered_urls, filtered_snippets, filtered_titles = [], [], []
    count = 0
    for url, snippet, title in zip(urls, snippets, titles):
        if all(domain not in url
               for domain in black_list) and url.split('.')[-1] != 'pdf':
            filtered_urls.append(url)
            filtered_snippets.append(snippet)
            filtered_titles.append(title)
            count += 1
            if count >= topk:
                break

    return filtered_urls, filtered_snippets, filtered_titles


def engine(q,
           SERPER_KEY,
           ft_en,
           ft_zh,
           nlp_en,
           nlp_zh,
           measure_en,
           measure_zh,
           topk=3):
    is_eng = containenglish(q)

    response = search_api(q, SERPER_KEY)

    raw_urls = [i['link'] for i in response['organic']]
    raw_snippets = [i['snippet'] for i in response['organic']]
    raw_titles = [i['title'] for i in response['organic']]
    urls, snippets, titles = filter_urls(
        raw_urls, raw_snippets, raw_titles, topk=topk)

    results = {}
    for i, url in enumerate(urls):
        try:
            summ = summ_web(q, url, ft_en, ft_zh, is_eng, nlp_en, nlp_zh,
                            measure_en, measure_zh, snippets[i], titles[i])
        except Exception:
            summ = {
                'url': url,
                'summ': snippets[i],
                'note': 'unbelievable error, use snippet !',
                'type': 'snippet',
                'title': titles[i]
            }

        results[str(i)] = summ

    return results


def Search(q):
    results = engine(q, SERPER_KEY, ft_en, ft_zh, nlp_en, nlp_zh, None, None)
    ret = ''
    for idx, (k, v) in enumerate(results.items()):
        ret += f"<|{idx+1}|>: '{v['title']}. {v['summ']}'\n"
    return ret
