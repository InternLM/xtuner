# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys

import requests

try:
    SERPER_API_KEY = os.environ['SERPER_API_KEY']
except Exception:
    print('Please obtain the `SERPER_API_KEY` from https://serper.dev and '
          'set it using `export SERPER_API_KEY=xxx`.')
    sys.exit(1)


def parse_results(results, k=10):
    snippets = []

    for result in results['organic'][:k]:
        if 'snippet' in result:
            snippets.append(result['snippet'])
        for attribute, value in result.get('attributes', {}).items():
            snippets.append(f'{attribute}: {value}.')
    return snippets


def search(api_key, search_term, **kwargs):
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json',
    }
    params = {
        'q': search_term,
        **{key: value
           for key, value in kwargs.items() if value is not None},
    }
    try:
        response = requests.post(
            'https://google.serper.dev/search',
            headers=headers,
            params=params,
            timeout=5)
    except Exception as e:
        return -1, str(e)
    return response.status_code, response.json()


def Search(q, k=10):
    status_code, response = search(SERPER_API_KEY, q)
    if status_code != 200:
        ret = 'None\n'
    else:
        text = parse_results(response, k=k)
        ret = ''
        for idx, res in enumerate(text):
            ret += f"<|{idx+1}|>: '{res}'\n"
    return ret
