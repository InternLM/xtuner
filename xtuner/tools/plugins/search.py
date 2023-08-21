# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys

from lagent.actions import GoogleSearch
from lagent.schema import ActionStatusCode

try:
    SERPER_KEY = os.environ['SERPER_KEY']
except Exception:
    print('Please obtain the `SERPER_KEY` from https://serper.dev and '
          'set it using `export SERPER_KEY=xxx`.')
    sys.exit(1)

search_engine = GoogleSearch(SERPER_KEY)


def Search(q):
    action_return = search_engine(q)
    if action_return.state == ActionStatusCode.SUCCESS:
        response = eval(action_return.result['text'])
    else:
        response = action_return.errmsg

    ret = ''
    for idx, res in enumerate(response):
        ret += f"<|{idx+1}|>: '{res}'\n"
    return ret
