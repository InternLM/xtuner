# Copyright (c) OpenMMLab. All rights reserved.
from math import *  # noqa: F401, F403


def Calculate(expression):
    res = ''
    for exp in expression.split(';'):
        try:
            res += '{:.2f};'.format(eval(exp.replace('^', '**')))
        except Exception:
            res += 'No result.'
    if res[-1] == ';':
        res = res[:-1]
    return res
