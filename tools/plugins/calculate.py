from math import *  # noqa: F401, F403


def Calculate(expression):
    try:
        return '{:.2f}'.format(eval(expression.replace('^', '**')))
    except Exception:
        return 'No result.'
