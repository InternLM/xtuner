import random

RECOMPUTE_MODULES = ('InternLM2DecoderLayer', 'CLIPEncoderLayer')


def checkpoint_check_fn(submodule, target=RECOMPUTE_MODULES, selective=1.0):
    ret = False
    if type(submodule).__name__ in target:
        if random.uniform(0, 1) < selective:
            ret = True
    return ret
