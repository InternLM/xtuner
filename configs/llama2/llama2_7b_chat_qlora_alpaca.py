from mmengine.config import read_base

from mmchat.engine import LogSampleHook

with read_base():
    from .._base_.datasets.alpaca_enzh import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.models.llama2_7b_chat_qlora import *  # noqa: F401,F403
    from .._base_.schedules.cosine import *  # noqa: F401,F403

train_dataloader.dataset.tokenizer = tokenizer  # noqa: F405

default_hooks.checkpoint.update(  # noqa: F405
    dict(by_epoch=False, interval=500, max_keep_ckpts=2))

custom_hooks = [dict(type=LogSampleHook, tokenizer=tokenizer)]  # noqa: F405
