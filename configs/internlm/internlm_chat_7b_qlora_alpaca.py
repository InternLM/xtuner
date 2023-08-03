from mmengine.config import read_base

from mmchat.engine import LogSampleHook

with read_base():
    from .._base_.datasets.alpaca_enzh import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.models.internlm_chat_7b_qlora import *  # noqa: F401,F403
    from .._base_.schedules.cosine_e1 import *  # noqa: F401,F403

train_dataloader.dataset.tokenizer = tokenizer  # noqa: F405

custom_hooks = [dict(type=LogSampleHook, tokenizer=tokenizer)]  # noqa: F405
