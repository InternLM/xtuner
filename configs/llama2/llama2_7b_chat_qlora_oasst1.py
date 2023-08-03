from mmengine.config import read_base

from mmchat.engine import LogSampleHook

with read_base():
    from .._base_.datasets.oasst1 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.models.llama2_7b_chat_qlora import *  # noqa: F401,F403
    from .._base_.schedules.cosine_e3 import *  # noqa: F401,F403

train_dataloader.dataset.tokenizer = tokenizer  # noqa: F405

custom_hooks = [dict(type=LogSampleHook, tokenizer=tokenizer)]  # noqa: F405
