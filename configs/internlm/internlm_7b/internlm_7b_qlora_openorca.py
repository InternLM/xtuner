from mmengine.config import read_base

from mmchat.engine import LogSampleHook, SampleGenerateHook
from mmchat.utils import PROMPT_TEMPLATE

with read_base():
    from ..._base_.datasets.open_orca import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403
    from ..._base_.models.internlm_7b_qlora import *  # noqa: F401,F403
    from ..._base_.schedules.cosine_e1 import *  # noqa: F401,F403

train_dataloader.dataset.tokenizer = tokenizer  # noqa: F405

custom_hooks = [
    dict(type=LogSampleHook, tokenizer=tokenizer),  # noqa: F405
    dict(
        type=SampleGenerateHook,
        tokenizer=tokenizer,  # noqa: F405
        every_n_iters=5000,
        sample_inputs=[
            'Please explain AI to me.',
            'Please tell me five scenic spots in London.'
        ],
        instruction=PROMPT_TEMPLATE.alpaca.INSTRUCTION_START)
]
