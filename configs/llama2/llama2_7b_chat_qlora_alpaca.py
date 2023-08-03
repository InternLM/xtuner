from mmengine.config import read_base

from mmchat.engine import LogSampleHook, SampleGenerateHook

with read_base():
    from .._base_.datasets.alpaca_enzh import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.models.llama2_7b_chat_qlora import *  # noqa: F401,F403
    from .._base_.schedules.cosine_e1 import *  # noqa: F401,F403

train_dataloader.dataset.datasets_kwargs.tokenizer = tokenizer  # noqa: F405

custom_hooks = [
    dict(type=LogSampleHook, tokenizer=tokenizer),  # noqa: F405
    dict(
        type=SampleGenerateHook,
        tokenizer=tokenizer,  # noqa: F405
        every_n_iters=500,
        sample_inputs=[
            '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
        ],
        instruction=(
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n'
            '### Instruction:\n{input}\n\n'
            '### Response: '))
]
