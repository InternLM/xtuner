from mmengine.config import read_base

from mmchat.engine import LogSampleHook, SampleGenerateHook

with read_base():
    from ..._base_.datasets.cmd import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403
    from ..._base_.models.baichuan_7b_qlora import *  # noqa: F401,F403
    from ..._base_.schedules.cosine_e1 import *  # noqa: F401,F403

train_dataloader.dataset.tokenizer = tokenizer  # noqa: F405

custom_hooks = [
    dict(type=LogSampleHook, tokenizer=tokenizer),  # noqa: F405
    dict(
        type=SampleGenerateHook,
        tokenizer=tokenizer,  # noqa: F405
        every_n_iters=500,
        sample_inputs=[
            '我有家族遗传性的过敏，请问可以可以献血吗？', '我爷爷有高血压，请问他可以喝咖啡吗？',
            '我女儿今年3岁了，从昨天晚上九点开始腹泻，到现在已经八个小时了，请问应该怎么办？'
        ],
        instruction=('请从一名专业医生的角度，对下述医学问题给出安全、可靠的回答。\n\n'
                     '问：{input}\n\n答：'))
]
