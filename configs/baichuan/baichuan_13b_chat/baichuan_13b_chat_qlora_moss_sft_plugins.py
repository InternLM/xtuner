from mmengine.config import read_base

from mmchat.engine import LogSampleHook, SampleGenerateHook

with read_base():
    from ..._base_.datasets.moss_003_sft_plugins import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403
    from ..._base_.models.baichuan_13b_chat_qlora import *  # noqa: F401,F403
    from ..._base_.schedules.cosine_e1 import *  # noqa: F401,F403

bot_name = 'Baichuan'
train_dataloader.dataset.tokenizer = tokenizer  # noqa: F405
train_dataloader.dataset.bot_name = bot_name  # noqa: F405

custom_hooks = [
    dict(type=LogSampleHook, tokenizer=tokenizer),  # noqa: F405
    dict(
        type=SampleGenerateHook,
        tokenizer=tokenizer,  # noqa: F405
        every_n_iters=500,
        sample_inputs=[
            '一个球体的表面积是384平方厘米，求它的体积。', '今有鸡兔同笼，上有二十头，下有六十二足， 问鸡兔各几何？',
            '介绍一下比尔盖茨'
        ],
        instruction=(f'You are an AI assistant whose name is {bot_name}.\n'
                     f'Capabilities and tools that {bot_name} can possess.\n'
                     '- Inner thoughts: enabled.\n'
                     '- Web search: enabled. API: Search(query)\n'
                     '- Calculator: enabled. API: Calculate(expression)\n'
                     '- Equation solver: enabled. API: Solve(equation)\n'
                     '- Text-to-image: disabled.\n'
                     '- Image edition: disabled.\n'
                     '- Text-to-speech: disabled.\n'
                     '<|Human|>: {input}<eoh>\n'))
]
