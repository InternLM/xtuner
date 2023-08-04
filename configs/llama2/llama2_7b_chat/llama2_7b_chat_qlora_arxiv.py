from mmengine.config import read_base

from mmchat.engine import LogSampleHook, SampleGenerateHook

with read_base():
    from ..._base_.datasets.arxiv import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403
    from ..._base_.models.llama2_7b_chat_qlora import *  # noqa: F401,F403
    from ..._base_.schedules.cosine_e3 import *  # noqa: F401,F403

train_dataloader.dataset.tokenizer = tokenizer  # noqa: F405

custom_hooks = [
    dict(type=LogSampleHook, tokenizer=tokenizer),  # noqa: F405
    dict(
        type=SampleGenerateHook,
        tokenizer=tokenizer,  # noqa: F405
        every_n_iters=500,
        sample_inputs=[
            ('We present InternLM, a multilingual foundational language '
             'model with 104B parameters. InternLM is pre-trained on a large '
             'corpora with 1.6T tokens with a multi-phase progressive '
             'process, and then fine-tuned to align with human preferences. '
             'We also developed a training system called Uniscale-LLM for '
             'efficient large language model training. The evaluation on a '
             'number of benchmarks shows that InternLM achieves '
             'state-of-the-art performance in multiple aspects, including '
             'knowledge understanding, reading comprehension, mathematics, '
             'and coding. With such well-rounded capabilities, InternLM '
             'achieves outstanding performances on comprehensive exams, '
             'including MMLU, AGIEval, C-Eval and GAOKAO-Bench, without '
             'resorting to external tools. On these benchmarks, InternLM '
             'not only significantly outperforms open-source models, but '
             'also obtains superior performance compared to ChatGPT. Also, '
             'InternLM demonstrates excellent capability of understanding '
             'Chinese language and Chinese culture, which makes it a '
             'suitable foundation model to support Chinese-oriented language '
             'applications. This manuscript gives a detailed study of '
             'our results, with benchmarks and examples across a diverse '
             'set of knowledge domains and tasks.'),
            ('In this work, we develop and release Llama 2, a collection of '
             'pretrained and fine-tuned large language models (LLMs) ranging '
             'in scale from 7 billion to 70 billion parameters.\nOur '
             'fine-tuned LLMs, called LLAMA 2-CHAT, are optimized for '
             'dialogue use cases. Our models outperform open-source chat '
             'models on most benchmarks we tested, and based on our human '
             'evaluations for helpfulness and safety, may be a suitable '
             'substitute for closedsource models. We provide a detailed '
             'description of our approach to fine-tuning and safety '
             'improvements of LLAMA 2-CHAT in order to enable the community '
             'to build on our work and contribute to the responsible '
             'development of LLMs.')
        ],
        instruction=('If you are an expert in writing papers, please generate '
                     'a good paper title for this paper based on other '
                     "authors' descriptions of their abstracts.\n\n"
                     '### Descriptions:\n{input}\n\n### Title: '))
]
