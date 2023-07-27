import torch
from mmengine.config import read_base
from mmengine.model import BaseDataPreprocessor
from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from mmchat.engine import SampleGenerateHook
from mmchat.models import SupervisedQloraFinetune

with read_base():
    from .._base_.datasets.arxiv import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.internlm import *  # noqa: F401,F403

pretrained_model_name_or_path = './models/internlm-7b'
model = dict(
    type=SupervisedQloraFinetune,
    data_preprocessor=dict(type=BaseDataPreprocessor),
    llm=dict(
        type=AutoModel.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    use_fast=False,
    padding_side='right',
    trust_remote_code=True)

train_dataloader.dataset.tokenizer = tokenizer  # noqa: F405

default_hooks.checkpoint.update(  # noqa: F405
    dict(by_epoch=False, interval=500, max_keep_ckpts=2))

custom_hooks = [
    dict(
        type=SampleGenerateHook,
        tokenizer=tokenizer,
        every_n_iters=500,
        sample_inputs=[
            'We present InternLM, a multilingual foundational language model '
            'with 104B parameters. InternLM is pre-trained on a large corpora '
            'with 1.6T tokens with a multi-phase progressive process, and '
            'then fine-tuned to align with human preferences. We also '
            'developed a training system called Uniscale-LLM for efficient '
            'large language model training. The evaluation on a number of '
            'benchmarks shows that InternLM achieves state-of-the-art '
            'performance in multiple aspects, including knowledge '
            'understanding, reading comprehension, mathematics, and coding. '
            'With such well-rounded capabilities, InternLM achieves '
            'outstanding performances on comprehensive exams, including '
            'MMLU, AGIEval, C-Eval and GAOKAO-Bench, without resorting to '
            'external tools. On these benchmarks, InternLM not only '
            'significantly outperforms open-source models, but also obtains '
            'superior performance compared to ChatGPT. Also, InternLM '
            'demonstrates excellent capability of understanding Chinese '
            'language and Chinese culture, which makes it a suitable '
            'foundation model to support Chinese-oriented language '
            'applications. This manuscript gives a detailed study of '
            'our results, with benchmarks and examples across a diverse '
            'set of knowledge domains and tasks.',
            'In this work, we develop and release Llama 2, a collection of '
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
            'development of LLMs.',
        ],
        prompt='If you are an expert in writing papers, please generate '
        "a good paper title for this paper based on other authors' "
        'descriptions of their abstracts.\n\n'
        '### Descriptions:\n{input}\n\n### Title: ')
]
