from xtuner.model import OpenAIModel
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel)

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import mm_collate_fn1
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.model import LLaVAModel
from peft import LoraConfig
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.dataset.evaluation import MMEDataset, MultipleChoiceDataset, POPEDataset, \
    HallusionDataset, TextVQADataset, GQADataset, VQAv2Dataset
from xtuner.dataset import ConcatDataset
from xtuner.engine.runner import TrainLoop, ValLoop, TestLoop
from xtuner.dataset import LLaVAProxyEvalDataset1

llm_name_or_path = '/mnt/petrelfs/share_data/gaojianfei/Phi-3-mini-4k-instruct/models--microsoft--Phi-3-mini-4k-instruct/snapshots/3a811845d89f3c1b3f41b341d0f9f05104769f35'
visual_encoder_name_or_path = 'model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
model = dict(type=OpenAIModel, base_url='http://10.140.24.142:23333/v1')
prompt_template = None

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

proxy_eval_dataset = dict(type=LLaVAProxyEvalDataset1)

test_dataset = [
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMBench_DEV_EN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/SEEDBench_IMG.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/ScienceQA_VAL.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/ScienceQA_TEST.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMMU_DEV_VAL.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/AI2D_TEST.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=TextVQADataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/share_data/huanghaian/orig_llava_eval/textvqa/llava_textvqa_val_v051_ocr.jsonl',
        ann_file='/mnt/petrelfs/share_data/huanghaian/text_vqa/TextVQA_0.5.1_val.json',
        image_folder='/mnt/petrelfs/share_data/huanghaian/text_vqa/train_images',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MMEDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MME.tsv',
        image_folder='/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        # for_llava_prompt=True, # 开了后，perception 会掉
        pad_image_to_square=True),
    dict(
        type=HallusionDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/HallusionBench.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=POPEDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file=[
            '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_adversarial.json',
            '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_popular.json',
            '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_random.json'
        ],
        coco_val_path='/mnt/petrelfs/share_data/linzhihao/dataset/coco/val2014/',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=GQADataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/share_data/zhaoxiangyu/gqa_llava_eval/llava_gqa_testdev_balanced.jsonl',
        ann_file='/mnt/petrelfs/share_data/zhaoxiangyu/gqa_llava_eval/testdev_balanced_questions.json',
        image_folder='/mnt/petrelfs/share_data/basemodel/dataset/multimodality/gqa/images',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/share_data/zhaoxiangyu/datasets--Lin-Chen--MMStar/snapshots/mmstar/MMStar.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMBench_DEV_CN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/CCBench.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMBench_TEST_CN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=proxy_eval_dataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMBench_TEST_EN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    # dict(
    #     type=VQAv2Dataset,
    #     proxy_eval_dataset = proxy_eval_dataset,
    #     data_file='/mnt/petrelfs/share_data/zhaoxiangyu/vqav2_llava_eval/llava_vqav2_mscoco_test-dev2015.jsonl',
    #     test_file='/mnt/petrelfs/share_data/zhaoxiangyu/vqav2_llava_eval/llava_vqav2_mscoco_test2015.jsonl',
    #     image_folder='/mnt/petrelfs/share_data/zhaoxiangyu/vqav2_test2015',
    #     prompt_template=PROMPT_TEMPLATE.vicuna,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     pad_image_to_square=True),
]

# # TODO: We are not currently using test_evaluator
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=test_dataset),
    collate_fn=dict(type=mm_collate_fn1, extra_collate_keys=['img_id'])
)

test_evaluator = {}
test_cfg = dict(type=TestLoop, select_metric='first')
