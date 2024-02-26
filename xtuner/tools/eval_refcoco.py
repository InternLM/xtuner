# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re

import torch
import tqdm
from huggingface_hub import snapshot_download
from mmengine.dist import get_dist_info, init_dist, master_only
from mmengine.utils.dl_utils import set_multi_processing
from peft import PeftModel
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.refcoco_json import RefCOCOJsonEvalDataset
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def merge_outputs(otuputs):
    new_outputs = [None for _ in range(dist.get_world_size())]

    assert dist.is_initialized()

    dist.all_gather_object(new_outputs, otuputs)
    new_dict = []
    for output in new_outputs:
        new_dict.extend(output)
    return new_dict


@master_only
def master_print(msg):
    print(msg)


def parse_args():
    parser = argparse.ArgumentParser(description='MMBench')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')
    parser.add_argument('--data-path', default=None, help='data path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument('--llava', default=None, help='llava name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template',
    )
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.',
    )
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).',
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=100,
        help='Maximum number of new tokens allowed in generated text',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    args = parser.parse_args()
    return args


def eval_iou(answers):

    def computeIoU(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        intersection_x1 = max(x1, x3)
        intersection_y1 = max(y1, y3)
        intersection_x2 = min(x2, x4)
        intersection_y2 = min(y2, y4)
        intersection_area = max(0,
                                intersection_x2 - intersection_x1 + 1) * max(
                                    0, intersection_y2 - intersection_y1 + 1)
        bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
        union_area = bbox1_area + bbox2_area - intersection_area
        iou = intersection_area / union_area
        return iou

    right = 0
    for answer in answers:
        bbox = answer['bbox']
        bbox = RefCOCOJsonEvalDataset.normalize_bbox(bbox, answer['height'],
                                                     answer['width'])
        answer_bbox = [int(x) for x in re.findall(r'\d+', answer['ans'])]
        if len(answer_bbox) == 4:
            iou = computeIoU(answer_bbox, bbox)
            if iou > 0.5:
                right += 1
        else:
            print('Error format sample: ', answer)
    return right / len(answers)


def build_model(args):
    rank, world_size = get_dist_info()
    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': rank if world_size > 1 else 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype],
    }

    # build llm
    with LoadWoInit():
        llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                   **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        encode_special_tokens=True)
    master_print(f'Load LLM from {args.model_name_or_path}')

    llava_path = (
        snapshot_download(
            repo_id=args.llava) if not osp.isdir(args.llava) else args.llava)

    # build visual_encoder
    if 'visual_encoder' in os.listdir(llava_path):
        assert args.visual_encoder is None, (
            "Please don't specify the `--visual-encoder` since passed "
            '`--llava` contains a visual encoder!')
        visual_encoder_path = osp.join(llava_path, 'visual_encoder')
    else:
        assert (args.visual_encoder is not None
                ), 'Please specify the `--visual-encoder`!'  # noqa: E501
        visual_encoder_path = args.visual_encoder
    with LoadWoInit():
        visual_encoder = CLIPVisionModel.from_pretrained(
            visual_encoder_path, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
        image_processor = CLIPImageProcessor.from_pretrained(
            visual_encoder_path)
    master_print(f'Load visual_encoder from {visual_encoder_path}')

    # load adapter
    if 'llm_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'llm_adapter')

        with LoadWoInit():
            llm = PeftModel.from_pretrained(
                llm, adapter_path, offload_folder=args.offload_folder)

        master_print(f'Load LLM adapter from {args.llava}')

    if 'visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
        visual_encoder = PeftModel.from_pretrained(
            visual_encoder, adapter_path, offload_folder=args.offload_folder)
        master_print(f'Load visual_encoder adapter from {args.llava}')

    # build projector
    projector_path = osp.join(llava_path, 'projector')
    with LoadWoInit():
        projector = AutoModel.from_pretrained(
            projector_path, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
    master_print(f'Load projector from {args.llava}')

    projector.cuda()
    projector.eval()

    visual_encoder.cuda()
    visual_encoder.eval()

    llm.eval()
    return llm, visual_encoder, projector, tokenizer, image_processor


def generate(
    llm,
    visual_encoder,
    projector,
    tokenizer,
    samples,
    visual_select_layer,
):
    gen_config = GenerationConfig(
        max_new_tokens=100,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id
                      is not None else tokenizer.eos_token_id),
    )
    stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=['</s>'])

    device = next(llm.parameters()).device
    # prepare inputs
    inputs = samples['conversation'][0]['input'][0]
    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    assert len(chunk_encode) == 2
    ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    ids = torch.tensor(ids).cuda().unsqueeze(0)

    visual_outputs = visual_encoder(
        samples['pixel_values'].to(device), output_hidden_states=True)
    pixel_values = projector(
        visual_outputs.hidden_states[visual_select_layer][:, 1:])
    samples['pixel_values'] = pixel_values
    samples['input_ids'] = ids
    datax = prepare_inputs_labels_for_multimodal(
        llm=llm.to(device),
        input_ids=samples['input_ids'].to(device),
        pixel_values=samples['pixel_values'].to(device),
    )

    # generation
    generation = llm.generate(
        **datax,
        generation_config=gen_config,
        streamer=None,
        bos_token_id=tokenizer.bos_token_id,
        stopping_criteria=stop_criteria,
    )
    answer = tokenizer.decode(generation[0])
    return {
        'ans': answer,
        'id': samples['id'][0],
        'bbox': torch.tensor(samples['bbox']).tolist(),
        'height': samples['height'],
        'width': samples['width'],
    }


@torch.no_grad()
def main():
    # init
    args = parse_args()
    if args.launcher != 'none':
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    print(f'Rank: {rank} / World size: {world_size}')

    # build_model
    llm, visual_encoder, projector, tokenizer, image_processor = build_model(
        args)

    # dataset
    dataset = RefCOCOJsonEvalDataset(
        data_path=args.data_path,
        image_folder='data/llava_data/llava_images/',
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_dataset_length=None,
        dataset_map_fn=llava_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=PROMPT_TEMPLATE.vicuna),
        max_length=2048,
        pad_image_to_square=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=False, seed=0),
    )
    loader.sampler.set_epoch(0)

    answers = []
    for i, data in tqdm.tqdm(enumerate(loader), desc=f'Rank {rank}'):
        answer = generate(
            llm,
            visual_encoder,
            projector,
            tokenizer,
            data,
            args.visual_select_layer,
        )
        answers.append(answer)

    merged_outputs = merge_outputs(answers)
    acc = eval_iou(merged_outputs)
    master_print(f'Acc: {acc}')


if __name__ == '__main__':
    main()
