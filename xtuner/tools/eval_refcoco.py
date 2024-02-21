# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import re
import time

import torch
import tqdm
from mmengine import Config
from mmengine.dist import get_dist_info, init_dist
from mmengine.utils.dl_utils import set_multi_processing
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import CLIPImageProcessor, GenerationConfig

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.refcoco_json import RefCOCOJsonEvalDataset
from xtuner.model.llava import LLaVAModel
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.registry import BUILDER
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def skip_init():

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


skip_init()


def merge_outputs(otuputs):
    new_outputs = [None for _ in range(dist.get_world_size())]

    assert dist.is_initialized()

    dist.all_gather_object(new_outputs, otuputs)
    new_dict = []
    for output in new_outputs:
        new_dict.extend(output)
    return new_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Refcoco Eval')
    parser.add_argument('--config', help='Path of model config.')
    parser.add_argument('--pth-path', default=None, help='Model pth path')
    args = parser.parse_args()
    return args


def load_model(config: Config, pth_path):
    model: LLaVAModel = BUILDER.build(config.model)
    state_dict = torch.load(
        pth_path + '/mp_rank_00_model_states.pt',
        map_location='cpu',
    )['module']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


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
        answer = [int(x) for x in re.findall(r'\d+', answer['ans'])]
        iou = computeIoU(answer, bbox)
        print(bbox, answer)
        if iou > 0.5:
            right += 1
    return right / len(answers)


@torch.no_grad()
def main():
    # init
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.launcher != 'none':
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    print(f'Rank: {rank} / World size: {world_size}')
    device = torch.device(f'cuda:{dist.get_rank()}')
    config = Config.fromfile(args.config)

    # load model
    model = load_model(config, args.pth_path)
    model.eval()
    tokenizer = BUILDER.build(config.tokenizer)

    # dataset
    dataset = RefCOCOJsonEvalDataset(
        data_path='data/llava_data/RefCOCOJson/eval_data/refcoco_testA.json',
        image_folder='data/llava_data/llava_images/coco/train2017/',
        tokenizer=tokenizer,
        image_processor=CLIPImageProcessor.from_pretrained(
            'openai/clip-vit-large-patch14-336'),
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

    # generation config
    gen_config = GenerationConfig(
        max_new_tokens=100,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id
                      is not None else tokenizer.eos_token_id),
    )
    stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=['</s>'])
    # generation
    answers = []
    for i, data in tqdm.tqdm(enumerate(loader), desc=f'Rank {rank}'):
        t0 = time.time()
        # prepare inputs
        inputs = data['conversation'][0]['input'][0]
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

        visual_outputs = model.visual_encoder(
            data['pixel_values'].to(device), output_hidden_states=True)
        pixel_values = model.projector(
            visual_outputs.hidden_states[model.visual_select_layer][:, 1:])
        data['pixel_values'] = pixel_values
        data['input_ids'] = ids
        datax = prepare_inputs_labels_for_multimodal(
            llm=model.llm.to(device),
            input_ids=data['input_ids'].to(device),
            pixel_values=data['pixel_values'].to(device),
        )

        # generation
        generation = model.llm.generate(
            **datax,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria,
        )
        answer = tokenizer.decode(generation[0])
        answers.append({
            'ans': answer,
            'id': data['id'][0],
            'bbox': torch.tensor(data['bbox']).tolist(),
        })
        if i % 100 == 0:
            print(f'{i}/{len(dataset)}: {time.time()-t0}, {answer}')
    merged_outputs = merge_outputs(answers)
    acc = eval_iou(merged_outputs)
    print(f'Acc: {acc}')


if __name__ == '__main__':
    main()
