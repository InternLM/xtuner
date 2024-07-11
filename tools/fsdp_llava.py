# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from collections import OrderedDict
from datasets import load_from_disk
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from mmengine import mkdir_or_exist
from mmengine.dist import init_dist
import copy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (get_state_dict,
                                                     set_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, _or_policy
from xtuner._lite.accelerate.fsdp import dp_lazy_init, LoadWoInit, checkpoint_check_fn, layer_auto_wrap_policy, RECOMPUTE_MODULES, all_required_grad_wrap_policy, token_embedding_wrap_policy, set_require_grad_param_to_fp32

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader

from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import packed_sequence, dispatch_modules
from xtuner._lite.chat import ChatMessages, HybridChatTemplate
from xtuner._lite.datasets import (SoftPackerForLlava, LlavaTokenizedDataset, LlavaRawDataset, LlavaCollator, LlavaTokenizeFunction)
                                       
from xtuner._lite.parallel import ParallelSampler
from xtuner._lite.datasets.format import OPENAI_FORMAT_MAP
from xtuner._lite.chat import CHAT_TEMPLATE_MAP
from xtuner._lite.datasets.load import load_datasets, LOAD_FN_MAP
from mmengine.utils.dl_utils import collect_env
from mmengine.utils import get_git_hash
from mmengine.dist import infer_launcher
from mmengine.runner import set_random_seed
import shutil
from tqdm import tqdm
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor,AutoModelForCausalLM, CLIPVisionModel,LlavaForConditionalGeneration, LlavaConfig, AutoModel
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.utils.import_utils import is_torch_sdpa_available
from peft import LoraConfig, get_peft_model

from mmengine import load
logger = get_logger()
def layer_auto_wrap_policy(
    module,
    recurse: bool,
    nonwrapped_numel: int,
    layer_cls = ['InternLM2DecoderLayer', 'CLIPEncoderLayer', 'LlavaMultiModalProjector'],
) -> bool:
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for
        # the leaf node or reminder
        if module.__class__.__name__ in layer_cls:
            logger.info(module.__class__.__name__ )
        return module.__class__.__name__ in layer_cls


def log_format(rank, debug=False):

    formatter = f'[XTuner][RANK {rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Model Related Settings')
    model_args.add_argument(
        '--llm', help='repo id or local path of the model')
    model_args.add_argument(
        '--vit', 
        default='openai/clip-vit-large-patch14-336',
        help='repo id or local path of the model')
    model_args.add_argument(
        '-t',
        '--tokenizer',
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
    model_args.add_argument(
        '--chat-template',
        choices = CHAT_TEMPLATE_MAP.keys(),
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
    parser.add_argument(
        '--freeze-llm',
        action='store_true', help="Not updating LLM's parameters")
    parser.add_argument(
        '--freeze-vit',
        action='store_true', help="Not updating vit's parameters")
    
    model_args.add_argument(
        '--dtype', 
        default='auto', 
        choices=['fp16', 'bf16', 'auto'], 
        help=("the dtype of the model forward. When set to 'auto', it will "
              "automatically determine whether bf16 is available, "
              "prioritizing the use of bf16."))
    
    model_args.add_argument(
        '--selective-recompute',
        default=1.0,
        type=float,
        help=('the ratio of re-computation for transforemer layers. '
              'The maximum is 1; the larger the value, the less memory '
              'required for training. The default is 1, meaning all layers '
              'need to be re-computated.'))

    data_args = parser.add_argument_group('data', 'Dataset Related Settings')
    data_args.add_argument(
        '--datasets', 
        help=('repo id or local path or dir of the datasets. For repo ids, '
              'the `dset-sources` needs to be appropriately set to '
              '`modelscope` or `huggingface`. For local dir, all json and '
              'jsonl files will be loaded by default. The type of loaded '
              'files can be controlled by setting `dset-file-type`'))
    data_args.add_argument(
        '--dset-cache-dir',
        help=('the cache dir of the loaded datasets. When the `datasets` is '
              'set, the loaded datasets will be cached to this dir. If the '
              '`datasets` are not set, the cached dataset in this dir will be '
              'loaded.'))
    data_args.add_argument(
        '--dset-from-cache',
        action='store_true',
        help=('Load data directly from `dset-cache-dir`. This can save time '
              'on online tokenization, but if the tokenizer changed, '
              'recaching is needed.'))
    data_args.add_argument(
        '--dset-file-types',
        nargs='*',
        default=LOAD_FN_MAP.keys(),
        choices = LOAD_FN_MAP.keys(),
        help='the file type that needs to be loaded')
    data_args.add_argument(
        '--dset-pack-level',
        choices=['hard', 'soft'],
        help=('the level of data packing. When `hard`, multiple data will be '
              'packed to `max_length`, potentially causing some data to be '
              'truncated, and the length of the packed data will always '
              'be `max_length`; When `soft`, it will pack multiple  data '
              'into nearly `max_length` without truncating the data.'))
    data_args.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help=('the maximum length of each piece of data, any excess will be '
              'truncated.'))
    data_args.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='how many subprocesses to use for data loading.')

    optim_args = parser.add_argument_group('optim', 'Optim Related Settings')
    optim_args.add_argument(
        '--mirco-batch-size',
        type=int,
        default=1,
        help='batch size for each forward + backward pass')
    optim_args.add_argument(
        '--global-batch-size',
        type=int,
        default=16,
        help='batch size for each parameter update')

    optim_args.add_argument(
        '--lr', default=4e-5, type=float, help='learning rate.')
    optim_args.add_argument(
        '--wd', default=0.01, type=float, help='weight decay.')
    optim_args.add_argument(
        '--max-grad-norm', default=1, type=float, help='gradient clipping')
    optim_args.add_argument(
        '-e', '--epochs', default=1, type=int, help='total training epochs.')
    optim_args.add_argument(
        '--warmup-ratio',
        default=0.03,
        type=float,
        help=('the proportion of training steps for learning rate warm-up in '
              'relation to the total training steps.'))

    parser.add_argument('-c', '--config', default=None)
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and checkpoints')
    parser.add_argument(
        '--checkpoint-interval',
        default=0.25,
        type=float,
        help=('how many steps to save a checkpoint; it can be a floating '
              'point number less than 1, or an integer greater than or equal '
              "to 1. When it's a floating point, it will be multiplied by the "
              'total number of training steps.'))
    parser.add_argument(
        '--checkpoint-drop-optimizer',
        action='store_true',
        help=('only model parameters are saved when saving a checkpoint. '
              'This can significantly reduce the size of checkpoint files, '
              'but the saved checkpoints cannot be resumed.'))
    parser.add_argument('--log-interval', default=1, type=int, help='log interval')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed for the training')
    parser.add_argument(
        '--debug',
        action='store_true', help='Set logger level to `DEBUG`')
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps

def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map

# @logger.catch
def llava(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    if args.dset_pack_level and not is_flash_attn_2_available():
        raise NotImplementedError('If you want to use the pack dataset, you '
                                  'need to install `flash_attn`. Refer to '
                                  'https://github.com/Dao-AILab/flash-attention/releases')
    
    
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(args.seed)
    
    world_size = int(os.environ['WORLD_SIZE'])
    dp_size = world_size 
    
    
    device_mesh = init_device_mesh(
        'cuda', (dp_size, ), mesh_dim_names=('dp',))

    dp_mesh = device_mesh['dp']

    rank = dp_mesh.get_local_rank()
    
    mkdir_or_exist(args.work_dir)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    log_file = os.path.join(args.work_dir,
                            f'{timestamp}.rank{rank}.log')

    # Change the log format printed in the terminal
    lvl = 'DEBUG' if args.debug else 'INFO'
    logger.add(sys.stderr, level=lvl, format=log_format(rank, args.debug))
    # Change the format saved in the log file
    logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)
    
    logger.info(args)
    if rank == 0 :
        env = collect_env()
        import transformers, xtuner
        env['Transformers'] = transformers.__version__
        env['XTuner'] = f'{xtuner.__version__}+{get_git_hash(digits=6)}' 
        runtime_env = OrderedDict()
        runtime_env.update(env)
        runtime_env['Seed'] = args.seed
        runtime_env['World Size'] = world_size
        runtime_env['Distributed launcher'] = dist_launcher
        
        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        logger.info('\n' + dash_line + 
                    '\nRuntime environment:' + runtime_env_info + '\n' +
                    dash_line + '\n')
    # -------------------    Environment  End  ------------------------------ #
    
    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################
    
    start_load_data_t = time.time()
    
    chat_template = CHAT_TEMPLATE_MAP[args.chat_template]
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.llm,
        trust_remote_code=True,
        padding_side='right')
    
    img_token = chat_template.image_token
    tokenizer.add_tokens([img_token], special_tokens=True)
    
    img_token_id = tokenizer.convert_tokens_to_ids([img_token])[0]
    logger.info(f'[Tokenizer] Added a new token `{img_token}`, '
                f'token id is {img_token_id}, the new vocab size is '
                f'{len(tokenizer)}')
    
    img_processor = AutoProcessor.from_pretrained(args.vit).image_processor
    
    dset_infos = load(args.datasets)
    if args.dset_from_cache:
        if dist.get_rank() == 0:
            _datasets = []
            cache_dir = args.dset_cache_dir
            desc = f'[Rank {rank}] Load Cached Datasets'
            for sub_dir in tqdm(os.listdir(cache_dir), desc=desc):
                dset = load_from_disk(os.path.join(cache_dir,sub_dir))
                _datasets.append(dset)

            objects = [_datasets]
        else:
            objects = [None]

        dist.broadcast_object_list(objects, src=0)
        hf_datasets = objects[0]
    else:
        
        # Define how to convert raw data into OpenAI format.
        # If you want to use other data format, you need to define the 
        # corresponding `foramt_fn` and `tokenize fn`.

        # The following function is used to tokenize an original sample.
        # If your data format is different, you should redefine a `tokenize_fn`
        # The tokenized data must include `input_ids`, `labels``, 
        # and `num_tokens`.
        
        def count_img_tokens(url):
            return 576
        
        
        if args.dset_pack_level is None:
            if args.dset_cache_dir:
                logger.warning('')
            online_tokenize = True
        else:
            online_tokenize = False
        
        tokenize_fns = []
        sample_ratios = []
        annotations = []
        init_fns = []
        for _, info in dset_infos.items():
            if 'format' in info:
                dset_format = info['format']
            else:
                dset_format = 'llava'
            
            image_dir = info['image_dir']
            
            tokenize_fn = LlavaTokenizeFunction(
                tokenizer, chat_template, image_dir, dset_format)
            
            if online_tokenize:
                init_fn = partial(
                    LlavaRawDataset, 
                    image_processor=img_processor,
                    tokenize_fn=tokenize_fn)
                init_fns.append(init_fn)
                tokenize_fns.append(None)
            else:
                init_fns.append(None)
                tokenize_fns.append(tokenize_fn)
            
            sample_ratios.append(info['sample_ratio'])
            annotations.append(info['annotations'])
        
        
        _datasets = load_datasets(
            paths=annotations,
            file_types=args.dset_file_types,
            sources='local',
            sample_ratios=sample_ratios,
            num_proc=max(args.num_workers, 1),
            map_fns=None if online_tokenize else tokenize_fns, 
            init_fns=init_fns)
        
        logger.info(_datasets[0][0])
        num_datasets = len(_datasets)
        
        if args.dset_cache_dir and (not online_tokenize) and rank==0:
    
            if os.path.isdir(args.dset_cache_dir):
                if len(os.listdir(args.dset_cache_dir)):
                    logger.warning(f"`{args.dset_cache_dir}` is not an empty "
                                   "folder, which may lead to inaccurate "
                                   "cache results.")
        
            for i, dset in tqdm(enumerate(_datasets), desc='Caching'):
                digits = len(str(abs(num_datasets)))
                cache_id = f'cache-{i+1:0{digits}}-of-{num_datasets:0{digits}}'
                cache_dir = args.dset_cache_dir
                sub_cache_dir = os.path.join(cache_dir, cache_id)
                if os.path.exists(sub_cache_dir):
                    shutil.rmtree(sub_cache_dir )
                    logger.warning(f"Found {sub_cache_dir} exists. "
                                   "Clear it and re-cache.")
                dset.save_to_disk(sub_cache_dir)

    if online_tokenize:
        datasets = _datasets
    else:
        datasets = []
        for i, dset in enumerate(_datasets):
                
            if args.dset_pack_level and args.dset_pack_level == 'soft':
                _dset = SoftPackerForLlava(dset,img_processor, args.max_length)
            elif not args.dset_pack_level:
                _dset = LlavaTokenizedDataset(dset, img_processor)
            else:
                raise RuntimeError
            
            datasets.append(_dset)

    train_dataset = ConcatDataset(datasets)
    
    if not online_tokenize and rank == 0:
        num_tokens = [torch.tensor(dset['num_tokens']) for dset in _datasets]
        num_tokens = torch.cat(num_tokens, dim=0)
        logger.info(f'[Dataset] {sum(num_tokens)} tokens.')
        for i in range(4):
            length = args.max_length //(i+1)
            greater = (num_tokens > length).sum()
            logger.info(f'[Dataset] (> {length} tokens) {greater} samples')
        
    
    if args.dset_pack_level and rank == 0:
        ori_samples = sum([len(dset) for dset in _datasets])
        packed_samples = len(train_dataset)
        logger.info(f'[Dataset] (Original) {ori_samples} samples.')
        logger.info(f'[Dataset] (Packed) {packed_samples} samples.')

    
    pack_batch = is_flash_attn_2_available()
    logger.info('`flash_attn` is available. To accelerate training, the '
                'data in one batch is concatenated into a single one, '
                'avoiding the pad tokens; this operation does not affect the '
                'calculation results, as the attention is computed in '
                'segments according to the original data.')

    collator = LlavaCollator(pack_batch=pack_batch)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        sampler=ParallelSampler(
            train_dataset, dp_mesh, args.global_batch_size, shuffle=True),
        collate_fn=collator,
        persistent_workers=args.num_workers > 0)
    
    
    if rank == 0:
        logger.info(f'[Dataloader] {len(train_dataloader)} batches.')
        
        _first_batch = collator([train_dataset[i] for i in range(args.mirco_batch_size)])
        _decoded = tokenizer.batch_decode(_first_batch['input_ids'])
        logger.debug(f'[Dataloader] Training Batch:\n{_first_batch}')
        logger.debug(f"[Dataloader] Training Batch(Decoded):\n{_decoded}")
    dist.barrier()
    
    load_data_cost_time = time.time() - start_load_data_t
    logger.info(f'[Dataset & Dataloader] Cost {load_data_cost_time:.2f}s')
    # -------------------    Dataset & Dataloader  End  --------------------- #

    
    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################
    
    start_model_t = time.time()
    
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            dtype == torch.bfloat16
        else:
            raise RuntimeError("The device does not support `bf16`, "
                               "please set `dtype` to `fp16`.")
    elif args.dtype == 'auto':
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16  
        else:
            dtype = torch.float16
    else:
        raise RuntimeError("`dtype` only supports `fp16`，`bf16`, or `auto`, "
                           f"but found {args.dtype}.")
    
    
    _text_config = AutoConfig.from_pretrained(args.llm, trust_remote_code=True)
    if is_flash_attn_2_available():
        _text_config.attn_implementation = 'flash_attention_2'
    elif is_torch_sdpa_available():
        _text_config.attn_implementation = 'sdpa'
        
    _text_config.use_cache = False
    _vision_config = AutoConfig.from_pretrained(args.vit).vision_config
    llava_config = LlavaConfig(
        _vision_config, _text_config, image_token_index=img_token_id)
    text_config = _text_config
    vision_config = _vision_config
    
    with torch.device('meta'):
        
        meta_llava_cfg = copy.deepcopy(llava_config)
        meta_llava = LlavaForConditionalGeneration(meta_llava_cfg)
        # model parameters must be in fp32.
        # this ensures that all numerical values in the optimizer are in fp32.
        # FSDP will use low precision during forward.
        meta_llava.to(torch.float32)
        
        
        ori_emb_shape = meta_llava.get_input_embeddings().weight.shape
        meta_llava.resize_token_embeddings(len(tokenizer))
        new_emb_shape = meta_llava.get_input_embeddings().weight.shape
        logger.info('Pad the parameters of `embbedings` and `output` from '
                    f'shape {ori_emb_shape} to shape {new_emb_shape}')
        
        if args.freeze_llm:
            meta_llava.language_model.requires_grad_(False)
            meta_llava.language_model.to(dtype)
        if args.freeze_vit:
            meta_llava.vision_tower.requires_grad_(False)
            meta_llava.vision_tower.to(dtype)
        # meta_lora_llm = get_peft_model(
        #     meta_llava.language_model, 
        #     LoraConfig(
        #         r=64,lora_alpha=16,lora_dropout=0.1,bias='none',
        #         target_modules=['wqkv'],task_type='CAUSAL_LM'))
        # breakpoint()
        # set_require_grad_param_to_fp32(meta_lora_llm)
        # for name, param in meta_lora_llm.named_parameters(recurse=False):
        #     if param.requires_grad:
        #         module.register_parameter(name, Parameter(param.to(torch.float32)))
        # meta_llava.language_model = meta_lora_llm

        # meta_llava.multi_modal_projector.to(torch.float32)
        
        
        if pack_batch or args.dset_pack_level:
            dispatch_modules(meta_llava)
        
    # Only load parameters on rank 0 to avoid each rank repeatedly loading the 
    # same model into the CPU, wasting memory
    if rank == 0 :
        with torch.device('cpu'):
            llava = LlavaForConditionalGeneration(llava_config).to(dtype)
            
            # llava has not loaded the pre-trained parameters of llm and vit
            del llava.language_model
            del llava.vision_tower
            llm = AutoModelForCausalLM.from_pretrained(args.llm, config=text_config)
            vit = CLIPVisionModel.from_pretrained(args.vit, config=vision_config).to(dtype)
            for param in llava.multi_modal_projector.parameters():
                param.data = torch.ones_like(param.data) / param.numel()
            
            # lora_llm = get_peft_model(
            #     llm, 
            #     LoraConfig(
            #         r=64,lora_alpha=16,lora_dropout=0.1,bias='none',
            #         target_modules=['wqkv'],task_type='CAUSAL_LM'))
            llava.language_model = llm
            llava.vision_tower = vit
            # llava.multi_modal_projector.to(torch.float32)
            llava.resize_token_embeddings(len(tokenizer))
        meta_llava_map = map_meta_modules(llava, meta_llava) 
    else:
        meta_llava_map = None
    
    dist.barrier()
        
    param_init_fn = partial(dp_lazy_init, module_map=meta_llava_map, dp_mesh=dp_mesh)
    
    policies = [
        all_required_grad_wrap_policy, 
        partial(token_embedding_wrap_policy, vocab_size=meta_llava.vocab_size),
        layer_auto_wrap_policy
    ]
    
    torch.cuda.reset_peak_memory_stats()
    shard_llava = FSDP(
        meta_llava,
        device_mesh=dp_mesh,
        auto_wrap_policy=partial(_or_policy, policies=policies),
        sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        param_init_fn=param_init_fn,
        sync_module_states=True,
    )
    
    max_memory = torch.cuda.max_memory_allocated()
    logger.info('The peak GPU memory when building the FSDP model is '
                f'{max_memory/1024**3:.1f}GB.')
    
    if args.selective_recompute:
        check_fn = partial(
            checkpoint_check_fn, 
            target= RECOMPUTE_MODULES, 
            selective=args.selective_recompute)
        apply_activation_checkpointing(shard_llava, check_fn=check_fn)
        
    fsdp_cost_time = time.time() - start_model_t
    logger.info(f'[Model] Cost {fsdp_cost_time:.2f}s')
    # --------------------------    FSDP  End  ------------------------------ #
    
    
    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################
    optimizer = AdamW(
        shard_llava.multi_modal_projector.parameters(), lr=args.lr, weight_decay=args.wd)

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    # `iter` means once forward+backward
    # `step` means once optimizer step
    # `per_step_iters` means gradient accumulative counts
    per_step_iters = global_batch_size // mirco_batch_size // dp_size
    per_epoch_iters = len(train_dataloader)
    per_epoch_steps = math.ceil(per_epoch_iters / per_step_iters)

    total_epochs = args.epochs
    total_steps = per_epoch_steps * total_epochs
    
    if args.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * args.checkpoint_interval)
    else:
        checkpoint_interval = int(args.checkpoint_interval)

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=0)
    
    start_step = 0
    
    # ----------------    Optimizer & Scheduler End   ----------------------- #
    
    

    ############################## 5. Training ###############################
    
    start_train_t = time.time()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024**3):.1f}GB')
    for step in range(start_step, total_steps):

        epoch = step // per_epoch_iters
        epoch_inner_step = step % per_epoch_steps
        if epoch_inner_step == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.
            # train_dataloader.sampler.set_epoch(epoch, inner_step)
            train_dataloader.sampler.set_epoch(epoch, epoch_inner_step)
            data_iterator = iter(train_dataloader)

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_lr()[0]

        torch.cuda.reset_peak_memory_stats()

        step_loss = 0
        step_data_time = 0
        step_start_t = time.time()
        step_consumed_tokens = 0
        step_consumed_img_tokens = 0
        for i in range(per_step_iters):

            _data_start_t = time.time()
            data = next(data_iterator)
            step_data_time += time.time() - _data_start_t
            
            input_ids = data['input_ids'].cuda()
            pixel_values = data['pixel_values']
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.cuda()
            labels = data['labels'].cuda()
            attention_mask = data['attention_mask'].cuda()
            num_tokens = data['num_tokens'].cuda()
            num_img_tokens = data['num_img_tokens'].cuda()
            # image_ranges = data['image_ranges']
            
            enable_pack_ctx = pack_batch or args.dset_pack_level
            with packed_sequence(num_tokens, enable=enable_pack_ctx):
                outputs = shard_llava(
                    input_ids = input_ids, labels = labels,
                    pixel_values = pixel_values, attention_mask=attention_mask
                )
                avg_iter_loss = outputs.loss / per_step_iters
                avg_iter_loss.backward()
            
            # breakpoint()
            step_loss += avg_iter_loss.item()
            step_consumed_tokens += num_tokens.sum()
            step_consumed_img_tokens += num_img_tokens.sum()

        grad_norm = shard_llava.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        # breakpoint()
        step_text_tokens = step_consumed_tokens - step_consumed_img_tokens
        step_img_tokens = step_consumed_img_tokens
        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time)
        max_memory = torch.cuda.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            logger.info(f'[Train] (Epoch {epoch}) Step {step+1}/{total_steps}  '
                        f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                        f'grad_norm: {grad_norm:.2f}  '
                        f'max_memory: {(max_memory / 1024**3):.1f}GB  '
                        f'text_tokens: {step_text_tokens}  '
                        f'image_tokens: {step_img_tokens}  '
                        f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                        f'time: {step_time:.2f}s  '
                        f'eta: {eta}')

        if is_interval(step, total_steps, checkpoint_interval):
            # FSDP cannot be saved via torch.load
            # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
            model_state_dict, optimizer_state_dict = get_state_dict(
                shard_llava, optimizer)

            state_dict = {
                'model': model_state_dict,
                'optimizer': optimizer_state_dict,
                'step': step,
                'total_steps': total_steps,
                'warmup_scheduler': warmup_scheduler.state_dict(),
                'cosine_scheduler': cosine_scheduler.state_dict()
            }

            num_digits = len(str(abs(total_steps)))
            work_dir = args.work_dir
            ckpt_dir = os.path.join(work_dir, f'ckpt-{step:0{num_digits}}')
            writer = dcp.FileSystemWriter(ckpt_dir)
            mkdir_or_exist(ckpt_dir)
            dcp.save(state_dict, writer)
        
    train_cost_time = time.time() - start_train_t
    logger.info(f'[Train] Cost {train_cost_time}s')
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':

    args = parse_args()
    llava(args)
