import os
import socket
from typing import Optional, Union

import ray
import torch
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from loguru import logger
from ray.util.placement_group import placement_group as create_placement_group
from ray.util.placement_group import remove_placement_group
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers import get_scheduler as transformers_get_scheduler
from transformers.dynamic_module_utils import init_hf_modules
from transformers.generation.utils import GenerateDecoderOnlyOutput

from ..config.config_consts import (ENGINE_PLUGIN_DDP, ENGINE_PLUGIN_DEEPSPEED,
                                    ENGINE_PLUGIN_FSDP)
from ..config.config_utils import get_dp_size, get_gpu_requirement
from ..policy_output import (PolicyOutput, concat_policy_outputs,
                             logprobs_from_logits)
from ..tokenizer import get_tokenizer
from ..utils import set_seed
from .dist_utils import init_process_group
from .generate_utils import (get_answer_str, get_question_answer_mask,
                             partition_by_micro_batch_size,
                             partition_list_by_micro_batch_size)
from .ray_actor_group import RayActorGroup
from .ray_actor_mixin import RayActorMixin
from .ray_utils import DEFAULT_NUM_CPUS, DEFAULT_NUM_GPUS, create_ray_actors

DEFAULT_NEW_TOKENS = 64
MAXIMUM_NEW_TOKENS = 1024
"""
HfModelRunner can be individually called by other process
HfModelRunnerRayActor is called by ModelServer with .remote()
"""


class HfModelRunner:
    """HfModelRunner is capable of training, inference, and generation."""

    def __init__(self, model_config):
        self.model_config: dict = model_config

    def initialize(self):
        # 0. Environment
        envs = self.model_config.get('envs', {})
        for key, value in envs.items():
            os.environ[key] = value

        # Parallel Settings
        parallel: dict = self.model_config['parallel']
        assert parallel['tensor']['size'] == 1  # TODO: support TP
        assert parallel['pipeline']['size'] == 1  # TODO: support PP
        self.step = 0
        self.zero_stage = 1
        mixed_precision = self.model_config.get('mixed_precision', None)
        if parallel['data'].get('mode') == ENGINE_PLUGIN_FSDP:
            self.accelerator = Accelerator(
                fsdp_plugin=FullyShardedDataParallelPlugin())
            self.zero_stage = 3
        elif parallel['data'].get('mode') == ENGINE_PLUGIN_DEEPSPEED:
            from accelerate import DeepSpeedPlugin

            ds_config = self.model_config['deepspeed_config']  # requisite
            self.accelerator = Accelerator(
                deepspeed_plugin=DeepSpeedPlugin(ds_config))
            self.zero_stage = ds_config['zero_optimization']['stage']
        else:
            self.accelerator = Accelerator(mixed_precision=mixed_precision)
            self.zero_stage = 0

        # 1. Model
        model_path = self.model_config.get('model_path')
        self.model_type = self.model_config.get('model_type', '').lower()
        torch_dtype = self.model_config.get('torch_dtype', 'auto')
        use_flash_attn = self.model_config.get('use_flash_attn', None)
        model_class = self.model_config.get('model_class',
                                            AutoModelForCausalLM)
        self.model: PreTrainedModel = model_class.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map=None if self.zero_stage == 3 else 'auto',
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation='flash_attention_2'
            if use_flash_attn else None,
        )

        # Graident checkpointing
        gradient_checkpointing = self.model_config.get(
            'gradient_checkpointing', False)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.vocab_size = self.model.config.vocab_size

        # 2. Tokenizer
        tokenizer_path = self.model_config.get('tokenizer_path', model_path)
        tokenizer_config = self.model_config.get('tokenizer_config', {})
        self.tokenizer = get_tokenizer(
            tokenizer_path, trust_remote_code=True, **tokenizer_config)

        # 3. Trainer
        train_kwargs = self.model_config.get('train_kwargs')
        if train_kwargs is None:  # requires no training
            self.model = self.accelerator.prepare(
                self.model) if self.zero_stage == 3 else self.model
            self.device = self.accelerator.device
            logger.info(
                f'[{self.model_type}] __init__() done without train_kwargs.')
            return
        optimizer_type = train_kwargs.get('optimizer', torch.optim.AdamW)
        learning_rate = train_kwargs.get('lr', 1e-5)
        self.clip_grad_norm = train_kwargs.get('clip_grad_norm', 1.0)
        self.optimizer: torch.optim.Optimizer = optimizer_type(
            params=self.model.parameters(),
            lr=learning_rate,
        )

        lr_scheduler_type = train_kwargs.get('lr_scheduler', 'linear')
        lr_scheduler_kwargs = train_kwargs.get(
            'lr_scheduler_kwargs',
            {
                'num_warmup_steps': 0,
                'num_training_steps': 10000000000
            },
        )
        self.lr_scheduler: _LRScheduler = transformers_get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            **lr_scheduler_kwargs,
        )
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(  # noqa: E501
            self.model, self.optimizer, self.lr_scheduler)

        # Others
        self.device = self.accelerator.device
        set_seed(self.model_config.get('seed'))
        if mixed_precision is not None:
            self.info_rank0(
                f'[{self.model_type}]: Enable mixed_precision = {mixed_precision}'  # noqa: E501
            )
        if gradient_checkpointing:
            self.info_rank0(
                f'[{self.model_type}]: Enable gradient_checkpointing')
        self.info_rank0(
            f'[{self.model_type}] __init__() done with optimizer {self.optimizer.optimizer}.'  # noqa: E501
        )

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        criterion: Optional[_Loss] = None,
        loss_weight: Optional[float] = None,
        **_ignored,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        labels = input_ids.clone() if labels is None else labels
        if attention_mask is not None:
            if position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
        batch = {
            'input_ids':
            input_ids,
            'attention_mask':
            attention_mask.to(self.device)
            if attention_mask is not None else None,
            'position_ids':
            position_ids.to(self.device) if position_ids is not None else None
        }
        self.model.train()

        if criterion is None:
            # OPT. A) Default settings
            assert isinstance(
                labels, torch.Tensor
            ), 'Please pass in `criterion` for non-tensor labels'
            batch['labels'] = labels.to(self.device)
            fwd_output = self.model(**batch, use_cache=False)
            loss = fwd_output.loss
        elif isinstance(labels, torch.Tensor):
            # OPT. B) Use preset loss functions, e.g., torch.nn.CrossEntropyLoss()  # noqa: E501
            # Adopted from: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L1199  # noqa: E501
            logits: torch.Tensor = self.model(**batch, use_cache=False).logits
            labels = labels.to(self.device)
            loss = criterion(logits, labels)
        elif isinstance(labels, dict):
            # OPT. C) Use customized loss function, see loss/policy_loss.py
            logits: torch.Tensor = self.model(
                **batch, use_cache=False, return_dict=True).logits
            for k, v in labels.items():
                labels[k] = v.to(self.device)
            loss = criterion(logits, labels)
        else:
            raise ValueError(f'labels of unsupported type: {type(labels)}')

        if loss_weight is not None:
            loss *= loss_weight
        return loss

    def parameter_update(self, step_interval=1):
        self.info_rank0(f'[{self.model_type}] self.parameter_update()')
        self.step += 1
        if self.step % step_interval == 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(),
                                             self.clip_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

    def train(
        self,
        input_ids: Union[list[torch.Tensor], torch.Tensor],
        labels: Optional[Union[list[torch.Tensor], torch.Tensor,
                               dict[str, torch.Tensor]]] = None,
        attention_mask: Optional[Union[list[torch.Tensor],
                                       torch.Tensor]] = None,
        position_ids: Optional[Union[list[torch.Tensor], torch.Tensor]] = None,
        criterion: Optional[Union[list[_Loss], _Loss]] = None,
        loss_weights: Optional[Union[list[float], float]] = None,
        step_interval: int = 1,
        # None means using the entire input as one batch
        micro_batch_size: Optional[Union[list[int], int]] = None,
        debug=False,
        **_ignored,
    ):
        if isinstance(input_ids, torch.Tensor):
            input_ids = [input_ids]
            labels = [labels]
            attention_mask = [attention_mask]
            position_ids = [position_ids]
            criterion = [criterion]
            loss_weights = [loss_weights]
            micro_batch_size = [micro_batch_size]
        else:
            if attention_mask is None:
                attention_mask = [None for _ in range(len(input_ids))]
            if position_ids is None:
                position_ids = [None for _ in range(len(input_ids))]
            if criterion is None:
                criterion = [None for _ in range(len(input_ids))]
            if loss_weights is None:
                loss_weights = [None for _ in range(len(input_ids))]
            if micro_batch_size is None:
                micro_batch_size = [None for _ in range(len(input_ids))]

        assert isinstance(input_ids, list)

        loss_list = [[] for _ in range(len(input_ids))]
        for index in range(len(input_ids)):
            mb_size_entry = micro_batch_size[index]
            if mb_size_entry is None:
                micro_batches: list[dict[str, torch.Tensor]] = []
                micro_batches.append({
                    'input_ids': input_ids[index],
                    'attention_mask': attention_mask[index],
                    'position_ids': position_ids[index],
                    'labels': labels[index]
                })
            else:
                micro_batches = partition_by_micro_batch_size(
                    input_ids=input_ids[index],
                    micro_batch_size=micro_batch_size[index],
                    attention_mask=attention_mask[index],
                    position_ids=position_ids[index],
                    labels=labels[index],
                )
            loss_entry = []
            for mb_index, micro_batch in enumerate(micro_batches):
                if mb_index == 0:
                    self.info_rank0(
                        f"[{self.model_type}] will train input_ids[{mb_index}] shape[{micro_batch['input_ids'].shape}] * {len(micro_batches)} times"  # noqa: E501
                    )
                # compute loss and backward
                loss = self.compute_loss(
                    input_ids=micro_batch['input_ids'],
                    labels=micro_batch['labels'],
                    attention_mask=micro_batch['attention_mask'],
                    position_ids=micro_batch['position_ids'],
                    criterion=criterion[index],
                    loss_weight=loss_weights[index],
                )
                self.accelerator.backward(loss)
                loss_entry.append(loss)
                if debug:
                    set_seed(1234)
            loss_list[index] = sum(loss_entry) / len(loss_entry)

        self.parameter_update(step_interval)
        return loss_list if len(loss_list) > 1 else loss_list[0]

    # Inference
    @torch.no_grad()
    def _infer(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        output_logprobs=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        infer_kwargs: Optional[dict] = {},
        **_ignored,
    ) -> PolicyOutput:
        assert isinstance(input_ids, torch.Tensor)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        model_output = self.model(
            input_ids.to(self.device),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids.to(self.device),
            return_dict=True,
            **infer_kwargs,
        )

        output = PolicyOutput()
        if output_logits:
            output['logits'] = model_output['logits']
        if output_attentions:
            output['attentions'] = model_output['attentions']
        if output_hidden_states:
            output['hidden_states'] = model_output['hidden_states']
        if output_logprobs:
            log_probs = logprobs_from_logits(
                logits=model_output['logits'][:, :-1, :],
                labels=input_ids[:, 1:],
                gather=True,
            )
            output['logprobs'] = log_probs
        output.to('cpu')
        return output

    @torch.no_grad()
    def infer(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: Optional[
            int] = -1,  # -1: use the entire input as one batch
        tokenizer=None,  # Only used for reward models
        attention_mask=None,
        output_logprobs=False,
        output_logits=True,
        output_attentions=False,
        output_hidden_states=False,
        infer_kwargs: Optional[dict] = {},
        debug=False,
        **_ignored,
    ) -> PolicyOutput:
        self.info_rank0(
            f'[{self.model_type}] self.infer() kwargs: {infer_kwargs}')
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        # returns entire-input-as-one-batch inference results
        if micro_batch_size < 0:
            self.info_rank0(
                f'[{self.model_type}] infer() input_ids.shape: {input_ids.shape}'  # noqa: E501
            )
            return self._infer(
                input_ids,
                attention_mask,
                output_logprobs,
                output_logits,
                output_attentions,
                output_hidden_states,
                infer_kwargs,
            )

        # Otherwise, partition the input into micro batches and run inference on each micro batch separately  # noqa: E501
        micro_batches = partition_by_micro_batch_size(input_ids,
                                                      micro_batch_size,
                                                      attention_mask)
        policy_outputs = []
        for index, micro_batch in enumerate(micro_batches):
            input_ids_mb = micro_batch['input_ids']
            attention_mask_mb = micro_batch['attention_mask']
            if index == 0:
                self.info_rank0(
                    f'[{self.model_type}] will infer() input_ids_mb.shape: {input_ids_mb.shape} * {len(micro_batches)} times'  # noqa: E501
                )
            policy_output_mb = self._infer(
                input_ids_mb,
                attention_mask_mb,
                output_logprobs,
                output_logits,
                output_attentions,
                output_hidden_states,
                infer_kwargs,
            )
            policy_outputs.append(policy_output_mb)
            if debug:
                self.set_seed(1234)
        # Concatenate the policy outputs from each micro batch and return the result  # noqa: E501
        return concat_policy_outputs(policy_outputs)

    # Generate
    @torch.no_grad()
    def _generate(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        step=-1,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        generate_kwargs: Optional[dict] = {},
    ) -> PolicyOutput:
        assert isinstance(input_ids, torch.Tensor)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model

        max_new_tokens = (
            MAXIMUM_NEW_TOKENS
            if 'eos_token_id' in generate_kwargs else DEFAULT_NEW_TOKENS)
        max_new_tokens = step if step > 0 else max_new_tokens

        # TODO: stop if meeting eos_token_id
        model_output: GenerateDecoderOnlyOutput = model.generate(
            input_ids.to(model.device),
            use_cache=True,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_logits=output_logits,  # transformers >= 4.38.2
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        output_ids = model_output['sequences']
        self.info_rank0(
            f'generate input_ids shape:[{input_ids.shape}], output_ids shape:[{output_ids.shape}]'  # noqa: E501
        )
        output = PolicyOutput(output_ids=output_ids)
        # masks
        output['question_mask'], output[
            'answer_mask'] = get_question_answer_mask(
                input_ids,
                output_ids,
                tokenizer_pad_token_id=self.tokenizer.pad_token_id,
                generate_pad_token_id=generate_kwargs.get('pad_token_id'),
            )
        output['attention_mask'] = output.question_mask + output.answer_mask
        output['action_mask'] = output['attention_mask'][:,
                                                         input_ids.size(1) -
                                                         1:-1]

        if output_logits:
            output['logits'] = model_output['logits']  # tuple(torch.Tensor, )
        if output_attentions:
            output['attentions'] = model_output['attentions']
        if output_hidden_states:
            output['hidden_states'] = model_output['hidden_states']
        if output_str:  # customized post processing
            output['output_str'] = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output['output_ans_str'] = get_answer_str(
                tokenizer=self.tokenizer,
                output_ids=output_ids,
                answer_mask=output.answer_mask,
            )

        output.to('cpu')
        return output

    # Generate
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: Optional[
            int] = -1,  # -1: use the entire input as one batch
        attention_mask=None,
        step=-1,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        chat_template=None,
        generate_kwargs: Optional[dict] = {},
        debug=False,
        **_ignored,
    ) -> PolicyOutput:
        self.info_rank0(
            f'[{self.model_type}] self.generate() kwargs: {generate_kwargs}')
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            assert isinstance(attention_mask, torch.Tensor)
            attention_mask = attention_mask.to(self.device)

        if micro_batch_size < 0:
            return self._generate(
                input_ids,
                attention_mask,
                step,
                output_str,
                output_logits,
                output_attentions,
                output_hidden_states,
                generate_kwargs,
            )

        micro_batches = partition_by_micro_batch_size(input_ids,
                                                      micro_batch_size,
                                                      attention_mask)
        policy_outputs = []
        for micro_batch in micro_batches:
            input_ids_mb = micro_batch['input_ids']
            attention_mask_mb = micro_batch['attention_mask']
            policy_output_mb = self._generate(
                input_ids_mb,
                attention_mask_mb,
                step,
                output_str,
                output_logits,
                output_attentions,
                output_hidden_states,
                generate_kwargs,
            )
            policy_outputs.append(policy_output_mb)
            if debug:
                self.set_seed(1234)

        padding_token_map = {'output_ids': self.tokenizer.pad_token_id}
        return concat_policy_outputs(policy_outputs, padding_token_map)

    def get_model(self):
        parallel: dict = self.model_config['parallel']
        dp = parallel['data'].get('size')
        dp_mode = parallel['data'].get('mode')
        if dp > 1 and dp_mode != ENGINE_PLUGIN_DDP:
            raise ('please use get_state_dict instead when using parallel')
        _model = self.accelerator.unwrap_model(self.model)
        return _model

    def get_state_dict(self):
        state_dict = self.accelerator.get_state_dict(self.model)
        if not self.accelerator.is_main_process:
            return None
        return state_dict

    def set_seed(self, seed=None):
        set_seed(seed)

    def save_model(self, path):
        if not self.accelerator.is_main_process:
            self.accelerator.get_state_dict(self.model)
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if not os.path.exists(path):
            os.makedirs(path)
        unwrapped_model.save_pretrained(
            path,
            is_main_process=True,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model),
        )
        logger.info(f'save model to {path}')

    def info_rank0(self, content):
        if self.accelerator.is_main_process:
            logger.info(content)


# Adapted from https://github.com/OpenLLMAI/OpenRLHF/blob/v0.2.5/openrlhf/trainer/ray/ppo_actor.py  # noqa: E501
class HfModelRunnerRayActor(HfModelRunner, RayActorMixin):
    """A ray.remote Actor Class initialized by HfModelRunnerRayActorGroup,
    extending HfModelRunner with ray related method via RayActorMixin."""

    def init_process_group(self, generator):
        if self.accelerator.is_main_process:
            # init process groups for vllm engine
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(('', 0))
                master_port = sock.getsockname()[1]

            world_size = generator.dp_size * generator.tp_size + 1
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * generator.tp_size + 1,
                    world_size,
                    'vllm',
                ) for i, engine in enumerate(generator.ray_actors)
            ]
            self._model_update_group = init_process_group(
                backend='nccl',
                init_method=f'tcp://{master_address}:{master_port}',
                world_size=world_size,
                rank=0,
                group_name='vllm',
            )
            ray.get(refs)

    def broadcast_model_to_generator(self, generator):
        # TODO: Support Pytorch FSDP.
        if self.model_config['parallel']['data'].get(
                'mode') == ENGINE_PLUGIN_FSDP:
            raise NotImplementedError('FSDP is not supported yet.')
        logger.info('Broadcast BEGIN')
        model = self.accelerator.unwrap_model(self.model)
        for name, param in model.named_parameters():
            if self.accelerator.is_main_process:
                shape = param.shape if self.zero_stage != 3 else param.ds_shape

                for engine in generator.ray_actors:
                    engine.update_weight.remote(
                        name, dtype=param.dtype, shape=shape)

            if self.zero_stage != 3:
                if self.accelerator.is_main_process:
                    torch.distributed.broadcast(
                        param.data, 0, group=self._model_update_group)
            else:
                from deepspeed.runtime.zero.partition_parameters import \
                    GatheredParameters

                with GatheredParameters([param]):
                    if self.accelerator.is_main_process:
                        torch.distributed.broadcast(
                            param.data, 0, group=self._model_update_group)

        logger.info('Broadcast END')


class HfModelRunnerRayActorGroup(RayActorGroup):
    """HfModelRunnerRayActorGroup manages a list of HfModelRunnerRayActor
    create ray actors."""

    # avoid ModuleNotFoundError: No module named 'transformers_modules'
    # refer to https://github.com/vllm-project/vllm/pull/871
    init_hf_modules()

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.released = True
        num_gpus = get_gpu_requirement(config)
        self.dp_size = get_dp_size(config)
        self.tokenizer_pad_token_id = config.tokenizer_config['pad_token_id']
        bundles = [{
            'CPU': DEFAULT_NUM_CPUS,
            'GPU': DEFAULT_NUM_GPUS
        } for _ in range(num_gpus)]
        self.placement_group = create_placement_group(bundles)
        self.ray_actors: list[HfModelRunnerRayActor] = create_ray_actors(
            name_prefix=name,
            config=config,
            placement_group=self.placement_group,
            trainer_class=ray.remote(
                num_cpus=DEFAULT_NUM_CPUS,
                num_gpus=DEFAULT_NUM_GPUS)(HfModelRunnerRayActor),
        )
        self.released = False

        master_ip = ray.get(self.ray_actors[0].get_metadata.remote()).node_ip
        master_port = ray.get(self.ray_actors[0].get_free_port.remote())
        ray.get([
            actor.inject_distribute_env.remote(
                master_ip=master_ip,
                master_port=master_port,
                rank_id=rank,
                world_size=len(self.ray_actors),
            ) for rank, actor in enumerate(self.ray_actors)
        ])
        self.initialize_ref = [
            actor.initialize.remote() for actor in self.ray_actors
        ]

    def initialize_get(self):
        if self.initialize_ref is not None:
            ray.get(self.initialize_ref)
        else:
            logger.info(
                'self.initialize_get None, maybe self.generator==self.trainer')
        self.initialize_ref = None

    # Training
    def train_async(self, input_ids, labels, attention_mask, position_ids,
                    *args, **kwargs):
        if isinstance(input_ids, torch.Tensor):
            micro_batch_size = input_ids.shape[0] // self.dp_size + (
                input_ids.shape[0] % self.dp_size > 0
            )  # round up division, i.e., math.ceil(a / b)
            micro_batches = partition_by_micro_batch_size(
                input_ids, micro_batch_size, attention_mask, position_ids,
                labels)
            assert len(micro_batches) == self.dp_size
            return [
                self.ray_actors[index].train.remote(
                    input_ids=micro_batch['input_ids'],
                    attention_mask=micro_batch['attention_mask'],
                    position_ids=micro_batch['position_ids'],
                    labels=micro_batch['labels'],
                    *args,
                    **kwargs,
                ) for index, micro_batch in enumerate(micro_batches)
            ]
        elif isinstance(input_ids, list):
            """a list of tensors whose training loss will be taken average."""
            assert isinstance(input_ids[0], torch.Tensor)
            micro_batch_size = [i for i in range(len(input_ids))]
            for index, input_id in enumerate(input_ids):
                micro_batch_size[index] = input_id.shape[0] // self.dp_size + (
                    input_id.shape[0] % self.dp_size > 0
                )  # round up division, i.e., math.ceil(a / b)
            micro_batches = partition_list_by_micro_batch_size(
                input_ids=input_ids,
                micro_batch_size=micro_batch_size,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            assert len(micro_batches) == self.dp_size
            object_refs = []
            for index, micro_batch in enumerate(micro_batches):
                input_ids_mb = []
                attention_mask_mb = []
                position_ids_mb = []
                labels_mb = []
                for i in range(len(micro_batch)):
                    input_ids_mb.append(micro_batch[i]['input_ids'])
                    attention_mask_mb.append(micro_batch[i]['attention_mask'])
                    position_ids_mb.append(micro_batch[i]['position_ids'])
                    labels_mb.append(micro_batch[i]['labels'])
                object_ref = self.ray_actors[index].train.remote(
                    input_ids=input_ids_mb,
                    attention_mask=attention_mask_mb,
                    position_ids=position_ids_mb,
                    labels=labels_mb,
                    *args,
                    **kwargs,
                )
                object_refs.append(object_ref)
            return object_refs

    def train_get(self, object_refs, timeout=None):
        losses = ray.get(object_refs, timeout=timeout)
        if isinstance(losses[0], list):
            p_loss = [sub_loss[0] for sub_loss in losses]
            pt_loss = [sub_loss[1] for sub_loss in losses]
            return [sum(p_loss) / len(p_loss), sum(pt_loss) / len(pt_loss)]
        else:
            return sum(losses) / len(losses)

    def train(self, *args, **kwargs):
        object_refs = self.train_async(*args, **kwargs)
        return self.train_get(object_refs)

    # Inference
    def infer_async(self, input_ids, attention_mask, *args, **kwargs):
        micro_batch_size = input_ids.shape[0] // self.dp_size + (
            input_ids.shape[0] % self.dp_size > 0
        )  # round up division, i.e., math.ceil(a / b)
        micro_batches = partition_by_micro_batch_size(input_ids,
                                                      micro_batch_size,
                                                      attention_mask)
        assert len(micro_batches) == self.dp_size
        return [
            self.ray_actors[index].infer.remote(
                input_ids=micro_batch['input_ids'],
                attention_mask=micro_batch['attention_mask'],
                *args,
                **kwargs,
            ) for index, micro_batch in enumerate(micro_batches)
        ]

    def infer_get(self, object_refs, timeout=None):
        outputs = ray.get(object_refs, timeout=timeout)
        return concat_policy_outputs(outputs)

    def infer(self, *args, **kwargs):
        object_refs = self.infer_async(*args, **kwargs)
        return self.infer_get(object_refs)

    # Generation
    def generate_async(self, input_ids, attention_mask, *args, **kwargs):
        micro_batch_size = input_ids.shape[0] // self.dp_size + (
            input_ids.shape[0] % self.dp_size > 0
        )  # round up division, i.e., math.ceil(a / b)
        micro_batches = partition_by_micro_batch_size(input_ids,
                                                      micro_batch_size,
                                                      attention_mask)
        assert len(micro_batches) == self.dp_size
        return [
            self.ray_actors[index].generate.remote(
                input_ids=micro_batch['input_ids'],
                attention_mask=micro_batch['attention_mask'],
                *args,
                **kwargs,
            ) for index, micro_batch in enumerate(micro_batches)
        ]

    def generate_get(self, object_refs, timeout=None):
        outputs = ray.get(object_refs, timeout=timeout)
        padding_token_map = {
            'output_ids': self.config.tokenizer_config.pad_token_id
        }
        return concat_policy_outputs(outputs, padding_token_map)

    def generate(self, *args, **kwargs):
        object_refs = self.generate_async(*args, **kwargs)
        return self.generate_get(object_refs)

    # Others
    def get_model(self):
        return self.ray_actors[0].get_model.remote()

    def get_state_dict(self):
        state_dicts = [
            actor.get_state_dict.remote() for actor in self.ray_actors
        ]
        return state_dicts[0]

    def set_seed(self, seed=None):
        ray.get([actor.set_seed.remote(seed) for actor in self.ray_actors])

    def release_resources(self):
        """release ray resources."""
        if self.released:
            return
        for actor in self.ray_actors:
            try:
                ray.kill(actor=actor, no_restart=True)
            except BaseException as exp:
                logger.error(f'failed to kill ray actor {actor}. {exp}')
        remove_placement_group(self.placement_group)
        self.released = True

    def save_model(self, path):
        ray.get([actor.save_model.remote(path) for actor in self.ray_actors])

    def init_process_group(self, generator):
        refs = [
            hfm.init_process_group.remote(generator)
            for i, hfm in enumerate(self.ray_actors)
        ]
        ray.get(refs)

    def broadcast_model_to_generator(self, generator: None):
        refs = [
            hfm.broadcast_model_to_generator.remote(generator)
            for i, hfm in enumerate(self.ray_actors)
        ]
        ray.get(refs)
