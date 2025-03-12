import copy
import os
import types
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF
import torch.nn as nn
from packaging import version
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor, Partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleOutput,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from xtuner._lite.accelerate import liger_kernel_is_available
from xtuner._lite.chat import HybridChatTemplate
from xtuner._lite.modelings.internvl_chat import InternVLChatConfig, InternVLChatModel
from xtuner._lite.parallel.sequence import split_for_sequence_parallel
from xtuner._lite.patches.base import (
    FSDPConfig,
    HFCheckpointLoader,
    ModelConfig,
    PatchedCausalLM,
    clip_grad_norm_,
    lazy_init_fn,
)
from xtuner._lite.patches.mixins import GenerateMixin
from xtuner._lite.patches.utils import pad_to_max_length, pad_to_multiple_of


def _replicate_other_params(module, device_mesh):
    if type(module).__name__ in ('ExpertEp', 'GroupedLinear'):
        return
    for name, param in module.named_parameters(recurse=False):
        dist_param = nn.Parameter(
            distribute_tensor(param, device_mesh, [Replicate()])
        )
        module.register_parameter(name, dist_param)
    for child in module.children():
        _replicate_other_params(child, device_mesh)


class CUDAPatchedInternVLChatModel(PatchedCausalLM):
    device_type = "cuda"

    # TODO: vision_model tensor parallel
    layer_tp_plan = {
        "norm1": PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
        ),
        # "attn": PrepareModuleInput(
        #     input_layouts=(Shard(1),),
        #     desired_input_layouts=(Replicate(),),
        # ),
        # "attn.qkv": ColwiseParallel(),
        "attn.inner_attn": PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        ),
        "attn.proj": PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
        ),
        # "attn.proj_drop": SequenceParallel(),
        "drop_path1": PrepareModuleOutput(
            output_layouts=(Replicate(),),
            desired_output_layouts=(Replicate(),),
        ),
        "norm2": PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
        ),
        # "mlp": PrepareModuleInput(
        #     input_layouts=(Shard(1),),
        #     desired_input_layouts=(Replicate(),),
        # ),
        # "mlp.fc1": ColwiseParallel(),
        # "mlp.fc2": RowwiseParallel(output_layouts=Shard(1)),
        "drop_path2": PrepareModuleOutput(
            output_layouts=(Replicate(),),
            desired_output_layouts=(Replicate(),),
        ),
    }

    # TODO: vision_model tensor parallel
    casual_tp_plan = {
        "vision_model.embeddings": PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
        ),
        # "vision_model.embeddings.patch_embeddings": None, # This is conv2d
        "vision_model.encoder": PrepareModuleInput(
            input_kwarg_layouts={"inputs_embeds": Shard(-1)},
            desired_input_kwarg_layouts={"inputs_embeds": Replicate()},
            use_local_output=True,
        ),
        # "vision_model.encoder": PrepareModuleInput(
        #     input_kwarg_layouts={"inputs_embeds": Replicate()},
        #     desired_input_kwarg_layouts={"inputs_embeds": Shard(1)},
        #     use_local_output=True,
        # ),
        # "vision_model": PrepareModuleOutput(
        #     output_layouts=(Shard(1),),
        #     desired_output_layouts=(Replicate(),),
        # ),
        # "mlp1": PrepareModuleInput(
        #     input_layouts=(Replicate(),),
        #     desired_input_layouts=(Replicate(),),
        # ),
        "mlp1.0": PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
        ),  # LayerNorm
        # "mlp1.1": ColwiseParallel(),  # Linear
        "mlp1.3": PrepareModuleOutput(
            output_layouts=(Replicate(),),
            desired_output_layouts=(Replicate(),),
        ),  # Linear
    }

    def __init__(self, model: InternVLChatModel, fsdp_config: Optional[FSDPConfig]):
        super().__init__(model, fsdp_config)

        if dist.is_initialized() and dist.is_available():
            rank = dist.get_rank()
        else:
            rank = 0

        self._rank0_model: Optional[InternVLChatModel] = None
        if rank == 0:
            self._rank0_model = copy.deepcopy(model)

        # This only dispatch InternVisionModel
        self._patched_model = self.dispatch_hf_code(model)
        # Dispatch language_model according to the architecture
        # TODO: How to avoid patched_lm_cls copy language_model once again?
        patched_lm_cls = self._get_patched_lm_cls(model)
        self._patched_lm = patched_lm_cls(model.language_model, fsdp_config=fsdp_config)
        self._patched_model.language_model = self._patched_lm.patched_model

        self.init_model_config(fsdp_config)

        self._fsdp_config = fsdp_config
        if self._fsdp_config is not None:
            self.init_device_mesh(fsdp_config)

    @property
    def patched_model(self) -> InternVLChatModel:
        return self._patched_model

    @property
    def rank0_model(self) -> Optional[InternVLChatModel]:
        return self._rank0_model

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    @property
    def fsdp_config(self) -> FSDPConfig:
        return self._fsdp_config

    @property
    def data_parallel_mesh(self) -> DeviceMesh:
        return self.dp_mesh

    @property
    def data_mesh(self) -> DeviceMesh:
        return self._data_mesh

    @property
    def sequence_parallel_mesh(self) -> DeviceMesh:
        return self.sp_mesh

    def trainable_parameters(self):
        _requried_grad_params = [
            param for param in self.patched_model.parameters() if param.requires_grad
        ]
        return _requried_grad_params

    def clip_grad_norm(self, max_norm):
        if self.tp_mesh.size() > 1:
            lm = self.patched_model.language_model
            dist.all_reduce(
                lm.lm_head.weight.grad.to_local(),
                group=self.tp_mesh.get_group(),
            )
            dist.all_reduce(
                lm.model.norm.weight.grad.to_local(),
                group=self.tp_mesh.get_group(),
            )
            lm.lm_head.weight.grad.div_(self.tp_mesh.size())
            lm.model.norm.weight.grad.div_(self.tp_mesh.size())

        grad_norm = clip_grad_norm_(self.trainable_parameters(), max_norm)
        return grad_norm

    def init_model_config(self, fsdp_config: FSDPConfig):
        self._patched_lm.init_model_config(fsdp_config)

        vision_config = self.patched_model.config.vision_config
        # TODO: vision_model tensor parallel
        # assert vision_config.num_attention_heads >= fsdp_config.tp_size
        # assert vision_config.num_attention_heads % fsdp_config.tp_size == 0

        self._model_config = ModelConfig(
            num_hidden_layers=vision_config.num_hidden_layers,
            num_attention_heads=vision_config.num_attention_heads,
            # num_key_value_heads=vision_config.num_attention_heads // fsdp_config.tp_size,
            num_key_value_heads=vision_config.num_attention_heads,
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
            vocab_size=-1,
            head_dim=vision_config.hidden_size // vision_config.num_attention_heads,
        )

    @classmethod
    def _get_patched_lm_cls(cls, model: InternVLChatModel):
        # patch language model according to the architecture
        config = cast(InternVLChatConfig, model.config)
        architecture = config.llm_config.architectures[0]
        if architecture == "LlamaForCausalLM":
            from .llama import CUDAPatchedLlamaForCausalLM as PatchedModel
        elif architecture == "Qwen2ForCausalLM":
            from .qwen2 import CUDAPatchedQwen2ForCausalLM as PatchedModel
        elif architecture == "InternLM2ForCausalLM":
            # TODO: support internlm2 model dispatch
            raise NotImplementedError(f"xtuner does not support {architecture} now")
        else:
            raise NotImplementedError(f"{architecture} is not implemented.")
        return PatchedModel

    @classmethod
    def dispatch_hf_code(cls, model: InternVLChatModel) -> InternVLChatModel:
        model.forward = types.MethodType(cls.patched_causal_forward, model)
        return model

    def init_device_mesh(self, parallel_config: FSDPConfig) -> None:
        self._patched_lm.init_device_mesh(parallel_config)
        self.world_mesh = self._patched_lm.world_mesh
        self.tp_mesh = self._patched_lm.tp_mesh
        self.fsdp_mesh = self._patched_lm.fsdp_mesh
        self.dp_mesh = self._patched_lm.dp_mesh
        self.sp_mesh = self._patched_lm.sp_mesh
        self._data_mesh = self._patched_lm._data_mesh

    def fully_shard(
        self,
        *,
        module2name: Optional[Dict[nn.Module, str]] = None,
        checkpoint_loader: Optional[HFCheckpointLoader] = None,
    ):
        if module2name is None:
            module2name = {mod: name for name, mod in self.patched_model.named_modules()}

        if checkpoint_loader is None:
            checkpoint_loader = HFCheckpointLoader(self.patched_model.config._name_or_path)

        # First shard language model, and derive parallelism from language model
        self._patched_lm.fully_shard(
            module2name=module2name,
            checkpoint_loader=checkpoint_loader,
        )

        # Then shard vision model
        param_init_fn = partial(
            lazy_init_fn,
            module2name={mod: name for name, mod in self.patched_model.named_modules()},
            checkpoint_loader=HFCheckpointLoader(
                self.patched_model.config._name_or_path
            ),
        )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype,
            reduce_dtype=self.fsdp_config.reduce_dtype,
        )

        vision_model = self.patched_model.vision_model
        compiled_layers: List[nn.Module] = []
        num_recompute_layers = int(
            self.model_config.num_hidden_layers * self.fsdp_config.recompute_ratio
        )
        for layer_idx, layer in enumerate(vision_model.encoder.layers):
            layer.apply(param_init_fn)

            if self.tp_mesh.size() > 1:
                # TODO: vision_model tensor parallel
                pass
                # NOTE: This is a workaround before tp works: Replicate all parameters
                # _replicate_other_params(layer, self.tp_mesh)
                # parallelize_module(
                #     module=layer,
                #     device_mesh=self.tp_mesh,
                #     parallelize_plan=self.layer_tp_plan,
                # )

            # NOTE: InternVLChatModel hardcode enable gradient checkpointing for
            # all layers, so we don't need to do it here
            # if layer_idx < num_recompute_layers:
            #     layer = checkpoint_wrapper(layer, preserve_rng_state=True)

            if self.fsdp_config.torch_compile:
                # TODO: vision encoder compilation. should ensure static batch_size
                # layer = torch.compile(layer, fullgraph=True)
                # compiled_layers.append(layer)
                pass

            vision_model.encoder.layers.register_module(str(layer_idx), layer)

            fully_shard(
                layer,
                # mesh=self.fsdp_mesh,
                # TODO: vision_model tensor parallel. Pure FSDP currently
                mesh=self.world_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=self.fsdp_config.reshard_after_forward,
                offload_policy=CPUOffloadPolicy()
                if self.fsdp_config.cpu_offload
                else None,
            )

        if version.parse(torch.__version__) >= version.parse("2.5.0"):
            for layer_cur, layer_next in zip(
                vision_model.encoder.layers[:-1], vision_model.encoder.layers[1:],
            ):
                layer_cur.set_modules_to_forward_prefetch([layer_next])

        vision_model.embeddings.apply(param_init_fn)
        self.patched_model.mlp1.apply(param_init_fn)

        if self.tp_mesh.size() > 1:
            # TODO: vision_model tensor parallel
            pass
            # # Channel-wise parallel for convolution/embedding
            # param = vision_model.embeddings.patch_embedding.weight
            # vision_model.embeddings.patch_embedding.register_parameter(
            #     "weight",
            #     nn.Parameter(distribute_tensor(param, self.tp_mesh, [Shard(0)])),
            # )  # (C_out, C_in, kernel_h, kernel_w)
            # param = vision_model.embeddings.patch_embedding.bias
            # vision_model.embeddings.patch_embedding.register_parameter(
            #     "bias",
            #     nn.Parameter(distribute_tensor(param, self.tp_mesh, [Shard(0)])),
            # )  # (C_out)
            # param = vision_model.embeddings.class_embedding
            # vision_model.embeddings.register_parameter(
            #     "class_embedding",
            #     nn.Parameter(distribute_tensor(param, self.tp_mesh, [Shard(-1)])),
            # )
            # param = vision_model.embeddings.position_embedding
            # vision_model.embeddings.register_parameter(
            #     "position_embedding",
            #     nn.Parameter(distribute_tensor(param, self.tp_mesh, [Shard(-1)])),
            # )
            # # Tensor-parallel for other parts
            # _replicate_other_params(self.patched_model.mlp1, self.tp_mesh)
            # parallelize_module(
            #     module=self.patched_model,
            #     device_mesh=self.tp_mesh,
            #     parallelize_plan=self.casual_tp_plan,
            # )

        fully_shard(
            self.patched_model,
            # mesh=self.fsdp_mesh,
            # TODO: vision_model tensor parallel. Pure FSDP currently
            mesh=self.world_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )


    @staticmethod
    def patched_causal_forward(
        module: InternVLChatModel,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tp_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        assert input_ids is not None
        assert image_flags is not None

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else module.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else module.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else module.config.use_return_dict
        )

        # TODO: sequence parallelism for InternVLChatModel, where
        # input_ids/pixel_values might have been splited
        image_flags = cast(torch.LongTensor, image_flags.squeeze(-1))
        input_embeds = module.language_model.get_input_embeddings()(input_ids)

        if tp_mesh is not None and tp_mesh.size() > 1:
            # If tp enabled, LM embedding will be colwise parallelized
            input_embeds = (
                DTensor.from_local(
                    input_embeds, device_mesh=tp_mesh, placements=(Shard(1),)
                )
                .redistribute(device_mesh=tp_mesh, placements=(Replicate(),))
                .to_local()
            )
        # in-place op on custom-function outputs will spoil autograd
        input_embeds = input_embeds.clone()

        vit_embeds = module.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(
        #         f"dynamic ViT batch size: {vit_batch_size}, "
        #         f"images per sample: {vit_batch_size / B}, "
        #         f"dynamic token length: {N}"
        #     )

        input_ids = cast(torch.LongTensor, input_ids.reshape(B * N))

        selected = input_ids == module.img_context_token_id
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
            -1, C
        )

        input_embeds = input_embeds.reshape(B, N, C)

        if tp_mesh is not None and tp_mesh.size() > 1:
            # If tp enabled, LM embedding will return a sharded DTensor
            input_embeds = (
                DTensor.from_local(
                    input_embeds,
                    device_mesh=tp_mesh,
                    placements=(Replicate(),),
                )
                .redistribute(device_mesh=tp_mesh, placements=(Shard(1),))
                .to_local()
            )

        outputs = module.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return (outputs.loss,) + output if outputs.loss is not None else output

        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        label_shifted: bool = False,
        gather_logprobs: bool = False,
        cu_seq_lens_q: Optional[torch.IntTensor] = None,
        cu_seq_lens_k: Optional[torch.IntTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        block_table: Optional[torch.LongTensor] = None,
        prefilling: bool = False,
        sequence_parallel_mesh: Optional[DeviceMesh] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        assert input_ids is not None
        assert cu_seq_lens_q is not None
        assert cu_seq_lens_k is not None
        assert max_length_q is not None
        assert max_length_k is not None
        # TODO: sequence parallelism for InternVLChatModel
        assert sequence_parallel_mesh is None or sequence_parallel_mesh.size() == 1, (
            "sequence parallel not supported for InternVLChatModel yet"
        )

        if gather_logprobs:
            assert labels is not None and label_shifted

        _input_ids = input_ids
        _labels = labels
        _position_ids = position_ids
        _cu_seq_lens_q = cu_seq_lens_q
        _cu_seq_lens_k = cu_seq_lens_k
        _max_length_q = max_length_q
        _max_length_k = max_length_k

        if self.fsdp_config.torch_compile:
            # TODO: should we compile the vision model? Then process pixel_values
            _input_ids = pad_to_max_length(
                _input_ids, 0, self.fsdp_config.max_length, 1
            )
            _position_ids = pad_to_max_length(
                _position_ids, 0, self.fsdp_config.max_length, 1
            )
            if labels is not None:
                _labels = pad_to_max_length(
                    _labels, -100, self.fsdp_config.max_length, 1
                )
        else:
            if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
                multiple_of = sequence_parallel_mesh.size() * self.tp_mesh.size()
            else:
                multiple_of = self.tp_mesh.size()

            _input_ids = pad_to_multiple_of(_input_ids, 0, multiple_of, 1)
            _position_ids = pad_to_multiple_of(_position_ids, 0, multiple_of, 1)
            if labels is not None:
                _labels = pad_to_multiple_of(_labels, -100, multiple_of, 1)

        num_padded_tokens = _input_ids.numel() - input_ids.numel()

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            _input_ids = split_for_sequence_parallel(
                _input_ids, dim=1, sp_mesh=sequence_parallel_mesh
            )
            _position_ids = split_for_sequence_parallel(
                _position_ids, dim=1, sp_mesh=sequence_parallel_mesh
            )

            if labels is not None:
                _labels = split_for_sequence_parallel(
                    _labels, dim=1, sp_mesh=sequence_parallel_mesh
                )

        if self.tp_mesh.size() > 1:
            if labels is not None:
                _labels = split_for_sequence_parallel(
                    _labels, dim=1, sp_mesh=self.tp_mesh
                )

        if self.training and num_padded_tokens > 0:
            assert torch.all(cu_seq_lens_k == cu_seq_lens_q)
            _cu_seq_lens_q = torch.cat(
                [_cu_seq_lens_q, _cu_seq_lens_q[-1:] + num_padded_tokens], dim=0
            ).int()
            _cu_seq_lens_k = _cu_seq_lens_q

            _max_length_q = max(_max_length_q, num_padded_tokens)
            _max_length_k = _max_length_q

        outputs = self.patched_model(
            pixel_values,
            _input_ids,
            attention_mask,
            _position_ids,
            image_flags,
            past_key_values,
            _labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            label_shifted=label_shifted,
            cu_seq_lens_q=_cu_seq_lens_q,
            cu_seq_lens_k=_cu_seq_lens_k,
            max_length_q=_max_length_q,
            max_length_k=_max_length_k,
            block_table=block_table,
            prefilling=prefilling,
            sequence_parallel_mesh=self.sequence_parallel_mesh,
            tp_mesh=self.tp_mesh,
        )

        if outputs.loss is not None:
            valid_tokens = (cast(torch.Tensor, _labels) >= 0).sum()
            global_valid_tokens = (cast(torch.Tensor, labels) >= 0 ).sum()
            outputs.loss = outputs.loss * valid_tokens
            if self.tp_mesh.size() > 1:
                outputs.loss = DTensor.from_local(
                    outputs.loss, self.tp_mesh, placements=(Partial(),)
                ).full_tensor()
            if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
                outputs.loss = distF.all_reduce(
                    outputs.loss, group=sequence_parallel_mesh.get_group()
                )
            outputs.loss = outputs.loss / global_valid_tokens

        return outputs
