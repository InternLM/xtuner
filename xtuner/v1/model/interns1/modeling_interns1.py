import types
from typing import cast, Callable
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF

from xtuner.v1.config.base_model import MoEModelOutputs
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.ops.comm import split_for_sequence_parallel
from xtuner.v1.utils import get_logger, get_padding_length, get_device
from xtuner.v1.model import BaseModel

from .modeling_vision import InternS1VisionModel, InternS1MultiModalProjector, init_world_mesh
from typing_extensions import override
from xtuner.v1.config import FSDPConfig, InternS1Config
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.data_proto import CELossContext
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

DEVICE = get_device()
logger = get_logger()


def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


def to_hf_key_list_wrapper(fn: Callable[[str], list[str]], convertor: Callable[[str], str]):
    def wrapper(self, *args, **kwargs):
        return [convertor(i) for i in fn(*args, **kwargs)]
    return wrapper


class InternS1ForConditionalGeneration(BaseModel):
    def __init__(self, config: InternS1Config):
        super().__init__()

        self.select_layer = config.vision_feature_layer
        self.downsample_ratio = config.downsample_ratio

        vision_config = config.vision_config
        text_config = config.text_config

        self.vision_tower = InternS1VisionModel(vision_config)
        self.multi_modal_projector = InternS1MultiModalProjector(config)

        self.language_model = text_config.build()

        # TODO(YHC): This is a hack to make the language model compatible with HF
        _hf_prefix = "model.language_model."
        self.language_model.to_hf_key_list = types.MethodType(to_hf_key_list_wrapper(  # type: ignore
            fn=self.language_model.to_hf_key_list,
            convertor=lambda x: x.replace('model.', _hf_prefix)),
            self.language_model)
        self.language_model.load_spec_mapping = self.language_model._init_load_spec()

        self.img_context_token_id = config.image_token_id

    @override
    def fully_shard(
            self,
            fsdp_config: FSDPConfig,
            float8_handler: Float8Handler | None = None,
    ):
        self.language_model.fully_shard(fsdp_config, float8_handler=float8_handler)
        self.vision_tower.fully_shard(fsdp_config, float8_handler=float8_handler)
        self.multi_modal_projector.fully_shard(fsdp_config, float8_handler=float8_handler)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        fsdp_mesh = init_world_mesh(fsdp_config)
        fully_shard(
            self,
            mesh=fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self

    def from_hf(self, hf_path: str | Path, strict=True):
        _, _, missing_llm_keys = self.language_model.from_hf(hf_path,  strict=False)
        _, _, missing_vision_keys = self.vision_tower.from_hf(hf_path,  strict=False)
        _, _, missing_project_keys = self.multi_modal_projector.from_hf(hf_path, strict=False)

        missing = missing_llm_keys | missing_vision_keys | missing_project_keys
        if strict:
            if missing:
                raise RuntimeError(f"Missing parameters from {hf_path}: {list(missing)}. ")

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_tower(
                pixel_values=pixel_values, output_hidden_states=False
            ).last_hidden_state
        else:
            vit_embeds = self.vision_tower(
                pixel_values=pixel_values, output_hidden_states=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.multi_modal_projector(vit_embeds)
        return vit_embeds

    def forward(
            self,
            seq_ctx: SequenceContext,  # todo(@yehaochen): support intra layer micro-batch
            loss_ctx: CELossContext,
            return_router_results: bool = True,
            return_hidden_states: bool = False
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        pixel_values = seq_ctx.pixel_values
        image_flags = seq_ctx.image_flags
        sequence_parallel_mesh = seq_ctx.sequence_parallel_mesh

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None and image_flags is not None:
            image_flags = cast(torch.LongTensor, image_flags.squeeze(-1))

            # in-place op on custom-function outputs will spoil autograd
            inputs_embeds = inputs_embeds.clone()

            if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
                vit_batch_size = pixel_values.shape[0]
                divisors = [sequence_parallel_mesh.size()]
                pad_size = get_padding_length(vit_batch_size, divisors)
                if pad_size != 0:
                    pixel_values = torch.cat(
                        [
                            pixel_values,  # type: ignore
                            pixel_values[0:1].repeat(pad_size, *[1] * (pixel_values.dim() - 1)),
                        ],
                        dim=0,
                    )
                pixel_values = pixel_values.chunk(sequence_parallel_mesh.size(), dim=0)[  # type: ignore
                    sequence_parallel_mesh.get_local_rank()
                ]

            vit_embeds = self.extract_feature(pixel_values)

            if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
                vit_embeds_list = distF.all_gather(vit_embeds, group=sequence_parallel_mesh.get_group())
                vit_embeds = torch.cat(vit_embeds_list, dim=0)[:vit_batch_size]

            vit_embeds = vit_embeds[image_flags == 1]

            if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
                inputs_embeds_list = distF.all_gather(inputs_embeds, group=sequence_parallel_mesh.get_group())
                inputs_embeds = torch.cat(inputs_embeds_list, dim=1)

                input_ids_list = [torch.empty_like(input_ids) for _ in range(sequence_parallel_mesh.size())]
                dist.all_gather(input_ids_list, input_ids, group=sequence_parallel_mesh.get_group())
                input_ids = torch.cat(input_ids_list, dim=1)  # type: ignore

            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)

            input_ids = cast(torch.LongTensor, input_ids.reshape(B * N))

            selected = input_ids == self.img_context_token_id

            try:
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, inputs_embeds[selected].shape={inputs_embeds[selected].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.sum() * 0

            inputs_embeds = inputs_embeds.reshape(B, N, C)
        else:
            # in-place op on custom-function outputs will spoil autograd
            inputs_embeds = inputs_embeds.clone()

        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            # TODO: 如果外面传过来的没有 pad，那么这个地方会比较麻烦，因为不仅仅要 input_embeds 需要 pad，
            #  position_ids,cu_seq_lens_k 等都要重算？
            inputs_embeds = split_for_sequence_parallel(inputs_embeds, dim=1, sp_mesh=sequence_parallel_mesh)

        seq_ctx.image_flags = None
        seq_ctx.pixel_values = None
        seq_ctx.input_ids = None  # type: ignore
        seq_ctx.inputs_embeds = inputs_embeds

        outputs = self.language_model(
            seq_ctx,
            loss_ctx,
            return_router_results=return_router_results,
            return_hidden_states=return_hidden_states,
        )
        return outputs
