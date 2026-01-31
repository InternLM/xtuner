import types
from typing import cast, Self

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF

from xtuner.v1.model.moe.moe import MoEModelOutputs
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.utils import get_logger, get_padding_length, get_device
from xtuner.v1.model import TorchCompileOption, DEFAULT_FLOAT8_CFG
from xtuner.v1.loss.utils import sp_split

from .modeling_vision import init_world_mesh
from typing_extensions import override
from xtuner.v1.config import FSDPConfig
from .intern_s1_config import InternS1BaseConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import CELossContext
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from ..base import BaseComposeModel, to_hf_key_list_wrapper

DEVICE = get_device()
logger = get_logger()

INTERNS1_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.model.compose.intern_s1.modeling_projector.InternS1MultiModalProjector.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.compose.intern_s1.modeling_vision.InternS1VisionLayer.attention_pre_forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.compose.intern_s1.modeling_vision.InternS1VisionLayer.attention_post_forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}


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


class InternS1ForConditionalGeneration(BaseComposeModel):
    config: InternS1BaseConfig

    def __init__(self, config: InternS1BaseConfig):
        super().__init__(config)  # type: ignore[arg-type]

        self.select_layer = config.vision_feature_layer
        self.downsample_ratio = config.downsample_ratio

        self.img_context_token_id = config.image_token_id
        self.image_size = config.vision_config.image_size[0]

    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
    ) -> Self:
        self.fsdp_config = fsdp_config
        self.language_model.fully_shard(self.fsdp_config)
        self.vision_tower.fully_shard(self.fsdp_config)
        self.multi_modal_projector.fully_shard(self.fsdp_config)
        # TODO: 判断其余模块是否已经被 fsdp 切分了

        # NOTE: 暂时只能在这个地方进行 checkpoint_wrapper
        # TODO: 当只训练某个部分时候，不能开启 checkpoint，否则 grad 是 None, 后续有需要再支持。
        # self.multi_modal_projector = checkpoint_wrapper(self.multi_modal_projector,  # type: ignore
        #                                                     checkpoint_impl=CheckpointImpl.REENTRANT)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )

        self.fsdp_mesh = init_world_mesh()
        # Note: 非常关键，不能删除这个 assert
        assert self.fsdp_mesh is not None

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )

        self.language_model.embed_tokens.set_modules_to_forward_prefetch(   # type: ignore
            [self.vision_tower.encoder.layer[0]])
        self.vision_tower.encoder.layer[-1].set_modules_to_forward_prefetch(   # type: ignore
            [self.multi_modal_projector])
        self.multi_modal_projector.set_modules_to_forward_prefetch([self.language_model])  # type: ignore
        self.language_model.set_modules_to_forward_prefetch([self.language_model.layers["0"]])  # type: ignore

        self._to_empty_meta()
        return self

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
            seq_ctx: SequenceContext,
            loss_ctx: CELossContext
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        pixel_values = seq_ctx.pixel_values
        sequence_parallel_mesh = seq_ctx.sequence_parallel_mesh

        inputs_embeds = self.language_model.embed_tokens(input_ids)  # type: ignore

        if pixel_values is not None:
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

            if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
                inputs_embeds_list = distF.all_gather(inputs_embeds, group=sequence_parallel_mesh.get_group())
                inputs_embeds = torch.cat(inputs_embeds_list, dim=1)

                assert input_ids is not None
                input_ids_list = [torch.empty_like(input_ids) for _ in range(sequence_parallel_mesh.size())]
                dist.all_gather(input_ids_list, input_ids, group=sequence_parallel_mesh.get_group())
                input_ids = torch.cat(input_ids_list, dim=1)  # type: ignore

            B, N, C = inputs_embeds.shape
            assert inputs_embeds is not None
            inputs_embeds = inputs_embeds.reshape(B * N, C)

            assert input_ids is not None
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

            if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
                inputs_embeds = sp_split(inputs_embeds, sequence_parallel_mesh, 1, 0)
        else:
            fake_pixel_values = torch.randn(1, 3, self.image_size, self.image_size,
                                            device=inputs_embeds.device,
                                            dtype=inputs_embeds.dtype)
            vit_embeds = self.extract_feature(fake_pixel_values)
            inputs_embeds = inputs_embeds + vit_embeds.sum() * 0

        # NOTE: 一定不要原地覆盖，否则第二次 forward 会缺少数据
        lang_seq_ctx = SequenceContext(input_ids=None,
                                       cu_seq_lens_q=seq_ctx.cu_seq_lens_q,
                                       cu_seq_lens_k=seq_ctx.cu_seq_lens_k,
                                       max_length_q=seq_ctx.max_length_q,
                                       max_length_k=seq_ctx.max_length_k,
                                       position_ids=seq_ctx.position_ids,
                                       num_padding=seq_ctx.num_padding,
                                       sequence_parallel_mesh=seq_ctx.sequence_parallel_mesh,
                                       rollout_routed_experts=seq_ctx.rollout_routed_experts,
                                       inputs_embeds=inputs_embeds)

        outputs = self.language_model(
            lang_seq_ctx,
            loss_ctx
        )
        return outputs

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        return INTERNS1_COMPILE_CFG
