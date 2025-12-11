import torch
import types
from .qwen3_vl_config import Qwen3VLBaseConfig
from .modeling_vision import Qwen3VLVisionModel
from .modeling_projector import Qwen3VLProjector
from pathlib import Path
from xtuner.v1.loss import CELossContext
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
    FSDPModule,
)
from typing import Callable
import torch.distributed as dist
import torch.distributed.nn.functional as distF
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.utils import get_logger
from .modeling_vision import init_world_mesh
from typing_extensions import override
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.moe.moe import MoEModelOutputs
from xtuner.v1.model.moe.qwen3 import Qwen3MoE
from xtuner.v1.model.moe.qwen3vl_text import Qwen3VLTextMoE
from xtuner.v1.float8.float8_handler import Float8Handler
from torch.distributed.device_mesh import DeviceMesh
from xtuner.v1.data_proto.utils import split_for_sequence_parallel
from xtuner.v1.model import BaseModel, TorchCompileOption, DEFAULT_FLOAT8_CFG

logger = get_logger()


QWEN3VL_COMPILE_CFG: dict[str, TorchCompileOption] = {
    # "xtuner.v1.model.compose.qwen3_vl.modeling_projector.Qwen3VLProjector.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}


def to_hf_key_list_wrapper(fn: Callable[[str], list[str]], convertor: Callable[[str], str]):
    def wrapper(self, *args, **kwargs):
        return [convertor(i) for i in fn(*args, **kwargs)]

    return wrapper


class Qwen3VLForConditionalGeneration(BaseModel):
    config: Qwen3VLBaseConfig

    def __init__(self, config: Qwen3VLBaseConfig):
        super().__init__(config)  # type: ignore[arg-type]
        self.config = config

        self.vision_tower = Qwen3VLVisionModel(config.vision_config)
        self.multi_modal_projector = Qwen3VLProjector(config.projector_config)
        self.language_model = config.text_config.build()

        self._hf_path: Path | None = None

        if isinstance(self.language_model, Qwen3MoE):
            # TODO(YHC): This is a hack to make the language model compatible with HF
            _hf_prefix = "model.language_model."
            self.language_model.to_hf_key_list = types.MethodType(to_hf_key_list_wrapper(  # type: ignore
                fn=self.language_model.to_hf_key_list,
                convertor=lambda x: x.replace('model.', _hf_prefix)),
                self.language_model)
            self.language_model._init_load_spec()

        # Note: global load spec mapping for save_hf
        self.load_spec_mapping = {}
        for key, value in self.vision_tower.load_spec_mapping.items():
            self.load_spec_mapping['vision_tower.' + key] = value
        for key, value in self.multi_modal_projector.load_spec_mapping.items():
            self.load_spec_mapping['multi_modal_projector.' + key] = value
        for key, value in self.language_model.load_spec_mapping.items():
            self.load_spec_mapping['language_model.' + key] = value

        self._maybe_enable_compile(self.compile_cfg)
        self._freeze_modules()

    def _freeze_modules(self):
        freeze_vision = self.config.freeze_vision
        if freeze_vision:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()
            logger.info("Freeze vision tower")
        freeze_projector = self.config.freeze_projector
        if freeze_projector:
            self.multi_modal_projector.requires_grad_(False)
            self.multi_modal_projector.eval()
            logger.info("Freeze multi modal projector")
        freeze_language = self.config.freeze_language
        if freeze_language:
            self.language_model.requires_grad_(False)
            self.language_model.eval()
            logger.info("Freeze language model")

    @override
    def init_weights(self) -> None:
        self.vision_tower.init_weights()
        self.language_model.init_weights()
        self.multi_modal_projector.init_weights()

    def fully_shard(
            self,
            fsdp_config: FSDPConfig,
            float8_handler: Float8Handler | None = None,
    ):
        self.fsdp_config = fsdp_config
        # TODO: 判断其余模块是否已经被 fsdp 切分了

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

        if isinstance(self.vision_tower.blocks[0], FSDPModule):
            self.language_model.embed_tokens.set_modules_to_forward_prefetch(  # type: ignore
                [self.vision_tower.blocks[0]])
            self.vision_tower.blocks[-1].set_modules_to_forward_prefetch(  # type: ignore
                [self.multi_modal_projector])
        self.multi_modal_projector.set_modules_to_forward_prefetch([self.language_model])  # type: ignore
        self.language_model.set_modules_to_forward_prefetch([self.language_model.layers["0"]])  # type: ignore

        self._to_empty_meta()
        return self

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        return QWEN3VL_COMPILE_CFG

    def from_hf(self, hf_path: str | Path, strict=True):
        self._hf_path = Path(hf_path)

        if isinstance(hf_path, Path):
            hf_path = str(hf_path)

        _, _, missing_llm_keys = self.language_model.from_hf(hf_path, strict=False)
        _, _, missing_vision_keys = self.vision_tower.from_hf(hf_path, strict=False)
        _, _, missing_project_keys = self.multi_modal_projector.from_hf(hf_path, strict=False)

        missing = missing_llm_keys | missing_vision_keys | missing_project_keys
        if strict:
            if missing:
                raise RuntimeError(f"Missing parameters from {hf_path}: {list(missing)}. ")

    # llm 中的 param_to_safetensor 不会被触发，只能重写
    def param_to_safetensor(
        self,
        safetensor: torch.Tensor,
        hf_param_name: str,
    ):
        assert isinstance(hf_param_name, str)
        if isinstance(self.language_model, Qwen3VLTextMoE):
            if "gate_up_proj" in hf_param_name:
                # xtuner: num_experts * 2 * expert_dim, hidden_size
                # hf: num_experts, hidden_size, 2 * expert_dim
                num_experts = self.language_model.config.n_routed_experts
                hidden_size = safetensor.size(1)
                safetensor = safetensor.reshape(num_experts, -1, hidden_size)  # num_experts, 2 * expert_dim, hidden_size
                safetensor = safetensor.transpose(1, 2).contiguous()  # num_experts, hidden_size, 2 * expert_dim
            elif "down_proj" in hf_param_name:
                # xtuner: num_experts * hidden_size, expert_dim
                # hf: num_experts, expert_dim, hidden_size
                num_experts = self.language_model.config.n_routed_experts
                expert_dim = safetensor.size(1)
                safetensor = safetensor.reshape(num_experts, -1, expert_dim).transpose(1, 2).contiguous()
        else:
            safetensor = super().param_to_safetensor(safetensor, hf_param_name)
        return safetensor

    def scale_and_reduce_grad(self):
        self.language_model.scale_and_reduce_grad()

    def get_visual_features(self,
                            pixel_values: torch.Tensor,
                            grid_thw: torch.Tensor,
                            sequence_parallel_mesh: DeviceMesh | None = None):
        """Encodes images into continuous embeddings that can be forwarded to
        the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(n, h)`):
                The tensors corresponding to the input images.
            grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image in LLM.
            sequence_parallel_mesh (`DeviceMesh`, *optional*):
                The mesh to use for sequence parallelism.
        """
        image_embeds, deepstack_image_embeds = self.vision_tower(pixel_values,
                                                                 grid_thw=grid_thw,
                                                                 sequence_parallel_mesh=sequence_parallel_mesh)
        # merge
        image_embeds, deepstack_image_embeds = self.multi_modal_projector(image_embeds, deepstack_image_embeds)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
        deepstack_visual_embeds: list[torch.Tensor],
        origin_pixel_len: int,
        sequence_parallel_mesh: DeviceMesh | None = None,
    ):
        """Obtains multimodal placeholder mask from `input_ids` or
        `inputs_embeds`, and checks that the placeholder token count is equal
        to the length of multimodal features.

        If the lengths are different, an error is raised.
        """
        assert origin_pixel_len % 4 == 0, f"origin_pixel_len must be divisible by 4, but got {origin_pixel_len}. " \
                                          f"Please check dataset setting."
        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            input_ids_list = [torch.empty_like(input_ids) for _ in range(sequence_parallel_mesh.size())]
            dist.all_gather(input_ids_list, input_ids, group=sequence_parallel_mesh.get_group())
            input_ids = torch.cat(input_ids_list, dim=1)  # type: ignore

            visual_features_list = distF.all_gather(visual_features, group=sequence_parallel_mesh.get_group())
            visual_features = torch.cat(visual_features_list, dim=0)[:origin_pixel_len // 4]

            deepstack_image_embeds_list = []
            for deepstack_image_embed in deepstack_visual_embeds:
                deepstack_image_embed_ = distF.all_gather(deepstack_image_embed,
                                                          group=sequence_parallel_mesh.get_group())
                deepstack_image_embed = torch.cat(deepstack_image_embed_, dim=0)
                deepstack_image_embed = deepstack_image_embed[:origin_pixel_len// 4]
                deepstack_image_embeds_list.append(deepstack_image_embed)
            deepstack_visual_embeds = deepstack_image_embeds_list

        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        special_visual_mask = special_image_mask | special_video_mask

        n_visual_tokens = special_visual_mask.sum()

        if n_visual_tokens != visual_features.shape[0]:
            raise ValueError(
                f"Visual features and image|video tokens do not match: tokens: {n_visual_tokens}, features {visual_features.shape[0]}"
            )

        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            assert special_visual_mask.size(0) == 1
            # special_visual_mask must be divisible by sp size
            special_visual_mask_per_rank = split_for_sequence_parallel(
                special_visual_mask,
                dim=1,
                sp_mesh=sequence_parallel_mesh
            )
            n_visual_tokens_per_rank = special_visual_mask_per_rank.sum()
            sp_rank = sequence_parallel_mesh.get_local_rank()
            start_index = special_visual_mask[:, :(special_visual_mask.shape[1] // sequence_parallel_mesh.size()) * sp_rank].sum()
            end_index = start_index + n_visual_tokens_per_rank
            visual_features = visual_features[start_index:end_index]
            assert n_visual_tokens_per_rank == visual_features.shape[0]
            special_visual_mask = special_visual_mask_per_rank

            for i, deepstack_visual_embed in enumerate(deepstack_visual_embeds):
                deepstack_visual_embeds[i] = deepstack_visual_embed[start_index:end_index]
                assert n_visual_tokens_per_rank == deepstack_visual_embeds[i].shape[0]

        return special_visual_mask, visual_features, deepstack_visual_embeds

    def forward(
            self,
            seq_ctx: SequenceContext,
            loss_ctx: CELossContext
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        pixel_values = seq_ctx.pixel_values
        image_grid_thw = seq_ctx.image_grid_thw
        sequence_parallel_mesh = seq_ctx.sequence_parallel_mesh

        inputs_embeds = self.language_model.embed_tokens(input_ids)  # type: ignore

        if pixel_values is not None:
            assert image_grid_thw is not None
            assert input_ids is not None
            visual_embeds, deepstack_visual_embeds = self.get_visual_features(pixel_values,
                                                                              image_grid_thw,
                                                                              sequence_parallel_mesh)
            try:
                # To simplify and facilitate the processing of deepstack_visual_embeds inside language_model,
                # we all-gather visual_embeds, and then split them based on the input_ids,
                # then non-uniformly split them based on the input_ids
                visual_pos_masks, visual_features, deepstack_visual_embeds = self.get_placeholder_mask(
                    input_ids,
                    visual_features=visual_embeds,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    sequence_parallel_mesh=sequence_parallel_mesh,
                    origin_pixel_len = pixel_values.size(0)
                )
                inputs_embeds[visual_pos_masks] = inputs_embeds[visual_pos_masks] * 0.0 + visual_features
            except RuntimeError as e:
                print(f"!!!Warning: {e}, but continue anyway!!!!")
                inputs_embeds = inputs_embeds + visual_embeds.sum() * 0.0
                deepstack_visual_embeds = None
                visual_pos_masks = None
        else:
            pixel_values_dump = torch.randn(4, 1536, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_grid_thw = torch.tensor([[1, 2, 2]], device=inputs_embeds.device)
            viusal_embeds, deepstack_visual_embeds = self.get_visual_features(pixel_values_dump, image_grid_thw)
            inputs_embeds = inputs_embeds + viusal_embeds.sum() * 0.0
            for deepstack_visual_embed in deepstack_visual_embeds:
                inputs_embeds = inputs_embeds + deepstack_visual_embed.sum() * 0.0

            deepstack_visual_embeds = None
            visual_pos_masks = None

        if deepstack_visual_embeds is not None and len(deepstack_visual_embeds) == 0:
            assert seq_ctx.position_ids is not None
            assert seq_ctx.position_ids.ndim == 2, f"position_ids must be 2-dim when deepstack_visual_embeds is None," \
                                                   f" but got {seq_ctx.position_ids.ndim}"
            deepstack_visual_embeds = None
            visual_pos_masks = None

        # NOTE: 一定不要原地覆盖，否则第二次 forward 会缺少数据
        lang_seq_ctx = SequenceContext(input_ids=None,
                                       cu_seq_lens_q=seq_ctx.cu_seq_lens_q,
                                       cu_seq_lens_k=seq_ctx.cu_seq_lens_k,
                                       max_length_q=seq_ctx.max_length_q,
                                       max_length_k=seq_ctx.max_length_k,
                                       position_ids=seq_ctx.position_ids,
                                       num_padding=seq_ctx.num_padding,
                                       sequence_parallel_mesh=seq_ctx.sequence_parallel_mesh,
                                       inputs_embeds=inputs_embeds,
                                       deepstack_visual_embeds=deepstack_visual_embeds,
                                       visual_pos_masks=visual_pos_masks)
        outputs = self.language_model(
            lang_seq_ctx,
            loss_ctx
        )
        return outputs
