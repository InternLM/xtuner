import torch
from xtuner.v1.model import BaseModel
from .qwen3_vl_config import Qwen3VLBaseConfig
from .modeling_vision import Qwen3VLVisionModel
from .modeling_projector import Qwen3VLProjector
from pathlib import Path
from xtuner.v1.loss import CELossContext
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.utils import get_logger
from .modeling_vision import init_world_mesh
from typing_extensions import override
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.moe.moe import MoEModelOutputs
from xtuner.v1.model.moe.qwen3vl_text import Qwen3VLTextMoE
from xtuner.v1.float8.float8_handler import Float8Handler

logger = get_logger()


class Qwen3VLForConditionalGeneration(BaseModel):
    config: Qwen3VLBaseConfig

    def __init__(self, config: Qwen3VLBaseConfig):
        super().__init__()
        self.config = config

        self.vision_tower = Qwen3VLVisionModel(config.vision_config)
        self.multi_modal_projector = Qwen3VLProjector(config.projector_config)
        self.language_model = config.text_config.build()

        self._hf_path: Path | None = None

        # Note: global load spec mapping for save_hf
        self.load_spec_mapping = {}
        for key, value in self.vision_tower.load_spec_mapping.items():
            self.load_spec_mapping['vision_tower.' + key] = value
        for key, value in self.multi_modal_projector.load_spec_mapping.items():
            self.load_spec_mapping['multi_modal_projector.' + key] = value
        for key, value in self.language_model.load_spec_mapping.items():
            self.load_spec_mapping['language_model.' + key] = value

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
        self._to_empty_meta()
        return self

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

    def get_visual_features(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image in LLM.
        """
        image_embeds, deepstack_image_embeds = self.vision_tower(pixel_values, grid_thw=grid_thw)

        # merge
        image_embeds, deepstack_image_embeds = self.multi_modal_projector(image_embeds, deepstack_image_embeds)

        split_sizes = (grid_thw.prod(-1) // self.vision_tower.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        visual_features: torch.Tensor,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        special_visual_mask = special_image_mask | special_video_mask

        n_visual_tokens = special_visual_mask.sum()
        special_visual_mask = special_visual_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[special_visual_mask].numel() != visual_features.numel():
            raise ValueError(
                f"Visual features and image|video tokens do not match: tokens: {n_visual_tokens}, features {visual_features.shape[0]}"
            )
        return special_visual_mask

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
            viusal_embeds, deepstack_visual_embeds = self.get_visual_features(pixel_values, image_grid_thw)
            viusal_embeds = torch.cat(viusal_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            visual_pos_masks = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, visual_features=viusal_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(visual_pos_masks, viusal_embeds)
        else:
            # 构建假数据，考虑到 moe 特性，最好不要构建全 0 数据
            pixel_values = torch.randn(4, 1536, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_grid_thw = torch.tensor([[1, 2, 2]], device=inputs_embeds.device)
            viusal_embeds, _ = self.get_visual_features(pixel_values, image_grid_thw)
            viusal_embeds = torch.cat(viusal_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + viusal_embeds.sum() * 0.0
            deepstack_visual_embeds = None
            visual_pos_masks = None

        seq_ctx.pixel_values = None
        seq_ctx.input_ids = None  # type: ignore
        seq_ctx.inputs_embeds = inputs_embeds
        seq_ctx.deepstack_visual_embeds = deepstack_visual_embeds
        seq_ctx.visual_pos_masks = visual_pos_masks

        outputs = self.language_model(
            seq_ctx,
            loss_ctx
        )
        return outputs
