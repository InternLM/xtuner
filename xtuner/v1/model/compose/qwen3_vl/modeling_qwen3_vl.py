import torch
import types
from .qwen3_vl_config import Qwen3VLBaseConfig
from xtuner.v1.loss import CELossContext
import torch.distributed as dist
import torch.distributed.nn.functional as distF
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.utils import get_logger
from typing_extensions import override
from xtuner.v1.model.moe.moe import MoEModelOutputs
from xtuner.v1.model.moe.qwen3 import Qwen3MoE
from torch.distributed.device_mesh import DeviceMesh
from xtuner.v1.data_proto.utils import split_for_sequence_parallel
from xtuner.v1.model import TorchCompileOption, DEFAULT_FLOAT8_CFG
from ..base import BaseComposeModel, to_hf_key_list_wrapper

logger = get_logger()

QWEN3VL_COMPILE_CFG: dict[str, TorchCompileOption] = {
    # "xtuner.v1.model.compose.qwen3_vl.modeling_projector.Qwen3VLProjector.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}


class Qwen3VLForConditionalGeneration(BaseComposeModel):
    config: Qwen3VLBaseConfig

    def __init__(self, config: Qwen3VLBaseConfig):
        super().__init__(config)  # type: ignore[arg-type]

        # if type(self.language_model) is Qwen3MoE:
        #     # TODO(YHC): This is a hack to make the language model compatible with HF
        #     _hf_prefix = "model.language_model."
        #     self.language_model.to_hf_key_list = types.MethodType(to_hf_key_list_wrapper(  # type: ignore
        #         fn=self.language_model.to_hf_key_list,
        #         convertor=lambda x: x.replace('model.', _hf_prefix)),
        #         self.language_model)
        #     self.language_model._init_load_spec()

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        return QWEN3VL_COMPILE_CFG

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
                deepstack_image_embed = deepstack_image_embed[:origin_pixel_len // 4]
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
            start_index = special_visual_mask[:,
                          :(special_visual_mask.shape[1] // sequence_parallel_mesh.size()) * sp_rank].sum()
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
                    origin_pixel_len=pixel_values.size(0)
                )
                inputs_embeds[visual_pos_masks] = inputs_embeds[visual_pos_masks] * 0.0 + visual_features
            except Exception as e:
                logger.error(f"!!!Warning: {e}, but continue anyway!!!!")
                inputs_embeds = inputs_embeds + visual_embeds.sum() * 0.0
                for deepstack_visual_embed in deepstack_visual_embeds:
                    inputs_embeds = inputs_embeds + deepstack_visual_embed.sum() * 0.0
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
                                       rollout_routed_experts=seq_ctx.rollout_routed_experts,
                                       deepstack_visual_embeds=deepstack_visual_embeds,
                                       visual_pos_masks=visual_pos_masks)
        outputs = self.language_model(
            lang_seq_ctx,
            loss_ctx
        )
        return outputs
