import torch 
from xtuner.v1.model import BaseModel
from type import Optional
from .qwen3_vl_config import Qwen3VLBaseConfig
from .modeling_vision import Qwen3VLVisionModel
from .modeling_projector import Qwen3VLProjector

from xtuner.v1.loss import CELossContext
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from xtuner.v1.model.moe.moe import SequenceContext


class Qwen3VLForConditionalGeneration(BaseModel):
    config: Qwen3VLBaseConfig
    
    def __init__(self, config: Qwen3VLBaseConfig):
        super().__init__()
        self.config = config
        
        self.vision_tower = Qwen3VLVisionModel(config.vision_config)
        
        self.multi_modal_projector = Qwen3VLProjector(config.projector_config)
        
        self.language_model = config.text_config.build()
        
    def get_visual_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.vision_tower.dtype)
        image_embeds, deepstack_image_embeds = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
        
        # merge
        image_embeds = self.multi_modal_projector(image_embeds)
        for i, _image_embeds in enumerate(deepstack_image_embeds):
            _image_embeds = self.multi_modal_projector(_image_embeds)
            deepstack_image_embeds[i] = _image_embeds
            
        split_sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        visual_features: torch.FloatTensor,
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
            pass
        
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
 