# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash compose model, see doc/xtuner_glm5p3flash_design.md F6.

Splice uses the global ``mm_token_type_ids`` (1=image, 2=video) to place visual features, never
``input_ids == video_token_id`` -- that token is only a pre-expansion chat-template marker and
never appears in the tokenized sequence (design doc F1.b). A placeholder-count mismatch raises
immediately; this module never swallows the mismatch with a bare ``except Exception: continue``
the way the Qwen3-VL compose path does (design doc §16.2 explicitly forbids copying that).
"""

import torch
from typing_extensions import override

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import CELossContext
from xtuner.v1.model import TorchCompileOption
from xtuner.v1.model.moe.moe import MoEModelOutputs
from xtuner.v1.utils import log_rank0

from ..base import BaseComposeModel
from .glm53_config import Glm53BaseConfig
from .vision_utils import flatten_video_grid_thw


GLM53_COMPILE_CFG: dict[str, TorchCompileOption] = {}


class Glm53ForConditionalGeneration(BaseComposeModel):
    config: Glm53BaseConfig

    def __init__(self, config: Glm53BaseConfig) -> None:
        self.only_llm_forward = config.only_llm_forward
        if self.only_llm_forward:
            config.freeze_vision = True
            config.freeze_projector = True
            log_rank0.warning("only_llm_forward is True, vision and projector will be frozen.")
        super().__init__(config)  # type: ignore[arg-type]

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        return GLM53_COMPILE_CFG

    def get_visual_features(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.vision_tower(pixel_values, grid_thw)
        return self.multi_modal_projector(hidden_states)

    def _splice(
        self,
        inputs_embeds: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        modality: int,
        features: torch.Tensor,
    ) -> torch.Tensor:
        mask = mm_token_type_ids == modality
        n_tokens = int(mask.sum().item())
        if n_tokens != features.shape[0]:
            raise ValueError(
                f"GLM-5.3-Flash modality={modality} placeholder count {n_tokens} != visual feature "
                f"count {features.shape[0]}. Refusing to continue training on a corrupted splice "
                "(design doc §16.2 -- unlike Qwen3-VL, this is not caught and skipped)."
            )
        inputs_embeds[mask] = inputs_embeds[mask] * 0.0 + features.to(inputs_embeds.dtype)
        return inputs_embeds

    def _prepare_llm_inputs(self, seq_ctx: SequenceContext) -> torch.Tensor:
        # The splice below indexes inputs_embeds with the *global* mm_token_type_ids. Under LLM
        # sequence parallelism the embeddings would be sharded while the mask is not, and the
        # mismatch would surface as a misleading "visual feature count" error. Vision SP is a
        # documented gap (design doc F2 known gaps), so refuse it by name instead.
        sp_mesh = seq_ctx.sequence_parallel_mesh
        assert sp_mesh is None or sp_mesh.size() == 1, (
            "GLM-5.3-Flash VL does not support sequence parallel training yet: the visual splice "
            "needs global mm_token_type_ids (design doc F2 known gaps)."
        )
        input_ids = seq_ctx.input_ids
        assert input_ids is not None
        inputs_embeds = self.language_model.embed_tokens(input_ids)  # type: ignore[attr-defined]

        has_image = seq_ctx.pixel_values is not None
        has_video = seq_ctx.pixel_values_videos is not None

        if not has_image and not has_video:
            if not self.only_llm_forward:
                # Dummy visual forward so every FSDP rank materializes/updates vision+projector
                # params symmetrically even on a pure-text micro-batch (design doc F6 point 4).
                # 2x2 grid (4 raw patches) is the smallest valid input: one spatial_merge_size
                # block, so the projector's downsample/merger see exactly one output token.
                patch_embed = self.vision_tower.patch_embed
                merge = self.vision_tower.spatial_merge_size
                patch_dim = patch_embed.in_channels * patch_embed.temporal_patch_size * patch_embed.patch_size**2
                dummy_pixel_values = torch.randn(
                    merge * merge, patch_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                dummy_grid_thw = torch.tensor([[1, merge, merge]], device=inputs_embeds.device)
                dummy_feats = self.get_visual_features(dummy_pixel_values, dummy_grid_thw)
                inputs_embeds = inputs_embeds + dummy_feats.sum() * 0.0
            return inputs_embeds

        assert not self.only_llm_forward, "only_llm_forward is True, but pixel_values/pixel_values_videos is not None."
        assert not (has_image and has_video), (
            "GLM-5.3-Flash TokenizeFn only supports image-only or video-only samples (F1.b); "
            "a mixed-media SequenceContext should never reach the compose model."
        )
        assert seq_ctx.mm_token_type_ids is not None, (
            "mm_token_type_ids is required to splice visual features; input_ids == video_token_id "
            "cannot be used post-expansion (design doc F1.b/§16.2)."
        )
        # Kept as [batch, seq] (not squeezed) so boolean-indexing inputs_embeds[mask] collapses
        # the matching leading dims instead of indexing along the batch dim. It is the global
        # mask -- the assertion at the top of this method is what keeps that true.
        mm_token_type_ids = seq_ctx.mm_token_type_ids

        if has_image:
            assert seq_ctx.image_grid_thw is not None
            features = self.get_visual_features(seq_ctx.pixel_values, seq_ctx.image_grid_thw)  # type: ignore[arg-type]
            inputs_embeds = self._splice(inputs_embeds, mm_token_type_ids, modality=1, features=features)
        else:
            assert seq_ctx.video_grid_thw is not None
            flat_grid_thw = flatten_video_grid_thw(seq_ctx.video_grid_thw)
            features = self.get_visual_features(seq_ctx.pixel_values_videos, flat_grid_thw)  # type: ignore[arg-type]
            inputs_embeds = self._splice(inputs_embeds, mm_token_type_ids, modality=2, features=features)

        return inputs_embeds

    def forward(
        self,
        seq_ctx: SequenceContext | list[SequenceContext],
        loss_ctx: dict[str, CELossContext] | None = None,
    ) -> MoEModelOutputs:
        if isinstance(seq_ctx, list):
            lang_seq_ctx: SequenceContext | list[SequenceContext] = [
                single.copy(input_ids=None, inputs_embeds=self._prepare_llm_inputs(single)) for single in seq_ctx
            ]
        else:
            lang_seq_ctx = seq_ctx.copy(input_ids=None, inputs_embeds=self._prepare_llm_inputs(seq_ctx))
        return self.language_model(lang_seq_ctx, loss_ctx)
