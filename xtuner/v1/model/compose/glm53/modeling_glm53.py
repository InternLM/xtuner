# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash compose model, see doc/xtuner_glm5p3flash_design.md F6.

Splice uses ``mm_token_type_ids`` (1=image, 2=video) to place visual features, never
``input_ids == video_token_id`` -- that token is only a pre-expansion chat-template marker and
never appears in the tokenized sequence (design doc F1.b). A placeholder-count mismatch raises
immediately; this module never swallows the mismatch with a bare ``except Exception: continue``
the way the Qwen3-VL compose path does (design doc §16.2 explicitly forbids copying that).
"""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF
from torch.distributed.device_mesh import DeviceMesh
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

    def get_visual_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        sequence_parallel_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        """Encode patches into language-width visual features.

        Under sequence parallelism the tower shards the patch sequence merge-aligned, so both the
        tower and the projector run on this rank's shard only; the caller gathers the features.

        Args:
            pixel_values (torch.Tensor): Raw patches ``[num_patches, patch_dim]``.
            grid_thw (torch.Tensor): Per-image ``[t, h, w]`` rows, video already expanded.
            sequence_parallel_mesh (DeviceMesh | None): The sequence-parallel mesh, if any.

        Returns:
            torch.Tensor: Visual features; this rank's shard when sequence parallel is active.
        """
        hidden_states = self.vision_tower(pixel_values, grid_thw, sequence_parallel_mesh)
        return self.multi_modal_projector(hidden_states)

    def _gather_visual_features(
        self, features: torch.Tensor, num_features: int, sequence_parallel_mesh: DeviceMesh
    ) -> torch.Tensor:
        """Reassemble the globally-ordered visual features from every SP rank's
        shard.

        The tower pads the patch sequence up to a multiple of ``sp_size * merge_unit``, so the
        gathered features carry trailing padding rows that `num_features` trims off. The gather is
        the autograd-aware one: its backward reduce-scatters, handing each rank the gradient of
        exactly the shard it produced.
        """
        gathered = distF.all_gather(features, group=sequence_parallel_mesh.get_group())
        return torch.cat(gathered, dim=0)[:num_features]

    def _splice(
        self,
        inputs_embeds: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        features: torch.Tensor,
        num_features: dict[int, int],
        sequence_parallel_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        """Write the pack's visual features over their placeholder positions.

        ``features`` comes from a single tower call over ``[image patches; video patches]``, so its
        rows are the modalities in that order, ``num_features`` of each. Under SP ``inputs_embeds``
        and ``mm_token_type_ids`` are this rank's shard (split with ``input_ids``) and ``features``
        is this rank's merge-aligned shard of the tower output: the mask is gathered once to learn
        where each modality's placeholders fall globally, and the features are gathered back to
        global order before each modality takes its slice. Every count check compares *global*
        totals, so a corrupted sample is caught identically with and without SP.
        """
        sp_size = sequence_parallel_mesh.size() if sequence_parallel_mesh is not None else 1
        if sp_size > 1:
            assert sequence_parallel_mesh is not None
            gathered = [torch.empty_like(mm_token_type_ids) for _ in range(sp_size)]
            dist.all_gather(gathered, mm_token_type_ids.contiguous(), group=sequence_parallel_mesh.get_group())
            global_mm_token_type_ids = torch.cat(gathered, dim=-1)
            features = self._gather_visual_features(features, sum(num_features.values()), sequence_parallel_mesh)
            local_len = mm_token_type_ids.shape[-1]
            preceding = global_mm_token_type_ids[..., : sequence_parallel_mesh.get_local_rank() * local_len]
        else:
            global_mm_token_type_ids = mm_token_type_ids

        offset = 0
        for modality, count in num_features.items():
            n_tokens = int((global_mm_token_type_ids == modality).sum().item())
            if n_tokens != count:
                raise ValueError(
                    f"GLM-5.3-Flash modality={modality} placeholder count {n_tokens} != visual feature "
                    f"count {count}. Refusing to continue training on a corrupted splice (design doc "
                    "§16.2 -- unlike Qwen3-VL, this is not caught and skipped)."
                )
            modality_features = features[offset : offset + count]
            offset += count
            mask = mm_token_type_ids == modality
            if sp_size > 1:
                # This rank's slice starts after the same-modality placeholders on preceding ranks.
                start = int((preceding == modality).sum().item())
                modality_features = modality_features[start : start + int(mask.sum().item())]
            inputs_embeds[mask] = inputs_embeds[mask] * 0.0 + modality_features.to(inputs_embeds.dtype)
        return inputs_embeds

    def _prepare_llm_inputs(self, seq_ctx: SequenceContext) -> torch.Tensor:
        # Under SP every rank holds the same sample, split along the sequence: `input_ids` and
        # `mm_token_type_ids` are this rank's slice, the tower shards the patches merge-aligned,
        # and `_splice` gathers the features back before writing this rank's share.
        sp_mesh = seq_ctx.sequence_parallel_mesh
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
        assert seq_ctx.mm_token_type_ids is not None, (
            "mm_token_type_ids is required to splice visual features; input_ids == video_token_id "
            "cannot be used post-expansion (design doc F1.b/§16.2)."
        )
        # Kept as [batch, seq] (not squeezed) so boolean-indexing inputs_embeds[mask] collapses
        # the matching leading dims instead of indexing along the batch dim. Under SP this is
        # this rank's slice, split alongside input_ids; `_splice` reconciles it with the features.
        mm_token_type_ids = seq_ctx.mm_token_type_ids

        # A *sample* never mixes image and video -- the tokenize fn rejects that (F1.b) -- but a
        # *pack* routinely holds an image sample next to a video sample, and this is the pack.
        #
        # The tower runs exactly once per pack, whatever mix of modalities it holds. Under FSDP each
        # tower call issues the tower's parameter all-gathers across every rank, and which
        # modalities a pack holds is per-rank data: running the tower once per modality made a rank
        # holding both call it twice while its peers called it once, desynchronizing those
        # all-gathers and hanging the job (NCCL watchdog on the tower's root unit,
        # patch_embed + post_layernorm). One call over the concatenated patches keeps every rank's
        # collective sequence identical -- the same reason the pure-text path above runs a dummy.
        merge_unit = self.vision_tower.spatial_merge_size**2
        pixel_values: list[torch.Tensor] = []
        grid_thw: list[torch.Tensor] = []
        num_features: dict[int, int] = {}
        if has_image:
            assert seq_ctx.image_grid_thw is not None
            pixel_values.append(seq_ctx.pixel_values)  # type: ignore[arg-type]
            grid_thw.append(seq_ctx.image_grid_thw)
            num_features[1] = int((seq_ctx.image_grid_thw.prod(-1) // merge_unit).sum().item())
        if has_video:
            assert seq_ctx.video_grid_thw is not None
            flat_grid_thw = flatten_video_grid_thw(seq_ctx.video_grid_thw)
            pixel_values.append(seq_ctx.pixel_values_videos)  # type: ignore[arg-type]
            grid_thw.append(flat_grid_thw)
            num_features[2] = int((flat_grid_thw.prod(-1) // merge_unit).sum().item())
        features = self.get_visual_features(torch.cat(pixel_values), torch.cat(grid_thw), sp_mesh)
        return self._splice(inputs_embeds, mm_token_type_ids, features, num_features, sp_mesh)

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
