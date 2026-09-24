"""GLM-5.3-Flash compose model tests, see doc/xtuner_glm5p3flash_design.md F6.

Tiny synthetic config (mirrors test_glm53_text_moe.py / test_glm53_vision.py fixtures). GPU
required: KDA's Triton kernel has no CPU backend.
"""

import pytest
import torch
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.compose.glm53 import Glm53BaseConfig, Glm53ProjectorConfig, Glm53VisionConfig
from xtuner.v1.model.moe.glm53.glm53 import Glm53TextMoEConfig
from xtuner.v1.model.moe.glm53.nope_dsa_mla import NoPEDSAMLAConfig
from xtuner.v1.module.attention.kda import KDAConfig
from xtuner.v1.module.decoder_layer.mhc import MHCConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.utils.test_utils import init_data_mesh


HIDDEN = 32
MERGE = 2
SEQ_LEN = 128


def _build_model():
    text_cfg = Glm53TextMoEConfig(
        compile_cfg=False,
        vocab_size=200,
        pad_token_id=0,
        eos_token_id=1,
        hf_eos_token_id=[1],
        num_hidden_layers=3,
        first_k_dense_replace=1,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=48,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        attention=NoPEDSAMLAConfig(
            num_attention_heads=4,
            head_dim=16,
            kv_lora_rank=24,
            q_lora_rank=16,
            qk_rope_head_dim=0,
            qk_nope_head_dim=8,
            v_head_dim=8,
            index_topk=4,
            index_head_dim=8,
            index_n_heads=2,
            index_kpool=2,
            sparse_mla_backend="torch",
            indexer_backend="torch",
            freeze_dsa_indexer=True,
        ),
        linear_attention=KDAConfig(num_heads=2, head_dim=16, gate_lower_bound=-5.0),
        glm53_layer_types=["linear_attention", "deepseek_sparse_attention", "linear_attention"],
        mhc=MHCConfig(hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=4),
        router=NoAuxRouterConfig(
            n_group=1, topk_group=1, scoring_func="sigmoid", norm_topk_prob=True, router_scaling_factor=1.0
        ),
        mtp_config=None,
        dispatcher="all2all",
        ep_size=1,
    )
    vision_cfg = Glm53VisionConfig(
        depth=2,
        hidden_size=HIDDEN,
        num_heads=4,
        intermediate_size=64,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=MERGE,
        rms_norm_eps=1e-6,
        attn_impl="eager_attention",
        fully_shard=False,
    )
    proj_cfg = Glm53ProjectorConfig(
        vision_hidden_size=HIDDEN,
        out_hidden_size=HIDDEN,
        spatial_merge_size=MERGE,
        projection_intermediate_size=48,
        fully_shard=False,
    )
    compose_cfg = Glm53BaseConfig(
        compile_cfg=False, vision_config=vision_cfg, projector_config=proj_cfg, text_config=text_cfg
    )
    model = compose_cfg.build().cuda().to(torch.bfloat16)
    torch.manual_seed(0)
    for p in model.parameters():
        if p.is_floating_point():
            p.data.normal_(mean=0.0, std=0.02)
    return model


def _patch_dim(model) -> int:
    pe = model.vision_tower.patch_embed
    return pe.in_channels * pe.temporal_patch_size * pe.patch_size**2


class TestGlm53ComposeSequenceParallel(DistributedTestBase):
    """2 卡 VL 序列并行：视觉塔按 merge 块分片，特征 gather 回来后每个 rank 只写自己分片里的
    placeholder，logits 必须与非 SP 的对应片段一致。"""

    @pytest.mark.gpu
    def test_image_splice_under_sp_matches_non_sp(self, device="cuda"):
        self.create_pg(device)
        sp_size = self.world_size
        torch.manual_seed(0)
        model = _build_model()
        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)

        num_placeholders = 4  # merge_unit = MERGE**2 = 4
        raw_patches = num_placeholders * MERGE * MERGE
        grid_side = int(raw_patches**0.5)
        pixel_values = torch.randn(raw_patches, _patch_dim(model), device=device, dtype=torch.bfloat16)
        torch.distributed.broadcast(pixel_values, src=0)
        image_grid_thw = torch.tensor([[1, grid_side, grid_side]], device=device)

        input_ids = torch.randint(2, 200, (1, SEQ_LEN), device=device)
        torch.distributed.broadcast(input_ids, src=0)
        mm_type = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
        # 两段 placeholder 分别落在 rank0 / rank1 的分片里，强制走跨 rank 的特征切片逻辑。
        mm_type[0, 10:12] = 1
        mm_type[0, SEQ_LEN // 2 + 10 : SEQ_LEN // 2 + 12] = 1

        def _seq_ctx(sp_mesh):
            ctx = SequenceContext.from_input_ids((input_ids,), device=device)
            ctx.mm_token_type_ids = mm_type
            if sp_mesh is not None:
                ctx = ctx.split(sequence_parallel_mesh=sp_mesh)
            # 媒体张量保持全局，直到 splice 完成（与 qwen3_vl 的 VLM CP 约定一致）。
            ctx.pixel_values = pixel_values
            ctx.image_grid_thw = image_grid_thw
            return ctx

        reference = model(seq_ctx=_seq_ctx(None), loss_ctx=None).logits
        sp_mesh = init_data_mesh(device, sp_size)["sp"]
        sp_logits = model(seq_ctx=_seq_ctx(sp_mesh), loss_ctx=None).logits

        rank = sp_mesh.get_local_rank()
        local_len = SEQ_LEN // sp_size
        expected = reference[:, rank * local_len : (rank + 1) * local_len]
        torch.testing.assert_close(sp_logits, expected, rtol=2e-2, atol=2e-2)

    @property
    def world_size(self) -> int:
        return 2


@pytest.mark.gpu
class TestGlm53ComposeForward:
    def test_pure_text_forward(self):
        model = _build_model()
        input_ids = torch.randint(2, 200, (1, SEQ_LEN)).cuda()
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
        seq_ctx.mm_token_type_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long, device="cuda")
        out = model(seq_ctx=seq_ctx, loss_ctx=None)
        assert out.logits.shape == (1, SEQ_LEN, 200)
        assert torch.isfinite(out.logits).all()

    def test_image_splice_matches_placeholder_count(self):
        model = _build_model()
        num_placeholders = 4  # merge_unit = MERGE**2 = 4
        raw_patches = num_placeholders * MERGE * MERGE
        grid_side = int(raw_patches**0.5)
        pixel_values = torch.randn(raw_patches, _patch_dim(model), device="cuda", dtype=torch.bfloat16)
        image_grid_thw = torch.tensor([[1, grid_side, grid_side]], device="cuda")

        input_ids = torch.randint(2, 200, (1, SEQ_LEN)).cuda()
        mm_type = torch.zeros(1, SEQ_LEN, dtype=torch.long, device="cuda")
        mm_type[0, 10 : 10 + num_placeholders] = 1
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
        seq_ctx.mm_token_type_ids = mm_type
        seq_ctx.pixel_values = pixel_values
        seq_ctx.image_grid_thw = image_grid_thw

        out = model(seq_ctx=seq_ctx, loss_ctx=None)
        assert torch.isfinite(out.logits).all()

    def test_video_splice_uses_type_2_and_flattens_grid(self):
        model = _build_model()
        num_placeholders = 4
        raw_patches_per_tubelet = num_placeholders * MERGE * MERGE
        grid_side = int(raw_patches_per_tubelet**0.5)
        grid_t = 2
        pixel_values_videos = torch.randn(
            raw_patches_per_tubelet * grid_t, _patch_dim(model), device="cuda", dtype=torch.bfloat16
        )
        video_grid_thw = torch.tensor([[grid_t, grid_side, grid_side]], device="cuda")

        input_ids = torch.randint(2, 200, (1, SEQ_LEN)).cuda()
        mm_type = torch.zeros(1, SEQ_LEN, dtype=torch.long, device="cuda")
        mm_type[0, 10 : 10 + num_placeholders * grid_t] = 2
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
        seq_ctx.mm_token_type_ids = mm_type
        seq_ctx.pixel_values_videos = pixel_values_videos
        seq_ctx.video_grid_thw = video_grid_thw

        out = model(seq_ctx=seq_ctx, loss_ctx=None)
        assert torch.isfinite(out.logits).all()

    def test_placeholder_mismatch_raises_instead_of_silently_continuing(self):
        # design doc §16.2: unlike Qwen3-VL's compose path, a mismatch must never be caught and
        # skipped -- it must raise so a corrupted sample never silently enters training.
        model = _build_model()
        num_placeholders = 4
        raw_patches = num_placeholders * MERGE * MERGE
        grid_side = int(raw_patches**0.5)
        pixel_values = torch.randn(raw_patches, _patch_dim(model), device="cuda", dtype=torch.bfloat16)
        image_grid_thw = torch.tensor([[1, grid_side, grid_side]], device="cuda")

        input_ids = torch.randint(2, 200, (1, SEQ_LEN)).cuda()
        mm_type = torch.zeros(1, SEQ_LEN, dtype=torch.long, device="cuda")
        mm_type[0, 10 : 10 + num_placeholders - 1] = 1  # one short
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
        seq_ctx.mm_token_type_ids = mm_type
        seq_ctx.pixel_values = pixel_values
        seq_ctx.image_grid_thw = image_grid_thw

        with pytest.raises(ValueError, match="placeholder count"):
            model(seq_ctx=seq_ctx, loss_ctx=None)

    def test_mixed_image_and_video_in_one_sample_is_rejected(self):
        model = _build_model()
        input_ids = torch.randint(2, 200, (1, SEQ_LEN)).cuda()
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
        seq_ctx.mm_token_type_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long, device="cuda")
        seq_ctx.pixel_values = torch.randn(4, _patch_dim(model), device="cuda", dtype=torch.bfloat16)
        seq_ctx.image_grid_thw = torch.tensor([[1, 2, 2]], device="cuda")
        seq_ctx.pixel_values_videos = torch.randn(4, _patch_dim(model), device="cuda", dtype=torch.bfloat16)
        seq_ctx.video_grid_thw = torch.tensor([[1, 2, 2]], device="cuda")

        with pytest.raises(AssertionError, match="image-only or video-only"):
            model(seq_ctx=seq_ctx, loss_ctx=None)
