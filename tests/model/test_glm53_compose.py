"""GLM-5.3-Flash compose model tests, see doc/xtuner_glm5p3flash_design.md F6.

Tiny synthetic config (mirrors test_glm53_text_moe.py / test_glm53_vision.py fixtures). GPU
required: KDA's Triton kernel has no CPU backend.
"""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.compose.glm53 import Glm53BaseConfig, Glm53ProjectorConfig, Glm53VisionConfig
from xtuner.v1.model.moe.glm53.glm53 import Glm53TextMoEConfig
from xtuner.v1.model.moe.glm53.nope_dsa_mla import NoPEDSAMLAConfig
from xtuner.v1.module.attention.kda import KDAConfig
from xtuner.v1.module.decoder_layer.mhc import MHCConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig


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


@pytest.fixture
def two_rank_mesh():
    """A real 2-rank ``DeviceMesh`` over the fake process group. The guard under test reads only
    ``mesh.size()`` and raises before any collective, so no second process is needed -- and a
    real mesh keeps this a behaviour test rather than a test against a mock."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
    try:
        yield init_device_mesh("cpu", (2,))
    finally:
        dist.destroy_process_group()


def _patch_dim(model) -> int:
    pe = model.vision_tower.patch_embed
    return pe.in_channels * pe.temporal_patch_size * pe.patch_size**2


class TestGlm53ComposeSequenceParallelGuard:
    """LLM 序列并行尚未支持时的显式护栏。"""

    @pytest.mark.gpu
    def test_forward_rejects_a_sequence_parallel_context(self, two_rank_mesh):
        # splice 用全局 mm_token_type_ids 索引 inputs_embeds；SP 下 embeds 被分片而 mask 不是，
        # 数量对不上时报的是"视觉特征数不符"，会把人指向错误方向。这里必须按名字拒绝。
        model = _build_model()
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, SEQ_LEN, dtype=torch.long),), device="cuda")
        seq_ctx.sequence_parallel_mesh = two_rank_mesh

        with pytest.raises(AssertionError, match="sequence parallel"):
            model(seq_ctx=seq_ctx, loss_ctx=None)


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
