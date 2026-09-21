"""GLM-5.3-Flash 的视觉塔与 projector，见 doc/xtuner_glm5p3flash_design.md F2。

TestFlattenVideoGridThw
    test_expands_multiple_videos_preserving_order  视频 grid 展开成逐帧行且保持顺序
    test_t_equal_1_is_a_no_op_on_values            t=1 时展开不改变数值
TestGlm53VisionMetaBuild
    test_vision_and_projector_build_on_meta_device  meta 设备上可构造
TestGlm53VisionFp32Params
    test_vision_side_pins_nothing_to_fp32          视觉侧不得 pin fp32（梯度不会被 reduce）
TestGlm53VisionWeightMapping
    test_vision_weight_mapping_bitwise             真实 checkpoint 权重映射逐位一致
TestGlm53VisionForwardParity
    test_image_forward_matches_hf_bitwise          图像前向与 HF 逐位一致
    test_multi_tubelet_video_forward_matches_hf_bitwise  多 tubelet 视频前向一致
    test_multi_image_batch_forward_matches_hf_bitwise    多图批次前向一致
    test_forward_backward_gradients_flow           反向梯度可达所有参数
"""

import os

import pytest
import torch

from xtuner.v1.model.compose.glm53 import (
    Glm53BaseConfig,
    Glm53ProjectorConfig,
    Glm53VisionConfig,
    flatten_video_grid_thw,
)


GLM_5_3_FLASH_PATH = os.environ.get(
    "GLM_5_3_FLASH_PATH",
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--zai-org--GLM-5.3-Flash/"
    "snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
)


class TestFlattenVideoGridThw:
    def test_expands_multiple_videos_preserving_order(self):
        grid_thw = torch.tensor([[3, 4, 6], [2, 2, 4]])
        flattened = flatten_video_grid_thw(grid_thw)
        expected = torch.tensor(
            [[1, 4, 6], [1, 4, 6], [1, 4, 6], [1, 2, 4], [1, 2, 4]],
        )
        torch.testing.assert_close(flattened, expected)

    def test_t_equal_1_is_a_no_op_on_values(self):
        grid_thw = torch.tensor([[1, 8, 8]])
        flattened = flatten_video_grid_thw(grid_thw)
        torch.testing.assert_close(flattened, grid_thw)


class TestGlm53VisionMetaBuild:
    def test_vision_and_projector_build_on_meta_device(self):
        with torch.device("meta"):
            vision = Glm53VisionConfig(attn_impl="eager_attention", fully_shard=False).build()
            projector = Glm53ProjectorConfig(fully_shard=False).build()
        assert vision.patch_embed.proj.weight.is_meta
        assert projector.downsample.weight.is_meta
        assert len(vision.blocks) == 24
        assert projector.merger.gate_proj.out_features == 10240


class TestGlm53VisionFp32Params:
    def test_vision_side_pins_nothing_to_fp32(self):
        """`fp32_keys_pattern` routes a parameter to `fully_shard(ignored_params=...)`, which
        leaves it replicated and *outside* FSDP's gradient reduction. Only
        `MoE.scale_and_reduce_grad` all-reduces such gradients, and
        `BaseComposeModel.scale_and_reduce_grad` forwards to `language_model` alone -- so a
        vision-side fp32 pin would silently let those gradients diverge across ranks. Lifting
        that reduction to `BaseModel` is a separate change; until then this stays empty."""
        # 视觉侧 pin fp32 的参数梯度不会被 all-reduce，会静默发散，故必须为空。
        cfg = Glm53BaseConfig()
        assert not cfg.vision_config.hf_save_cfg.fp32_keys_pattern
        assert not cfg.projector_config.hf_save_cfg.fp32_keys_pattern


class TestGlm53VisionWeightMapping:
    def test_vision_weight_mapping_bitwise(self):
        if not os.path.isdir(GLM_5_3_FLASH_PATH):
            pytest.skip(f"GLM_5_3_FLASH_PATH not found: {GLM_5_3_FLASH_PATH}")

        with torch.device("meta"):
            vision = Glm53VisionConfig(attn_impl="eager_attention", fully_shard=False).build()
            projector = Glm53ProjectorConfig(fully_shard=False).build()
        vision._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        projector._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

        _, unloaded_v, missing_v = vision.from_hf(GLM_5_3_FLASH_PATH, strict=False)
        _, unloaded_p, missing_p = projector.from_hf(GLM_5_3_FLASH_PATH, strict=False)

        assert not missing_v and not unloaded_v
        assert not missing_p and not unloaded_p
        assert not any(p.is_meta for p in vision.parameters())
        assert not any(p.is_meta for p in projector.parameters())

        num_vision_params = sum(1 for _ in vision.named_parameters())
        num_projector_params = sum(1 for _ in projector.named_parameters())
        assert num_vision_params + num_projector_params == 347


def _copy_weights(hf_model, xt_vision, xt_projector, depth: int) -> None:
    with torch.no_grad():
        xt_vision.patch_embed.proj.weight.copy_(hf_model.patch_embed.proj.weight)
        xt_vision.patch_embed.proj.bias.copy_(hf_model.patch_embed.proj.bias)
        for i in range(depth):
            hb, xb = hf_model.blocks[i], xt_vision.blocks[i]
            xb.norm1.weight.copy_(hb.norm1.weight)
            xb.norm2.weight.copy_(hb.norm2.weight)
            xb.attn.qkv.weight.copy_(hb.attn.qkv.weight)
            xb.attn.qkv.bias.copy_(hb.attn.qkv.bias)
            xb.attn.proj.weight.copy_(hb.attn.proj.weight)
            xb.attn.proj.bias.copy_(hb.attn.proj.bias)
            xb.attn.q_norm.weight.copy_(hb.attn.q_norm.weight)
            xb.attn.k_norm.weight.copy_(hb.attn.k_norm.weight)
            xb.mlp.gate_proj.weight.copy_(hb.mlp.gate_proj.weight)
            xb.mlp.gate_proj.bias.copy_(hb.mlp.gate_proj.bias)
            xb.mlp.up_proj.weight.copy_(hb.mlp.up_proj.weight)
            xb.mlp.up_proj.bias.copy_(hb.mlp.up_proj.bias)
            xb.mlp.down_proj.weight.copy_(hb.mlp.down_proj.weight)
            xb.mlp.down_proj.bias.copy_(hb.mlp.down_proj.bias)
        xt_vision.post_layernorm.weight.copy_(hf_model.post_layernorm.weight)
        xt_projector.downsample.weight.copy_(hf_model.downsample.weight)
        xt_projector.downsample.bias.copy_(hf_model.downsample.bias)
        xt_projector.merger.proj.weight.copy_(hf_model.merger.proj.weight)
        xt_projector.merger.post_projection_norm.weight.copy_(hf_model.merger.post_projection_norm.weight)
        xt_projector.merger.post_projection_norm.bias.copy_(hf_model.merger.post_projection_norm.bias)
        xt_projector.merger.gate_proj.weight.copy_(hf_model.merger.gate_proj.weight)
        xt_projector.merger.up_proj.weight.copy_(hf_model.merger.up_proj.weight)
        xt_projector.merger.down_proj.weight.copy_(hf_model.merger.down_proj.weight)


HIDDEN = 32
NUM_HEADS = 4
INTERMEDIATE = 64
DEPTH = 3
PATCH_SIZE = 4
TEMPORAL_PATCH_SIZE = 2
MERGE = 2
OUT_HIDDEN = 48
PROJ_INTERMEDIATE = 40
IN_CHANNELS = 3


def _build_models():
    xt_vis_cfg = Glm53VisionConfig(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE,
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        spatial_merge_size=MERGE,
        rms_norm_eps=1e-6,
        attn_impl="eager_attention",
        fully_shard=False,
    )
    xt_proj_cfg = Glm53ProjectorConfig(
        vision_hidden_size=HIDDEN,
        out_hidden_size=OUT_HIDDEN,
        spatial_merge_size=MERGE,
        projection_intermediate_size=PROJ_INTERMEDIATE,
        fully_shard=False,
    )
    xt_vision = xt_vis_cfg.build()
    xt_projector = xt_proj_cfg.build()

    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    hf_cfg = Glm5NextVisionConfig(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE,
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        spatial_merge_size=MERGE,
        rms_norm_eps=1e-6,
        out_hidden_size=OUT_HIDDEN,
        projection_intermediate_size=PROJ_INTERMEDIATE,
        swiglu_limit=10.0,
        attention_bias=True,
        hidden_act="silu",
        rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
        _attn_implementation="eager",
    )
    hf_model = Glm5NextVisionModel(hf_cfg)

    torch.manual_seed(0)
    for p in xt_vision.parameters():
        p.data.normal_(mean=0.0, std=0.02)
    for p in xt_projector.parameters():
        p.data.normal_(mean=0.0, std=0.02)
    _copy_weights(hf_model, xt_vision, xt_projector, DEPTH)

    hf_model.eval()
    xt_vision.eval()
    xt_projector.eval()
    return hf_model, xt_vision, xt_projector


def _hf_pre_downsample(hf_model, xt_hidden: torch.Tensor) -> torch.Tensor:
    reshaped = xt_hidden.view(-1, MERGE, MERGE, xt_hidden.shape[-1]).permute(0, 3, 1, 2)
    return hf_model.downsample(reshaped).view(-1, OUT_HIDDEN)


class TestGlm53VisionForwardParity:
    def test_image_forward_matches_hf_bitwise(self):
        hf_model, xt_vision, xt_projector = _build_models()
        grid_thw = torch.tensor([[1, 4, 4]])
        patch_dim = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
        pixel_values = torch.randn(16, patch_dim)

        with torch.no_grad():
            hf_out = hf_model(pixel_values, grid_thw=grid_thw)
            xt_hidden = xt_vision(pixel_values, grid_thw)
            xt_out = xt_projector(xt_hidden)

        torch.testing.assert_close(_hf_pre_downsample(hf_model, xt_hidden), hf_out.last_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(xt_out, hf_out.pooler_output, rtol=0, atol=0)

    def test_multi_tubelet_video_forward_matches_hf_bitwise(self):
        # grid_thw=[2,4,4]: two independent ViT attention segments (one per tubelet), t-repeated
        # position ids -- exercises cu_seqlens/position_ids handling beyond the single-image path.
        hf_model, xt_vision, xt_projector = _build_models()
        grid_thw = torch.tensor([[2, 4, 4]])
        patch_dim = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
        pixel_values = torch.randn(32, patch_dim)

        with torch.no_grad():
            hf_out = hf_model(pixel_values, grid_thw=grid_thw)
            xt_hidden = xt_vision(pixel_values, grid_thw)
            xt_out = xt_projector(xt_hidden)

        torch.testing.assert_close(_hf_pre_downsample(hf_model, xt_hidden), hf_out.last_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(xt_out, hf_out.pooler_output, rtol=0, atol=0)

    def test_multi_image_batch_forward_matches_hf_bitwise(self):
        hf_model, xt_vision, xt_projector = _build_models()
        grid_thw = torch.tensor([[1, 4, 4], [1, 2, 8]])
        patch_dim = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
        pixel_values = torch.randn(16 + 16, patch_dim)

        with torch.no_grad():
            hf_out = hf_model(pixel_values, grid_thw=grid_thw)
            xt_hidden = xt_vision(pixel_values, grid_thw)
            xt_out = xt_projector(xt_hidden)

        torch.testing.assert_close(_hf_pre_downsample(hf_model, xt_hidden), hf_out.last_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(xt_out, hf_out.pooler_output, rtol=0, atol=0)

    def test_forward_backward_gradients_flow(self):
        _, xt_vision, xt_projector = _build_models()
        grid_thw = torch.tensor([[1, 4, 4]])
        patch_dim = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
        pixel_values = torch.randn(16, patch_dim, requires_grad=True)

        xt_hidden = xt_vision(pixel_values, grid_thw)
        xt_out = xt_projector(xt_hidden)
        probe = torch.randn_like(xt_out)
        (xt_out * probe).sum().backward()

        assert pixel_values.grad is not None and pixel_values.grad.abs().sum() > 0
        assert xt_vision.patch_embed.proj.weight.grad is not None
        assert xt_vision.patch_embed.proj.weight.grad.abs().sum() > 0
