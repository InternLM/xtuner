import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer
from transformers.models.glm_moe_dsa import GlmMoeDsaConfig as HFGlmMoeDsaConfig

from xtuner._testing import load_glm52_hf_oracle_model
from xtuner.v1.model import Glm52MoEConfig, get_model_config, get_model_config_from_hf
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.module.attention import DSAMLAConfig, MLAConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.utils.loader import HFCheckpointLoader


GLM5_2_MOE_PATH = os.environ["GLM5_2_MOE_PATH"]
GLM5_2_TINY_MOE_PATH = Path(
    os.environ.get("GLM5_2_TINY_MOE_PATH", "/mnt/shared-storage-user/zhaopenghao/slime0701/ckpts/GLM-5.2-tiny-4L")
)


def _tiny_glm52_config() -> Glm52MoEConfig:
    return Glm52MoEConfig(
        vocab_size=32,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        hf_eos_token_id=[1, 2],
        num_hidden_layers=3,
        first_k_dense_replace=1,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        attention=DSAMLAConfig(
            num_attention_heads=2,
            head_dim=4,
            kv_lora_rank=4,
            q_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            index_topk=4,
            index_head_dim=4,
            index_n_heads=2,
            indexer_types=["full", "shared", "shared"],
        ),
        hf_head_dim=4,
        qk_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        router=NoAuxRouterConfig(
            n_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            router_scaling_factor=2.5,
        ),
        mlp_layer_types=["dense", "sparse", "sparse"],
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        indexer_types=["full", "shared", "shared"],
        num_nextn_predict_layers=1,
        compile_cfg=False,
    )


def test_glm52_config_from_hf_preserves_native_v1_fields():
    hf_config = HFGlmMoeDsaConfig.from_pretrained(GLM5_2_MOE_PATH)

    config = get_model_config_from_hf(Path(GLM5_2_MOE_PATH))

    assert isinstance(config, Glm52MoEConfig)
    assert isinstance(get_model_config("glm-5.2"), Glm52MoEConfig)
    assert config.model_type == "glm_moe_dsa"
    assert config.vocab_size == hf_config.vocab_size
    assert config.pad_token_id == hf_config.pad_token_id
    assert config.eos_token_id == hf_config.eos_token_id[0]
    assert config.hf_eos_token_id == hf_config.eos_token_id
    assert config.max_position_embeddings == hf_config.max_position_embeddings
    assert config.num_hidden_layers == hf_config.num_hidden_layers
    assert config.hidden_size == hf_config.hidden_size
    assert config.intermediate_size == hf_config.intermediate_size
    assert config.moe_intermediate_size == hf_config.moe_intermediate_size
    assert config.rms_norm_eps == hf_config.rms_norm_eps
    assert config.tie_word_embeddings == hf_config.tie_word_embeddings
    assert config.rope_parameters["rope_theta"] == hf_config.rope_parameters["rope_theta"]
    assert config.rope_parameters["rope_type"] == hf_config.rope_parameters["rope_type"]

    assert isinstance(config.attention, DSAMLAConfig)
    assert isinstance(config.attention, MLAConfig)
    assert config.attention.num_attention_heads == hf_config.num_attention_heads
    assert config.num_key_value_heads == hf_config.num_key_value_heads
    assert config.attention.head_dim == hf_config.qk_rope_head_dim
    assert config.attention.kv_lora_rank == hf_config.kv_lora_rank
    assert config.attention.q_lora_rank == hf_config.q_lora_rank
    assert config.attention.qk_nope_head_dim == hf_config.qk_nope_head_dim
    assert config.attention.qk_rope_head_dim == hf_config.qk_rope_head_dim
    assert config.attention.v_head_dim == hf_config.v_head_dim
    assert config.attention.dropout == hf_config.attention_dropout
    assert config.attention.qkv_bias == hf_config.attention_bias
    assert config.attention.o_bias == hf_config.attention_bias

    assert config.first_k_dense_replace == hf_config.first_k_dense_replace
    assert config.mlp_layer_types == hf_config.mlp_layer_types
    assert config.n_routed_experts == hf_config.n_routed_experts
    assert config.n_shared_experts == hf_config.n_shared_experts
    assert config.num_experts_per_tok == hf_config.num_experts_per_tok

    assert isinstance(config.router, NoAuxRouterConfig)
    assert config.router.scoring_func == hf_config.scoring_func
    assert config.router.n_group == hf_config.n_group
    assert config.router.topk_group == hf_config.topk_group
    assert config.router.norm_topk_prob == hf_config.norm_topk_prob
    assert config.router.router_scaling_factor == hf_config.routed_scaling_factor

    assert config.index_topk == hf_config.index_topk
    assert config.index_head_dim == hf_config.index_head_dim
    assert config.index_n_heads == hf_config.index_n_heads
    assert config.index_topk_freq == hf_config.index_topk_freq
    assert config.index_skip_topk_offset == hf_config.index_skip_topk_offset
    assert config.index_share_for_mtp_iteration == hf_config.index_share_for_mtp_iteration
    assert config.indexer_rope_interleave == hf_config.indexer_rope_interleave
    assert config.indexer_types == hf_config.indexer_types

    assert config.mtp_config is None
    assert config.model_dump()["num_key_value_heads"] == hf_config.num_key_value_heads


def test_glm52_key_mapping_covers_native_shell_and_dsa_indexer():
    model = _tiny_glm52_config().build()
    model_keys = set(model.state_dict())

    assert model.to_hf_key_list("embed_tokens.weight") == ["model.embed_tokens.weight"]
    assert model.to_hf_key_list("norm.weight") == ["model.norm.weight"]
    assert model.to_hf_key_list("lm_head.weight") == ["lm_head.weight"]
    assert model.to_hf_key_list("layers.0.self_attn.q_a_proj.weight") == [
        "model.layers.0.self_attn.q_a_proj.weight"
    ]
    assert model.to_hf_key_list("layers.0.mlp.gate_proj.weight") == ["model.layers.0.mlp.gate_proj.weight"]
    assert model.to_hf_key_list("layers.1.gate.weight") == ["model.layers.1.mlp.gate.weight"]
    assert model.to_hf_key_list("layers.1.gate.router.e_score_correction_bias") == [
        "model.layers.1.mlp.gate.e_score_correction_bias"
    ]
    assert model.to_hf_key_list("layers.1.experts.fused_w1w3.weight") == [
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
        "model.layers.1.mlp.experts.1.gate_proj.weight",
        "model.layers.1.mlp.experts.1.up_proj.weight",
        "model.layers.1.mlp.experts.2.gate_proj.weight",
        "model.layers.1.mlp.experts.2.up_proj.weight",
        "model.layers.1.mlp.experts.3.gate_proj.weight",
        "model.layers.1.mlp.experts.3.up_proj.weight",
    ]
    assert model.to_hf_key_list("layers.1.experts.fused_w2.weight") == [
        "model.layers.1.mlp.experts.0.down_proj.weight",
        "model.layers.1.mlp.experts.1.down_proj.weight",
        "model.layers.1.mlp.experts.2.down_proj.weight",
        "model.layers.1.mlp.experts.3.down_proj.weight",
    ]
    assert model.to_hf_key_list("layers.1.shared_experts.gate_proj.weight") == [
        "model.layers.1.mlp.shared_experts.gate_proj.weight"
    ]
    assert model.to_hf_key_list("layers.1.self_attn.indexer.wq_b.weight") == [
        "model.layers.1.self_attn.indexer.wq_b.weight"
    ]
    assert "layers.0.self_attn.indexer.wq_b.weight" in model_keys
    assert "layers.1.self_attn.indexer.wq_b.weight" not in model_keys


def test_tiny_glm52_hf_checkpoint_load_reports_loaded_missing_and_ignored_keys():
    config = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
    config.compile_cfg = False
    model = config.build()
    checkpoint_loader = HFCheckpointLoader(str(GLM5_2_TINY_MOE_PATH))
    expected_hf_keys = set()
    for model_key in model.state_dict():
        expected_hf_keys.update(model.to_hf_key_list(model_key))

    loaded_keys, unloaded_keys, missing_keys = model.from_hf(GLM5_2_TINY_MOE_PATH, strict=False)
    ignored_hf_keys = set(checkpoint_loader.weight_map) - expected_hf_keys

    assert config.num_hidden_layers == 4
    assert config.first_k_dense_replace == 3
    assert "layers.0.self_attn.indexer.wq_b.weight" in loaded_keys
    assert "layers.1.self_attn.indexer.wk.weight" in loaded_keys
    assert "layers.2.self_attn.indexer.weights_proj.weight" in loaded_keys
    assert "layers.3.self_attn.indexer.wq_b.weight" not in model.state_dict()
    assert "layers.3.experts.fused_w1w3.weight" in loaded_keys
    assert "layers.3.experts.fused_w2.weight" in loaded_keys
    assert unloaded_keys == set()
    assert missing_keys == set()
    assert ignored_hf_keys == set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tiny GLM-5.2 numeric oracle requires CUDA")
def test_tiny_glm52_native_forward_matches_hf_numeric_oracle():
    tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)
    input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")

    hf_model = load_glm52_hf_oracle_model(GLM5_2_TINY_MOE_PATH)
    try:
        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits.detach().cpu()
    finally:
        del hf_model
        torch.cuda.empty_cache()

    with torch.device("meta"):
        config = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
        config.compile_cfg = False
        config.dispatcher = None
        config.ep_size = 1
        model = config.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

    try:
        model.from_hf(GLM5_2_TINY_MOE_PATH)
        model.eval()

        seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))

        with torch.no_grad():
            output = model(seq_ctx=seq_ctx, loss_ctx=None)

        native_logits = output["logits"].detach().to(hf_logits.dtype).cpu()
        # HF expands K/V and runs dense attention, while native uses absorbed MLA.
        # They are mathematically equivalent, but bf16 accumulates in different
        # matmul orders, so keep torch.testing's bf16 rtol and allow the observed
        # logit-scale absolute drift from that accumulation difference.
        torch.testing.assert_close(native_logits, hf_logits, rtol=1.6e-2, atol=1e-1)
        assert seq_ctx.dsa_topk_indices is not None
        assert sorted(seq_ctx.dsa_topk_indices) == [0, 1, 2]

        seq_len = input_ids.numel()
        for topk_indices in seq_ctx.dsa_topk_indices.values():
            assert topk_indices.shape == (seq_len, 1, seq_len)
            assert topk_indices.dtype == torch.int64
            assert topk_indices.device.type == "cuda"
            assert (topk_indices == -1).any()
    finally:
        del model
        torch.cuda.empty_cache()
