import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import parametrize
import pytest
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import AutoTokenizer
from transformers.models.glm_moe_dsa import GlmMoeDsaConfig as HFGlmMoeDsaConfig
from xtuner._testing import DeterministicDDPTestCase, load_glm52_hf_oracle_model
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Glm52MoEConfig, get_model_config, get_model_config_from_hf
from xtuner.v1.module.attention import DSAMLAConfig, MLAConfig
from xtuner.v1.module.attention.dsa_topk_sharing import configure_dsa_mtp_iteration_lifecycle
from xtuner.v1.module.mtp import MTPBlock, MTPConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.utils.compile import is_compiled_function
from xtuner.v1.utils.loader import HFCheckpointLoader
from xtuner.v1.utils.test_utils import init_data_mesh


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
            indexer_types=["full", "shared", "shared", "full"],
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
        num_nextn_predict_layers=1,
        compile_cfg=False,
    )


def _build_lm_loss_ctx(input_ids: torch.Tensor, loss_mode: str = "eager"):
    shift_input_ids = input_ids[:, :-1]
    shifted_labels = input_ids[:, 1:]
    seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to("cuda"),))

    loss_cfg = CELossConfig(mode=loss_mode)
    loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
    assert loss_ctx is not None
    loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
    return seq_ctx, loss_ctx


def _write_single_shard_hf_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir()
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, path / shard)
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": dict.fromkeys(tensors, shard),
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))


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

    assert config.attention.index_topk == hf_config.index_topk
    assert config.attention.index_head_dim == hf_config.index_head_dim
    assert config.attention.index_n_heads == hf_config.index_n_heads
    assert config.attention.index_topk_freq == hf_config.index_topk_freq
    assert config.attention.index_skip_topk_offset == hf_config.index_skip_topk_offset
    assert config.index_share_for_mtp_iteration == hf_config.index_share_for_mtp_iteration
    assert config.attention.indexer_rope_interleave == hf_config.indexer_rope_interleave
    assert config.attention.indexer_types == hf_config.indexer_types + ["full"]

    assert isinstance(config.mtp_config, MTPConfig)
    assert config.mtp_config.num_layers == hf_config.num_nextn_predict_layers
    assert config.mtp_config.share_weights
    assert config.num_nextn_predict_layers == hf_config.num_nextn_predict_layers
    assert config.model_dump()["num_key_value_heads"] == hf_config.num_key_value_heads
    assert config.hf_config.indexer_types == hf_config.indexer_types


def test_glm52_config_from_hf_preserves_cropped_30b_mtp(tmp_path):
    hf_config = HFGlmMoeDsaConfig.from_pretrained(GLM5_2_MOE_PATH)
    hf_config.num_hidden_layers = 5
    hf_config.first_k_dense_replace = 3
    hf_config.mlp_layer_types = hf_config.mlp_layer_types[:5]
    hf_config.indexer_types = hf_config.indexer_types[:5]
    hf_config.num_nextn_predict_layers = 1
    hf_config.save_pretrained(tmp_path)

    config = get_model_config_from_hf(tmp_path)

    assert isinstance(config.mtp_config, MTPConfig)
    assert config.mtp_config.num_layers == 1
    assert config.mtp_config.share_weights
    assert config.num_nextn_predict_layers == 1
    assert config.attention.indexer_types == hf_config.indexer_types[:5] + ["full"]
    assert config.layers_type == [
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
    ]


def test_glm52_key_mapping_covers_native_shell_and_dsa_indexer():
    model = _tiny_glm52_config().build()
    model_keys = set(model.state_dict())

    assert model.to_hf_key_list("embed_tokens.weight") == ["model.embed_tokens.weight"]
    assert model.to_hf_key_list("norm.weight") == ["model.norm.weight"]
    assert model.to_hf_key_list("lm_head.weight") == ["lm_head.weight"]
    assert model.to_hf_key_list("layers.0.self_attn.q_a_proj.weight") == ["model.layers.0.self_attn.q_a_proj.weight"]
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


def test_glm52_key_mapping_covers_mtp_layer_as_hf_final_layer():
    config = _tiny_glm52_config()
    config.mtp_config = MTPConfig(num_layers=1)
    model = config.build()

    assert model.to_hf_key_list("mtp_block.layers.0.enorm.weight") == ["model.layers.3.enorm.weight"]
    assert model.to_hf_key_list("mtp_block.layers.0.hnorm.weight") == ["model.layers.3.hnorm.weight"]
    assert model.to_hf_key_list("mtp_block.layers.0.eh_proj.weight") == ["model.layers.3.eh_proj.weight"]
    assert model.to_hf_key_list("mtp_block.layers.0.final_layernorm.weight") == [
        "model.layers.3.shared_head.norm.weight"
    ]
    assert model.to_hf_key_list("mtp_block.layers.0.decoder_layer.self_attn.q_a_proj.weight") == [
        "model.layers.3.self_attn.q_a_proj.weight"
    ]
    assert model.to_hf_key_list("mtp_block.layers.0.decoder_layer.self_attn.indexer.wq_b.weight") == [
        "model.layers.3.self_attn.indexer.wq_b.weight"
    ]
    assert model.to_hf_key_list("mtp_block.layers.0.decoder_layer.gate.router.e_score_correction_bias") == [
        "model.layers.3.mlp.gate.e_score_correction_bias"
    ]
    assert model.to_hf_key_list("mtp_block.layers.0.decoder_layer.experts.fused_w1w3.weight") == [
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.experts.0.up_proj.weight",
        "model.layers.3.mlp.experts.1.gate_proj.weight",
        "model.layers.3.mlp.experts.1.up_proj.weight",
        "model.layers.3.mlp.experts.2.gate_proj.weight",
        "model.layers.3.mlp.experts.2.up_proj.weight",
        "model.layers.3.mlp.experts.3.gate_proj.weight",
        "model.layers.3.mlp.experts.3.up_proj.weight",
    ]
    assert model.to_hf_key_list("mtp_block.layers.0.decoder_layer.experts.fused_w2.weight") == [
        "model.layers.3.mlp.experts.0.down_proj.weight",
        "model.layers.3.mlp.experts.1.down_proj.weight",
        "model.layers.3.mlp.experts.2.down_proj.weight",
        "model.layers.3.mlp.experts.3.down_proj.weight",
    ]
    assert "mtp_block.layers.0.decoder_layer.self_attn.indexer.wq_b.weight" in model.state_dict()


def test_glm52_hf_checkpoint_loads_mtp_final_layer_keys(tmp_path):
    config = _tiny_glm52_config()
    config.mtp_config = MTPConfig(num_layers=1)
    model = config.build()

    state_dict = model.state_dict()
    mtp_model_keys = {
        "mtp_block.layers.0.enorm.weight": 1.0,
        "mtp_block.layers.0.hnorm.weight": 2.0,
        "mtp_block.layers.0.eh_proj.weight": 3.0,
        "mtp_block.layers.0.decoder_layer.self_attn.q_a_proj.weight": 4.0,
        "mtp_block.layers.0.decoder_layer.self_attn.indexer.wq_b.weight": 5.0,
        "mtp_block.layers.0.decoder_layer.self_attn.indexer.wk.weight": 6.0,
        "mtp_block.layers.0.decoder_layer.self_attn.indexer.k_norm.weight": 7.0,
        "mtp_block.layers.0.decoder_layer.self_attn.indexer.k_norm.bias": 8.0,
        "mtp_block.layers.0.decoder_layer.self_attn.indexer.weights_proj.weight": 9.0,
        "mtp_block.layers.0.decoder_layer.gate.router.e_score_correction_bias": 10.0,
        "mtp_block.layers.0.final_layernorm.weight": 11.0,
    }
    tensors = {}
    expected_tensors = {}
    for model_key, value in mtp_model_keys.items():
        hf_keys = model.to_hf_key_list(model_key)
        assert len(hf_keys) == 1
        expected = torch.full_like(state_dict[model_key], value)
        expected_tensors[model_key] = expected
        tensors[hf_keys[0]] = expected

    checkpoint_dir = tmp_path / "fake-glm52-mtp"
    _write_single_shard_hf_checkpoint(checkpoint_dir, tensors)

    loaded_keys, unloaded_keys, missing_keys = model.from_hf(checkpoint_dir, strict=False)

    assert set(mtp_model_keys).issubset(loaded_keys)
    assert set(mtp_model_keys).isdisjoint(unloaded_keys)
    for model_key, expected in expected_tensors.items():
        assert missing_keys.isdisjoint(model.to_hf_key_list(model_key))
        torch.testing.assert_close(model.state_dict()[model_key], expected)


def test_glm52_update_bias_covers_main_and_mtp_moe_gates():
    config = _tiny_glm52_config()
    config.mtp_config = MTPConfig(num_layers=1)
    model = config.build()
    assert model.mtp_block is not None

    main_router_1 = model.layers["1"].gate.router
    main_router_2 = model.layers["2"].gate.router
    mtp_router = model.mtp_block.layers[0].decoder_layer.gate.router
    biases = [
        main_router_1.e_score_correction_bias,
        main_router_2.e_score_correction_bias,
        mtp_router.e_score_correction_bias,
    ]
    for bias in biases:
        bias.zero_()

    device = biases[0].device
    expert_counts = torch.tensor(
        [
            [2, 0, 1, 1],
            [0, 2, 1, 1],
            [1, 1, 2, 0],
        ],
        device=device,
    )
    model.update_bias(expert_counts, expert_counts.float().mean(dim=1))

    update_speed = config.router.router_bias_update_speed
    expected_biases = torch.tensor(
        [
            [-update_speed, update_speed, 0.0, 0.0],
            [update_speed, -update_speed, 0.0, 0.0],
            [0.0, 0.0, -update_speed, update_speed],
        ],
        device=device,
    )
    for bias, expected in zip(biases, expected_biases):
        torch.testing.assert_close(bias, expected)


def test_glm52_mtp_layer_extends_dsa_topk_release_plan():
    config = _tiny_glm52_config()
    config.mtp_config = MTPConfig(num_layers=1)

    model = config.build()

    main_attn = model.layers["2"].self_attn
    mtp_attn = model.mtp_block.layers[0].decoder_layer.self_attn  # type: ignore[union-attr]
    assert main_attn.dsa_topk_last_use[0] == 2
    assert mtp_attn.dsa_topk_last_use[3] == 3
    assert main_attn.dsa_topk_recompute_release[0] == 0
    assert mtp_attn.dsa_topk_recompute_release[3] == 3
    assert mtp_attn.source_layer_idx == 3
    assert hasattr(mtp_attn, "indexer")


@pytest.mark.parametrize("indexer_types", [None, ["full", "full", "full"]])
def test_glm52_mtp_physical_layer_is_full_independent_of_main_indexer_schedule(indexer_types):
    config = _tiny_glm52_config()
    config.attention.indexer_types = indexer_types
    config.attention.index_skip_topk_offset = 3
    config.attention.index_topk_freq = 4
    config.mtp_config = MTPConfig(
        num_layers=3,
        share_weights=True,
    )

    model = config.build()

    mtp_attn = model.mtp_block.layers[0].decoder_layer.self_attn  # type: ignore[union-attr]
    assert config.attention.indexer_types == ["full", "full", "full", "full"]
    assert mtp_attn.layer_idx == 3
    assert mtp_attn.source_layer_idx == 3
    assert hasattr(mtp_attn, "indexer")
    assert mtp_attn.dsa_topk_last_use[3] == 3
    assert mtp_attn.dsa_topk_recompute_release[3] == 3


def test_glm52_rejects_shared_physical_mtp_indexer():
    config = _tiny_glm52_config()
    config.attention.indexer_types = ["full", "full", "full", "shared"]
    config.mtp_config = MTPConfig(num_layers=1, share_weights=True)

    with pytest.raises(ValueError, match="physical MTP indexer_types"):
        config.build()


def test_glm52_multiple_physical_mtp_layers_share_first_mtp_indexer():
    config = _tiny_glm52_config()
    config.attention.indexer_types = ["full", "full", "full"]
    config.mtp_config = MTPConfig(num_layers=3)

    model = config.build()

    mtp_attentions = [layer.decoder_layer.self_attn for layer in model.mtp_block.layers]  # type: ignore[union-attr]
    assert config.attention.indexer_types == ["full", "full", "full", "full", "shared", "shared"]
    assert [attention.source_layer_idx for attention in mtp_attentions] == [3, 3, 3]
    assert hasattr(mtp_attentions[0], "indexer")
    assert not hasattr(mtp_attentions[1], "indexer")
    assert not hasattr(mtp_attentions[2], "indexer")
    assert mtp_attentions[0].dsa_topk_last_use[3] == 5
    assert mtp_attentions[0].dsa_topk_recompute_release[3] == 3


def test_glm52_mtp_iteration_shares_first_depth_topk():
    class TinyDsaMTPLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_layer = torch.nn.Module()
            self.decoder_layer.self_attn = DSAMLAConfig(
                num_attention_heads=2,
                head_dim=2,
                kv_lora_rank=3,
                q_lora_rank=4,
                qk_nope_head_dim=2,
                qk_rope_head_dim=2,
                v_head_dim=3,
                index_topk=4,
                index_head_dim=4,
                index_n_heads=2,
                indexer_types=["full", "shared", "shared", "full"],
            ).build(hidden_size=4, layer_idx=3)

        def forward(self, hidden_states, *, future_embeddings, position_embeddings, seq_ctx):
            outputs = self.decoder_layer.self_attn(hidden_states, position_embeddings, seq_ctx)
            projected_output = outputs["projected_output"]
            router_logits = torch.zeros(projected_output.shape[1], 1, device=projected_output.device)
            router_weights = torch.ones_like(router_logits)
            return projected_output, router_logits, router_weights

    hidden_size = 4
    mtp_block = MTPBlock(
        mtp_config=MTPConfig(num_layers=3, share_weights=True),
        mtp_layers=[TinyDsaMTPLayer()],
    )
    assert len(mtp_block.layers) == 1
    configure_dsa_mtp_iteration_lifecycle(
        mtp_block=mtp_block,
        attention=mtp_block.layers[0].decoder_layer.self_attn,
        num_iterations=3,
    )
    indexer_calls = 0

    def count_indexer_call(_module, _args, _output):
        nonlocal indexer_calls
        indexer_calls += 1

    mtp_block.layers[0].decoder_layer.self_attn.indexer.register_forward_hook(count_indexer_call)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
    hidden_states = torch.randn(1, input_ids.shape[1], hidden_size)
    position_embeddings = (torch.ones(1, input_ids.shape[1], 2), torch.zeros(1, input_ids.shape[1], 2))

    outputs = mtp_block(
        hidden_states,
        embed_tokens_fn=lambda token_ids: torch.zeros(*token_ids.shape, hidden_size),
        position_embeddings=position_embeddings,
        seq_ctx=seq_ctx,
    )

    assert len(outputs) == 3
    assert indexer_calls == 1
    assert set(seq_ctx.dsa_topk_cache.indices) == {3}


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
        assert sorted(seq_ctx.dsa_topk_cache.indices) == [0, 1, 2]

        seq_len = input_ids.numel()
        for topk_indices in seq_ctx.dsa_topk_cache.indices.values():
            assert topk_indices.shape == (seq_len, 1, seq_len)
            assert topk_indices.dtype == torch.int64
            assert topk_indices.device.type == "cuda"
            assert (topk_indices == -1).any()
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tiny GLM-5.2 compile oracle requires CUDA")
def test_tiny_glm52_native_compile_forward_matches_hf_numeric_oracle():
    tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)
    input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")

    hf_model = load_glm52_hf_oracle_model(GLM5_2_TINY_MOE_PATH)
    try:
        with torch.no_grad():
            expected_loss = hf_model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False).loss
    finally:
        del hf_model
        torch.cuda.empty_cache()

    with torch.device("meta"):
        config = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
        config.dispatcher = None
        config.ep_size = 1
        model = config.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

    assert model.compile_cfg["xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward"] == {
        "fullgraph": False
    }

    try:
        model.from_hf(GLM5_2_TINY_MOE_PATH)
        model.eval()

        seq_ctx, loss_ctx = _build_lm_loss_ctx(input_ids)
        with torch.no_grad():
            output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})

        # GLM keeps decoder-layer interfaces unchanged and lets seq_ctx cache writes
        # graph-break, while the heavy pure sub-functions remain compiled.
        torch.testing.assert_close(output["loss"], expected_loss.to(output["loss"].dtype), rtol=5e-2, atol=5e-2)
    finally:
        del model
        torch.cuda.empty_cache()


def test_glm52_micro_batch_forward_splits_dense_prefix_dsa_topk_cache_before_sparse_layers():
    class FakeEmbedding(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.float().unsqueeze(-1).expand(*input_ids.shape, self.hidden_size)

    class FakeRotaryEmbedding(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
            cos = torch.ones(hidden_states.shape[0], hidden_states.shape[1], 1, device=hidden_states.device)
            sin = torch.zeros_like(cos)
            return cos, sin

    class FakeDensePrefixLayer(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor, *, position_embeddings, seq_ctx: SequenceContext):
            seq_len = hidden_states.shape[1]
            seq_ctx.dsa_topk_cache.indices[0] = torch.arange(seq_len, device=hidden_states.device).view(seq_len, 1, 1)
            return hidden_states

    class FakeSparseSharedLayer(torch.nn.Module):
        def forward(self, *hidden_states_list: torch.Tensor, position_embeddings, seq_ctx: list[SequenceContext]):
            assert len(hidden_states_list) == len(seq_ctx)
            for micro_batch_idx, micro_batch_seq_ctx in enumerate(seq_ctx):
                assert set(micro_batch_seq_ctx.dsa_topk_cache.indices) == {0}
                topk_indices = micro_batch_seq_ctx.dsa_topk_cache.indices[0]
                assert topk_indices.shape[0] == micro_batch_seq_ctx.input_ids.shape[1]
                if micro_batch_idx == 0:
                    torch.testing.assert_close(topk_indices.flatten(), torch.tensor([0, 1]))
                else:
                    # Sparse micro-batches own local KV tensors, so cached
                    # indices are rebased from packed-global to local offsets.
                    torch.testing.assert_close(topk_indices.flatten(), torch.tensor([0, 1, 2]))

            router_logits = tuple(
                torch.zeros(hidden_states.shape[1], 1, device=hidden_states.device)
                for hidden_states in hidden_states_list
            )
            router_weights = tuple(
                torch.ones(hidden_states.shape[1], 1, device=hidden_states.device)
                for hidden_states in hidden_states_list
            )
            return (*hidden_states_list, *router_logits, *router_weights)

    class FakeAuxLoss(torch.nn.Module):
        def accumulate(self, *, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
            return hidden_states

        def finalize(self, **kwargs):
            return None, None, torch.zeros(1, 1)

    class FakeLossCtx:
        @classmethod
        def cat(cls, chunks):
            return cls()

    class FakeLMHead(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor, loss_ctx):
            return hidden_states.sum() * 0, (hidden_states, None)

    cfg = _tiny_glm52_config()
    model = cfg.build()
    model.embed_tokens = FakeEmbedding(cfg.hidden_size)
    model.rotary_emb = FakeRotaryEmbedding()
    model.layers = torch.nn.ModuleDict(
        {
            "0": FakeDensePrefixLayer(),
            "1": FakeSparseSharedLayer(),
        }
    )
    model.norm = torch.nn.Identity()
    model.lm_head = FakeLMHead()
    model.aux_loss = FakeAuxLoss()
    model.mtp_block = None

    seq_ctx_list = [
        SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu"),
        SequenceContext.from_input_ids((torch.tensor([[3, 4, 5]]),), device="cpu"),
    ]
    loss_ctx_list = [{"lm": FakeLossCtx()}, {"lm": FakeLossCtx()}]

    output = model(seq_ctx=seq_ctx_list, loss_ctx=loss_ctx_list)

    assert torch.isfinite(output["loss"])


def test_glm52_micro_batch_forward_accumulates_mtp_aux_stats_once_per_depth():
    class FakeEmbedding(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.float().unsqueeze(-1).expand(*input_ids.shape, self.hidden_size)

    class FakeRotaryEmbedding(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
            cos = torch.ones(hidden_states.shape[0], hidden_states.shape[1], 1, device=hidden_states.device)
            sin = torch.zeros_like(cos)
            return cos, sin

    class FakeDensePrefixLayer(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor, *, position_embeddings, seq_ctx: SequenceContext):
            return hidden_states

    class FakeSparseSharedLayer(torch.nn.Module):
        def forward(self, *hidden_states_list: torch.Tensor, position_embeddings, seq_ctx: list[SequenceContext]):
            router_logits = tuple(
                torch.zeros(hidden_states.shape[1], 1, device=hidden_states.device)
                for hidden_states in hidden_states_list
            )
            router_weights = tuple(
                torch.ones(hidden_states.shape[1], 1, device=hidden_states.device)
                for hidden_states in hidden_states_list
            )
            return (*hidden_states_list, *router_logits, *router_weights)

    class FakeMTPBlock(torch.nn.Module):
        def forward(self, *hidden_states_list: torch.Tensor, embed_tokens_fn, position_embeddings, seq_ctx):
            outputs = []
            for hidden_states in hidden_states_list:
                router_logits = torch.zeros(hidden_states.shape[1], 1, device=hidden_states.device)
                router_weights = torch.ones(hidden_states.shape[1], 1, device=hidden_states.device)
                outputs.append([(hidden_states, router_logits, router_weights)])
            return outputs

    class FakeAuxLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.accumulate_calls = 0

        def accumulate(self, *, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
            self.accumulate_calls += 1
            return hidden_states

        def finalize(self, **kwargs):
            return None, None, torch.zeros(self.accumulate_calls, 1)

    class FakeLossCtx:
        @classmethod
        def cat(cls, chunks):
            return cls()

    class FakeLMHead(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor, loss_ctx):
            return hidden_states.sum() * 0, (hidden_states, None)

    cfg = _tiny_glm52_config()
    cfg.mtp_config = MTPConfig(num_layers=1)
    model = cfg.build()
    model.embed_tokens = FakeEmbedding(cfg.hidden_size)
    model.rotary_emb = FakeRotaryEmbedding()
    model.layers = torch.nn.ModuleDict(
        {
            "0": FakeDensePrefixLayer(),
            "1": FakeSparseSharedLayer(),
        }
    )
    model.norm = torch.nn.Identity()
    model.lm_head = FakeLMHead()
    model.aux_loss = FakeAuxLoss()
    model.mtp_block = FakeMTPBlock()

    seq_ctx_list = [
        SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu"),
        SequenceContext.from_input_ids((torch.tensor([[3, 4, 5]]),), device="cpu"),
    ]
    loss_ctx_list = [
        {"lm": FakeLossCtx(), "mtp": [FakeLossCtx()]},
        {"lm": FakeLossCtx(), "mtp": [FakeLossCtx()]},
    ]

    output = model(seq_ctx=seq_ctx_list, loss_ctx=loss_ctx_list)

    assert model.aux_loss.accumulate_calls == 2
    assert output["tokens_per_expert_global"].shape[0] == 2


class TestGlm52MoE(DeterministicDDPTestCase):
    def test_sequence_parallel_mtp_loss_and_gradients_match_full_sequence(self):
        self.create_pg("cuda")

        cfg = _tiny_glm52_config()
        cfg.hidden_size = 128
        cfg.intermediate_size = 128
        cfg.moe_intermediate_size = 128
        cfg.mtp_config = MTPConfig(
            num_layers=2,
            share_weights=True,
        )
        cfg.lm_loss_cfg = CELossConfig(mode="eager")
        cfg.dispatcher = None
        cfg.ep_size = 1

        torch.manual_seed(17)
        baseline_model = cfg.build().to(device="cuda", dtype=torch.bfloat16)
        baseline_model.init_weights()
        sp_model = cfg.build().to(device="cuda", dtype=torch.bfloat16)
        sp_model.load_state_dict(baseline_model.state_dict())
        baseline_model.train()
        sp_model.train()

        # Five tokens in the first packed sequence place its boundary inside an
        # SP2/SP4 shard and directly across ranks for SP8.
        sequence_0 = torch.tensor([[2, 3, 4, 5, 6, 7]], device="cuda")
        sequence_1 = torch.tensor([[8, 9, 10, 11]], device="cuda")
        packed_inputs = (sequence_0[:, :-1], sequence_1[:, :-1])
        shifted_labels = torch.cat((sequence_0[:, 1:], sequence_1[:, 1:]), dim=1)

        baseline_seq_ctx = SequenceContext.from_input_ids(packed_inputs, device="cuda")
        baseline_data = {"seq_ctx": baseline_seq_ctx, "shifted_labels": shifted_labels}
        baseline_loss_ctx = baseline_model.build_loss_ctx_batch([baseline_data], sp_mesh=None)[0]
        baseline_output = baseline_model(seq_ctx=baseline_seq_ctx, loss_ctx=baseline_loss_ctx)
        baseline_total_loss = baseline_output["loss"] + baseline_output["mtp_loss"]
        baseline_total_loss.backward()

        baseline_gradients = {}
        for name, parameter in baseline_model.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float().clone()
            dist.all_reduce(gradient)
            baseline_gradients[name] = gradient / dist.get_world_size()

        for sp_size in (2, 4, 8):
            with self.subTest(sp_size=sp_size):
                sp_model.zero_grad(set_to_none=True)
                sp_mesh = init_data_mesh("cuda", sp_size=sp_size)["sp"]
                full_sp_seq_ctx = SequenceContext.from_input_ids(packed_inputs, device="cuda")
                sp_data = {"seq_ctx": full_sp_seq_ctx, "shifted_labels": shifted_labels}
                sp_loss_ctx = sp_model.build_loss_ctx_batch([sp_data], sp_mesh=sp_mesh)[0]
                sp_seq_ctx = full_sp_seq_ctx.split(sp_mesh)
                sp_output = sp_model(seq_ctx=sp_seq_ctx, loss_ctx=sp_loss_ctx)
                sp_total_loss = sp_output["loss"] + sp_output["mtp_loss"]
                sp_total_loss.backward()

                sp_gradients = {}
                for name, parameter in sp_model.named_parameters():
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad.detach().float().clone()
                    dist.all_reduce(gradient)
                    sp_gradients[name] = gradient / dist.get_world_size()

                torch.testing.assert_close(sp_output["loss"], baseline_output["loss"])
                torch.testing.assert_close(sp_output["mtp_loss"], baseline_output["mtp_loss"])
                self.assertEqual(sp_gradients.keys(), baseline_gradients.keys())
                max_relative_error = 0.0
                max_relative_error_name = ""
                min_cosine_similarity = 1.0
                for name in baseline_gradients:
                    actual = sp_gradients[name]
                    expected = baseline_gradients[name]
                    # SP changes BF16 GEMM and gradient-accumulation partitions.
                    # Compare global direction and magnitude, not bitwise values.
                    relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        actual.flatten(), expected.flatten(), dim=0
                    )
                    relative_error_value = float(relative_error)
                    if relative_error_value > max_relative_error:
                        max_relative_error = relative_error_value
                        max_relative_error_name = name
                    min_cosine_similarity = min(min_cosine_similarity, float(cosine_similarity))
                # More SP shards introduce more BF16 partial-gradient sums.
                # Across SP2/SP4/SP8 the worst observed relative L2 remains
                # below 2%, while every parameter keeps the same direction.
                message = f"sp_size={sp_size}, max_relative_error_parameter={max_relative_error_name}"
                self.assertLess(max_relative_error, 2e-2, message)
                self.assertGreater(min_cosine_similarity, 0.9999, message)

    @parametrize.parametrize(
        "device,dispatcher,ep_size,expected_full_layer_option",
        [
            ("cuda", None, 1, {"fullgraph": False}),
            ("cuda", "all2all", 8, None),
        ],
    )
    def test_glm52_default_compile_cfg_respects_ep_granularity(
        self, device, dispatcher, ep_size, expected_full_layer_option
    ):
        self.create_pg(device)

        with torch.device("meta"):
            cfg = _tiny_glm52_config()
            cfg.compile_cfg = None
            cfg.dispatcher = dispatcher
            cfg.ep_size = ep_size
            if dispatcher is not None:
                cfg.n_routed_experts = ep_size
            model = cfg.build()

        full_layer_target = "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward"
        expected_targets = {
            "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._pre_moe_forward": {"fullgraph": False},
            "xtuner.v1.module.attention.dsa_mla.DSAMultiLatentAttention.forward": {"fullgraph": False},
            "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": {"fullgraph": False},
            "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.forward": {"fullgraph": True},
            "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._shared_experts_forward": {
                "fullgraph": True
            },
            "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._post_moe_forward": {"fullgraph": True},
        }

        if expected_full_layer_option is not None:
            self.assertEqual(model.compile_cfg[full_layer_target], expected_full_layer_option)
        else:
            self.assertNotIn(full_layer_target, model.compile_cfg)

        for target, expected_option in expected_targets.items():
            self.assertEqual(model.compile_cfg[target], expected_option)

    @parametrize.parametrize(
        "device,dispatcher,ep_size,compile,tol,loss_mode",
        [
            ("cuda", "all2all", 8, False, 5e-2, "eager"),
            ("cuda", None, 1, False, 5e-2, "eager"),
            ("cuda", None, 1, False, 5e-2, "chunk"),
        ],
    )
    def test_glm52_moe_run(self, device, dispatcher, ep_size, compile, tol, loss_mode):
        self.create_pg(device)

        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
        hf_model = load_glm52_hf_oracle_model(GLM5_2_TINY_MOE_PATH)
        try:
            with torch.no_grad():
                expected_loss = hf_model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False).loss
        finally:
            del hf_model
            torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
            if not compile:
                cfg.compile_cfg = False
            cfg.dispatcher = dispatcher
            cfg.ep_size = ep_size
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

        try:
            model.from_hf(GLM5_2_TINY_MOE_PATH)
            model.eval()

            seq_ctx, loss_ctx = _build_lm_loss_ctx(input_ids, loss_mode=loss_mode)
            with torch.no_grad():
                output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})

            # See the logits oracle test above: HF expanded MLA and native
            # absorbed MLA accumulate bf16 through different matmul orders.
            self.assertTrue(torch.allclose(output["loss"], expected_loss.to(output["loss"].dtype), rtol=tol, atol=tol))
        finally:
            del model
            torch.cuda.empty_cache()

    @parametrize.parametrize(
        "device,dispatcher,ep_size,compile",
        [
            ("cuda", "all2all", 4, False),
            ("cuda", "all2all", 8, False),
            ("cuda", None, 1, False),
            ("cuda", None, 1, True),
        ],
    )
    def test_fsdp_accuracy(self, device, dispatcher, ep_size, compile):
        self.create_pg(device)

        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
        hf_model = load_glm52_hf_oracle_model(GLM5_2_TINY_MOE_PATH)
        try:
            with torch.no_grad():
                expected_loss = hf_model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False).loss
        finally:
            del hf_model
            torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
            if not compile:
                cfg.compile_cfg = False
            cfg.dispatcher = dispatcher
            cfg.ep_size = ep_size
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

        fsdp_config = FSDPConfig(ep_size=ep_size, cpu_offload=False)
        model.fully_shard(fsdp_config=fsdp_config)
        if compile:
            self.assertFalse(is_compiled_function(model.layers["0"].forward))
            self.assertTrue(is_compiled_function(model.layers["0"]._checkpoint_wrapped_module.forward))
            sparse_checkpoint_layer = next(
                (
                    layer
                    for layer in model.layers.values()
                    if hasattr(layer, "_checkpoint_wrapped_module")
                    and hasattr(layer._checkpoint_wrapped_module, "_pre_moe_forward")
                ),
                None,
            )
            # Keep checkpoint wrappers eager so activation checkpoint remains a
            # PyTorch higher-order op outside AOTAutograd; the wrapped decoder
            # layer still uses the class-level compile config.
            if sparse_checkpoint_layer is not None:
                sparse_wrapped_layer = sparse_checkpoint_layer._checkpoint_wrapped_module
                self.assertFalse(is_compiled_function(sparse_checkpoint_layer.forward))
                self.assertTrue(is_compiled_function(sparse_wrapped_layer.forward))
                self.assertTrue(is_compiled_function(sparse_wrapped_layer._pre_moe_forward))

        try:
            model.from_hf(GLM5_2_TINY_MOE_PATH)
            model.eval()

            seq_ctx, loss_ctx = _build_lm_loss_ctx(input_ids)
            with torch.no_grad():
                output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})

            self.assertTrue(
                torch.allclose(output["loss"], expected_loss.to(output["loss"].dtype), rtol=5e-2, atol=5e-2)
            )
        finally:
            del model
            torch.cuda.empty_cache()

    def test_activation_offload_compile_checkpoint_forward_backward_cleans_topk_cache(self):
        self.create_pg("cuda")

        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")

        with torch.device("meta"):
            cfg = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
            cfg.dispatcher = None
            cfg.ep_size = 1
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

        fsdp_config = FSDPConfig(ep_size=1, cpu_offload=False, recompute_ratio=1.0)
        model.fully_shard(fsdp_config=fsdp_config)

        try:
            model.from_hf(GLM5_2_TINY_MOE_PATH)
            model.train()

            seq_ctx, loss_ctx = _build_lm_loss_ctx(input_ids)
            with mock.patch.dict(os.environ, {"XTUNER_ACTIVATION_OFFLOAD": "1"}):
                output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
                self.assertTrue(torch.isfinite(output["loss"]).all())
                output["loss"].backward()
            torch.cuda.synchronize()

            self.assertEqual(seq_ctx.dsa_topk_cache.indices, {})
            self.assertEqual(seq_ctx.dsa_topk_cache.offloaded, {})
        finally:
            del model
            torch.cuda.empty_cache()

    def test_deepep_micro_batch_mtp_two_steps_clean_topk_cache(self):
        self.create_pg("cuda")

        cfg = _tiny_glm52_config()
        cfg.max_position_embeddings = 512
        cfg.num_hidden_layers = 5
        cfg.first_k_dense_replace = 3
        cfg.hidden_size = 128
        cfg.intermediate_size = 128
        cfg.moe_intermediate_size = 128
        cfg.mlp_layer_types = ["dense", "dense", "dense", "sparse", "sparse"]
        cfg.attention.indexer_types = ["full", "full", "full", "shared", "shared", "full"]
        cfg.mtp_config = MTPConfig(num_layers=1)
        cfg.lm_loss_cfg = CELossConfig(mode="chunk", chunk_size=128)
        cfg.compile_cfg = None
        cfg.dispatcher = "deepep"
        cfg.ep_size = 4
        model = cfg.build().to(device="cuda", dtype=torch.bfloat16)
        model.init_weights()

        fsdp_config = FSDPConfig(ep_size=4, cpu_offload=False, recompute_ratio=1.0)
        model.fully_shard(fsdp_config=fsdp_config)

        try:
            model.train()

            with mock.patch.dict(
                os.environ,
                {"XTUNER_ACTIVATION_OFFLOAD": "1", "XTUNER_DSA_TOPK_OFFLOAD": "1"},
            ):
                for step in range(2):
                    input_ids = ((torch.arange(257, device="cuda") + step) % 30 + 2).unsqueeze(0)
                    seq_ctx_list = [SequenceContext.from_input_ids(input_ids=(input_ids[:, :-1],)) for _ in range(2)]
                    data_batch = [{"seq_ctx": seq_ctx, "shifted_labels": input_ids[:, 1:]} for seq_ctx in seq_ctx_list]
                    loss_ctx_list = model.build_loss_ctx_batch(data_batch, sp_mesh=None)
                    self.assertTrue(all(loss_ctx["mtp"] is not None for loss_ctx in loss_ctx_list))

                    output = model(seq_ctx=seq_ctx_list, loss_ctx=loss_ctx_list)
                    self.assertTrue(torch.isfinite(output["loss"]).all())
                    self.assertIsNotNone(output["mtp_loss"])
                    total_loss = output["loss"] + output["mtp_loss"]
                    total_loss.backward()
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()

                    for seq_ctx in seq_ctx_list:
                        self.assertEqual(seq_ctx.dsa_topk_cache.indices, {})
                        self.assertEqual(seq_ctx.dsa_topk_cache.offloaded, {})
        finally:
            del model
            torch.cuda.empty_cache()

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", None, 1),
        ],
    )
    def test_save_hf(self, device, dispatcher, ep_size):
        self.create_pg(device)

        with torch.device("meta"):
            cfg = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
            cfg.compile_cfg = False
            cfg.dispatcher = dispatcher
            cfg.ep_size = ep_size
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

        fsdp_config = FSDPConfig(ep_size=ep_size, cpu_offload=False)
        origin_fh_cache = {}
        saved_fh_cache = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])

            model.fully_shard(fsdp_config=fsdp_config)
            model.from_hf(GLM5_2_TINY_MOE_PATH)
            model.save_hf(tmpdir)

            origin_hf_path = Path(GLM5_2_TINY_MOE_PATH)
            origin_index_path = origin_hf_path / "model.safetensors.index.json"
            saved_index_path = tmpdir / "model.safetensors.index.json"

            if dist.get_rank() == 0:
                with open(origin_index_path) as f:
                    origin_index = json.load(f)
                with open(saved_index_path) as f:
                    saved_index = json.load(f)

                self.assertListEqual(
                    sorted(origin_index["weight_map"]),
                    sorted(saved_index["weight_map"]),
                )

                for key, origin_safetensor_name in origin_index["weight_map"].items():
                    saved_safetensor_name = saved_index["weight_map"][key]

                    if origin_safetensor_name not in origin_fh_cache:
                        origin_fh_cache[origin_safetensor_name] = safe_open(
                            str(origin_hf_path / origin_safetensor_name), framework="pt"
                        )
                    if saved_safetensor_name not in saved_fh_cache:
                        saved_fh_cache[saved_safetensor_name] = safe_open(
                            str(tmpdir / saved_safetensor_name), framework="pt"
                        )

                    origin_tensor = origin_fh_cache[origin_safetensor_name].get_tensor(key)
                    saved_tensor = saved_fh_cache[saved_safetensor_name].get_tensor(key)
                    self.assertTrue(torch.equal(origin_tensor, saved_tensor), f"tensor {key} is not equal")

                safetensor_keys = []
                for safetensor_path in tmpdir.glob("*.safetensors"):
                    safetensor_keys.extend(saved_fh_cache[safetensor_path.name].keys())
                    safetensor_keys.sort()
                self.assertListEqual(safetensor_keys, sorted(saved_index["weight_map"]))
        dist.barrier()

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
