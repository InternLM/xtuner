import os
from typing import cast
import torch
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEGate
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
from xtuner.v1.utils import internal_metrics
from xtuner.v1.utils.internal_metrics import InternalMetricsConfig, InternalMetricsRecorder
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]

def _get_model_config() -> Qwen3MoEConfig:
    return Qwen3MoEConfig(
        vocab_size=151936,
        max_position_embeddings=4096,
        pad_token_id=0,
        bos_token_id=151643,
        eos_token_id=151645,
        num_hidden_layers=1,
        hidden_size=2048,
        intermediate_size=6144,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        hidden_act="silu",
        attention=MHAConfig(
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
        ),
        tie_word_embeddings=False,
        n_routed_experts=16,
        n_shared_experts=0,
        num_experts_per_tok=1,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=768,
        router=GreedyRouterConfig(
            scoring_func="softmax",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        ),
    )


class TestInternalMetricsRecorder(DeterministicDDPTestCase):
    def test_internal_metrics_run(self):
        self.create_pg("cuda")

        config = _get_model_config()
        with torch.device("meta"):
            model = config.build()

        fsdp_config = FSDPConfig()
        model.fully_shard(fsdp_config=fsdp_config)
        model.init_weights()

        internal_metrics_interval = 1

        internal_metrics_cfg = InternalMetricsConfig(
            internal_metrics_interval=internal_metrics_interval,
            monitor_weights_rms_norm=True,
            monitor_attn_logits_stats=True,
            monitor_moe_router_logits_stats=True,
            monitor_moe_load_balance_stats=True,
        )

        metrics_recorder = InternalMetricsRecorder(internal_metrics_cfg, model)

        hf_model_path = QWEN3_MOE_PATH
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

        text_list = [
            "一个好的研究者应自己先审视自己的 claim, 并真心地尝试用实验检验它们",
        ]

        data_batches = []

        for text in text_list:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))
            data_batches.append(ModelItem(seq_ctx=seq_ctx, loss_ctx=None))  # type: ignore[arg-type]

        metrics = metrics_recorder.pop_metrics(data_batches)

        # Check that all expected top-level keys exist
        assert "weight_rms" in metrics
        assert "router_logits_max" in metrics
        assert "router_logits_mean" in metrics
        assert "maxvio" in metrics
        assert "drop_ratio" in metrics

        if DEVICE != "npu":
            assert "attn_max_lse" in metrics or "attn_max_logits" in metrics

        # Check that all values are valid floats (not NaN or Inf)
        for metric_name, metric_dict in metrics.items():
            assert isinstance(metric_dict, dict), f"{metric_name} should be a dict"
            for key, value in metric_dict.items():
                assert isinstance(value, float), f"{metric_name}[{key}] should be float"
                assert not torch.isnan(torch.tensor(value)), f"{metric_name}[{key}] is NaN"
                assert not torch.isinf(torch.tensor(value)), f"{metric_name}[{key}] is Inf"

        for key in ["embed_tokens", "lm_head"] + [f"layers.{i}" for i in range(model.config.num_hidden_layers)]:
            assert key in metrics["weight_rms"], f"key: {key}, weight_rms: {metrics['weight_rms']}"

        for key in [f"layer{i}" for i in range(model.config.num_hidden_layers)]:
            assert key in metrics["maxvio"], f"key: {key}, maxvio: {metrics['maxvio']}"
            assert key in metrics["drop_ratio"], f"key: {key}, drop_ratio: {metrics['drop_ratio']}"
            assert key in metrics["router_logits_max"], f"key: {key}, router_logits_max: {metrics['router_logits_max']}"
            assert key in metrics["router_logits_mean"], f"key: {key}, router_logits_mean: {metrics['router_logits_mean']}"

        if DEVICE != "npu":
            for layer in range(model.config.num_hidden_layers):
                assert (
                        f"layers.{layer}.self_attn" in metrics["attn_max_lse"] or  # type: ignore[attr-defined] 
                        f"layers.{layer}.self_attn" in metrics["attn_max_logits"]  # type: ignore[attr-defined]
                )

        assert "total" in metrics["maxvio"]
        assert "total" in metrics["drop_ratio"]
