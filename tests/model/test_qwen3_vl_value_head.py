"""Qwen3 VL value-head behavior tests.

TestQwen3VLValueHead
    test_inference_head_supports_distributed_weight
"""

import tempfile

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, Replicate, distribute_tensor

from xtuner.v1.model.base import HeadConfig
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDenseConfig
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.module.attention import MHAConfig


class TestQwen3VLValueHead:
    """Check Qwen3 VL scalar-value inference through the public model forward."""

    def test_inference_head_supports_distributed_weight(self):
        # 验证 value head 使用 DTensor 权重时前向不再混用普通 Tensor 与 DTensor。
        with tempfile.TemporaryDirectory() as temp_dir:
            dist.init_process_group("gloo", init_method=f"file://{temp_dir}/store", rank=0, world_size=1)
            try:
                config = Qwen3_5_VLTextDenseConfig(
                    vocab_size=32,
                    max_position_embeddings=128,
                    eos_token_id=31,
                    hidden_size=16,
                    intermediate_size=64,
                    rms_norm_eps=1e-6,
                    hidden_act="silu",
                    attention=MHAConfig(num_attention_heads=4, num_key_value_heads=4, head_dim=4),
                    num_hidden_layers=0,
                    head_config=HeadConfig(type="value_head"),
                )
                model = config.build()
                mesh = DeviceMesh("cpu", [0])
                model.lm_head.weight = torch.nn.Parameter(distribute_tensor(model.lm_head.weight, mesh, [Replicate()]))
                seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3]]),))
                seq_ctx.position_ids = torch.arange(3).unsqueeze(0)
                seq_ctx.to("cpu")

                output = model(seq_ctx, loss_ctx=None)

                assert output["logits"].shape == (1, 3, 1)
                assert torch.isfinite(output["logits"]).all()
            finally:
                dist.destroy_process_group()
