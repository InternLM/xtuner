import os
import torch
import torch.distributed as dist
from torch import nn

from transformers import AutoModelForCausalLM, AutoConfig

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.module.rope.rope import FourierEmbedding
from xtuner.v1.module.rope.rope import RopeScalingConfig
from xtuner.v1.utils.device import get_device

DEVICE = get_device()

QWEN3_MOE_FOPE_PATH = os.environ["QWEN3_MOE_FOPE_PATH"]


class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        with torch.device(DEVICE):
            # same usage with MoE.__init__()
            self.fope = FourierEmbedding(config)
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x, position_ids):
        cos, sin = self.fope(x, position_ids)
        y = self.fc(x)
        return cos, sin, y


class TestFoPE(DeterministicDDPTestCase):
    def test_fope_init_sin_coef_same(self):
        self.create_pg(DEVICE)
        torch.accelerator.set_device_index(int(dist.get_rank()))

        # 1. create & operate
        model_cfg = Qwen3MoE30BA3Config(rope_scaling_cfg=RopeScalingConfig(
            type="default",
            # fope specific parameters
            fope_init_factor=0.1,
            fope_sep_head=True,
            num_inv_freq=None,
        ))
        model = MyModel(model_cfg).to(DEVICE)

        # 2. check
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # torch gather sin_coef and cos_coef tensors from all ranks
        sin_coef, cos_coef = model.fope.sin_coef, model.fope.cos_coef
        sin_coef_list = [torch.zeros_like(sin_coef) for _ in range(world_size)] if rank == 0 else None 
        cos_coef_list = [torch.zeros_like(cos_coef) for _ in range(world_size)] if rank == 0 else None 
        dist.gather(sin_coef, sin_coef_list, dst=0)
        dist.gather(cos_coef, cos_coef_list, dst=0)
        if rank == 0:
            for i in range(world_size):
                assert torch.equal(sin_coef, sin_coef_list[i])
                assert torch.equal(cos_coef, cos_coef_list[i])
    
    def test_fope_forward(self):
        self.create_pg(DEVICE)
        torch.accelerator.set_device_index(int(dist.get_rank()))

        fope_scaling_kwargs = dict(
            type="default",
            fope_init_factor=0.1,
            fope_sep_head=True,
            num_inv_freq=40,
        )
        seq_len = 32768
        # 1.create xtuner fope and forward
        model_cfg = Qwen3MoE30BA3Config(
            max_position_embeddings=seq_len,
            rope_scaling_cfg=RopeScalingConfig(
                **fope_scaling_kwargs,
            ))
        fope = FourierEmbedding(model_cfg).to(DEVICE)

        # prepare input
        hidden_size = model_cfg.hidden_size
        model_cfg.num_attention_heads
        x = torch.randn(1, seq_len, hidden_size).to(DEVICE)  # [batch_size, seq_len, hidden_size]
        position_ids = torch.arange(seq_len).reshape(1, seq_len).to(DEVICE)  # [batch_size, seq_len]

        # forward
        cos, sin = fope(x, position_ids)  # [batch_size, kv_heads, seq_len, head_dim]
        print(f"cos.shape: {cos.shape}, sin.shape: {sin.shape}")  # [1, 4, 32768, 128]

        # 2.create hf remote_code fope and forward
        hf_config = AutoConfig.from_pretrained(
            QWEN3_MOE_FOPE_PATH,
            trust_remote_code=True,
        )
        hf_config.rope_scaling = fope_scaling_kwargs
        hf_config.max_position_embeddings = seq_len
        print(f"hf_config: {hf_config}")

        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MOE_FOPE_PATH,
            config=hf_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=DEVICE
        )
        print(f"hf_model.model.rotary_emb.num_inv_freq: {hf_model.model.rotary_emb.num_inv_freq}")
        hf_fope_emb = hf_model.model.rotary_emb.to(torch.float32)
        hf_fope_emb.sin_coef.copy_(fope.sin_coef)
        hf_fope_emb.cos_coef.copy_(fope.cos_coef)

        ref_cos, ref_sin = hf_fope_emb(x, position_ids)
        print(f"ref_cos.shape: {ref_cos.shape}, ref_sin.shape: {ref_sin.shape}")

        # 3.check xtuner fope and hf remote_code fope forward result are the same
        torch.testing.assert_close(cos, ref_cos)
        torch.testing.assert_close(sin, ref_sin)
