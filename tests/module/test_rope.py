import torch
import torch.distributed as dist
from torch import nn

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.module.rope.rope import FourierEmbedding
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.module.rope.rope import RopeScalingConfig
from xtuner.v1.utils.device import get_device

DEVICE = get_device()


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
