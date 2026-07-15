import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from xtuner._testing.testcase import DeterministicDDPTestCase
from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig


class _ReduceSumToyConfig(XTunerBaseModelConfig):
    hidden_size: int = 8

    def build(self) -> "_ReduceSumToyModel":
        return _ReduceSumToyModel(self)


class _ReduceSumToyModel(BaseModel):
    config: _ReduceSumToyConfig

    def __init__(self, config: _ReduceSumToyConfig):
        super().__init__(config)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestReduceSumGradient(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def test_bf16_reduce_sum_equals_local_grad_sum(self):
        # Regression guard for the torch 2.10 bf16 FSDP reduce-sum path: `set_gradient_reduce_sum`
        # must make the reduce-scatter yield the exact SUM of per-rank local gradients, not the
        # AVG default and not the all-zero result of the NCCL PreMulSum bf16 bug (the failure that
        # appears when only the divide factor is set). Assertions run in bf16 on purpose; fp32
        # reduction would mask the PreMulSum zeroing.
        self.create_pg("cuda")

        torch.manual_seed(0)
        dim = 8
        model = _ReduceSumToyConfig(hidden_size=dim, compile_cfg=False).build().cuda()
        # Gradient of `(x @ w.T).sum()` w.r.t. `w` is independent of the weight value, so the bf16
        # weight cast under FSDP does not perturb the reference; only the reduction dtype matters.
        ref_weight = model.fc.weight.detach().clone()

        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        fully_shard(model, mp_policy=mp_policy)
        model.set_gradient_reduce_sum()

        rank = dist.get_rank()
        world = dist.get_world_size()
        # Distinct input per rank so local gradients differ; SUM and MEAN are then clearly separable.
        x = (torch.arange(dim, dtype=torch.float32, device="cuda") + rank + 1).reshape(1, dim)
        model(x).sum().backward()
        full_grad = model.fc.weight.grad.full_tensor().float()

        # Independent reference: this rank's local gradient on an unsharded copy, all-gathered.
        w = ref_weight.clone().detach().requires_grad_(True)
        (x @ w.T).sum().backward()
        g_local = w.grad.float()
        gathered = [torch.zeros_like(g_local) for _ in range(world)]
        dist.all_gather(gathered, g_local)
        g_sum = sum(gathered)
        g_mean = g_sum / world

        assert not torch.all(full_grad.abs() < 1e-9), "bf16 reduce-sum returned all-zero gradients"
        rel_to_sum = ((full_grad - g_sum).norm() / (g_sum.norm() + 1e-12)).item()
        rel_to_mean = ((full_grad - g_mean).norm() / (g_mean.norm() + 1e-12)).item()
        assert rel_to_sum < 1e-2, f"expected SUM of local grads, rel_to_sum={rel_to_sum}"
        assert rel_to_mean > 0.1, f"gradient matched MEAN, reduce-sum not applied, rel_to_mean={rel_to_mean}"
