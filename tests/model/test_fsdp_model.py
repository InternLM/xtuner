import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner._testing.testcase import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import BaseModel, ModelOutputs, XTunerBaseModelConfig
from xtuner.v1.module import LMHead


class ToyModelConfig(XTunerBaseModelConfig):
    vocab_size: int = 32
    hidden_size: int = 16
    intermediate_size: int = 24

    def build(self) -> "ToyModel":
        return ToyModel(self)


class ToyModel(BaseModel):
    config: ToyModelConfig

    def __init__(self, config: ToyModelConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.lm_head = LMHead(config.intermediate_size, config.vocab_size, bias=False)
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]

    def forward(self, seq_ctx: SequenceContext, loss_ctx=None) -> ModelOutputs:
        assert seq_ctx.input_ids is not None
        hidden_states = self.embed_tokens(seq_ctx.input_ids)
        hidden_states = torch.relu(self.fc1(hidden_states))

        lm_loss_ctx = loss_ctx["lm"] if loss_ctx is not None else None
        loss, (logits, extra_info) = self.lm_head(hidden_states, lm_loss_ctx)
        return ModelOutputs(loss=loss, logits=logits, extra_info=extra_info)


class ReferenceToyModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16, intermediate_size: int = 24):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.lm_head = LMHead(intermediate_size, vocab_size, bias=False)

    def forward(self, seq_ctx: SequenceContext, loss_ctx=None) -> ModelOutputs:
        assert seq_ctx.input_ids is not None
        hidden_states = self.embed_tokens(seq_ctx.input_ids)
        hidden_states = torch.relu(self.fc1(hidden_states))

        lm_loss_ctx = loss_ctx["lm"] if loss_ctx is not None else None
        loss, (logits, extra_info) = self.lm_head(hidden_states, lm_loss_ctx)
        return ModelOutputs(loss=loss, logits=logits, extra_info=extra_info)


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _build_batch(vocab_size: int, device: str) -> tuple[SequenceContext, dict[str, object]]:
    input_ids = torch.randint(0, vocab_size, (1, 9), dtype=torch.int64, device=device)
    seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids[:, :-1],), device=device)

    loss_cfg = CELossConfig()
    loss_ctx = loss_cfg.build(data={"shifted_labels": input_ids[:, 1:]}, sp_mesh=None)
    assert loss_ctx is not None
    loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
    return seq_ctx, {"lm": loss_ctx}


class TestFSDPModel(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def test_model_forward_backward(self):
        self.create_pg("cuda")

        torch.manual_seed(0)
        device = "cuda"
        config = ToyModelConfig(compile_cfg=False)
        seq_ctx, loss_ctx = _build_batch(config.vocab_size, device)

        ref_model = ReferenceToyModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        ).to(device)
        model = config.build().to(device)
        model.load_state_dict(ref_model.state_dict())

        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-3, weight_decay=1e-2)
        ref_output = ref_model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        assert ref_output.loss is not None
        assert ref_output.logits is not None
        ref_output.loss.backward()
        for param in ref_model.parameters():
            assert param.grad is not None
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

        fsdp_config = FSDPConfig(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            torch_compile=False,
        )
        model.fully_shard(fsdp_config=fsdp_config)
        fsdp_optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        output = model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        assert output.loss is not None
        assert output.logits is not None
        output.loss.backward()

        torch.testing.assert_close(output.loss, ref_output.loss)
        torch.testing.assert_close(output.logits, ref_output.logits)

        ref_optim.step()
        fsdp_optim.step()

        for name, ref_param in ref_model.state_dict().items():
            torch.testing.assert_close(_full_tensor(model.state_dict()[name]), ref_param)
