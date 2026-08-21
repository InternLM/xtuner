import torch

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.dense.qwen3 import Qwen3DenseConfig
from xtuner.v1.module.attention import MHAConfig


class TestFSDPCheckpoint(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def test_reentrant_checkpoint_keeps_fsdp_outside_recompute(self):
        self.create_pg("cuda")
        config = Qwen3DenseConfig(
            vocab_size=64,
            max_position_embeddings=64,
            eos_token_id=2,
            bos_token_id=1,
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attention=MHAConfig(
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                qk_norm=True,
            ),
            compile_cfg=False,
        )
        model = config.build().cuda()
        grad_modes: list[bool] = []
        original_layer = model.layers["0"]
        original_layer.register_forward_pre_hook(lambda _module, _inputs: grad_modes.append(torch.is_grad_enabled()))

        model.fully_shard(
            FSDPConfig(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                torch_compile=False,
            )
        )
        checkpoint_calls = 0

        def record_checkpoint_call(_module, _inputs, _output):
            nonlocal checkpoint_calls
            checkpoint_calls += 1

        model.layers["0"].register_forward_hook(record_checkpoint_call)
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device="cuda")
        output = model(SequenceContext.from_input_ids((input_ids,)))
        assert output.logits is not None
        output.logits.sum().backward()

        # The original layer and its lifecycle hooks must be replayed, while
        # the outer FSDP/checkpoint boundary is one logical forward only.
        assert grad_modes == [False, True]
        assert checkpoint_calls == 1
