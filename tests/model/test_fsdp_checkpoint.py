import torch

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.dense.qwen3 import Qwen3DenseConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.utils.compile import is_compiled_function


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

    def test_qwen3_vl_checkpoint_compile_allows_pytree_boundary(self):
        self.create_pg("cuda")
        from xtuner.v1.model.compose.qwen3_vl.qwen3_vl_config import Qwen3VLVisionConfig

        compile_target = "xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward"
        config = Qwen3VLVisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=1,
            num_position_embeddings=4,
            deepstack_visual_indexes=[],
            attn_impl="eager_attention",
            compile_cfg={compile_target: {"fullgraph": True}},
        )
        model = config.build().cuda()
        model.fully_shard(FSDPConfig(vision_recompute_ratio=1.0))
        assert is_compiled_function(model.blocks[0].forward)

        hidden_states = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        cu_seqlens = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
        cos = torch.ones(4, 8, device="cuda", dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)
        output = model.blocks[0](hidden_states, cu_seqlens, 4, (cos, sin))
        output.square().sum().backward()

        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

    def test_intern_s1_checkpoint_compile_allows_pytree_boundary(self):
        self.create_pg("cuda")
        from xtuner.v1.model.compose.intern_s1.intern_s1_config import InternS1VisionConfig

        compile_target = "xtuner.v1.model.compose.intern_s1.modeling_vision.InternS1VisionLayer.forward"
        config = InternS1VisionConfig(
            image_size=(4, 4),
            patch_size=(2, 2),
            num_hidden_layers=1,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            attn_impl="eager_attention",
            compile_cfg={compile_target: {"fullgraph": True}},
        )
        model = config.build().cuda()
        model.fully_shard(FSDPConfig(vision_recompute_ratio=1.0))
        assert is_compiled_function(model.encoder.layer[0].forward)

        hidden_states = torch.randn(1, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        output = model.encoder.layer[0](hidden_states)
        output.square().sum().backward()

        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

    def test_mixed_dense_checkpoint_compile_allows_pytree_boundary(self):
        self.create_pg("cuda")
        from xtuner.v1.loss.ce_loss import CELossConfig
        from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDenseConfig
        from xtuner.v1.module.attention import GatedDeltaNetConfig

        config = Qwen3_5_VLTextDenseConfig(
            vocab_size=64,
            max_position_embeddings=64,
            eos_token_id=2,
            num_hidden_layers=4,
            hidden_size=128,
            intermediate_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attention=MHAConfig(
                with_gate=True,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                qk_norm=True,
                rms_norm_eps=1e-6,
                rms_norm_type="zero_centered",
            ),
            linear_attention=GatedDeltaNetConfig(
                num_value_heads=4,
                num_key_heads=4,
                key_head_dim=16,
                value_head_dim=16,
                conv_kernel_dim=4,
                hidden_act="silu",
                rms_norm_eps=1e-6,
            ),
        )
        model = config.build().cuda()
        model.fully_shard(FSDPConfig(recompute_ratio=1.0, torch_compile=True))
        assert all(is_compiled_function(layer.forward) for layer in model.layers.values())

        input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
        seq_ctx = SequenceContext.from_input_ids((input_ids[:, :-1],))
        loss_config = CELossConfig(mode="eager")
        loss_ctx = loss_config.build(
            data={"shifted_labels": input_ids[:, 1:]},
            sp_mesh=None,
        )
        loss_ctx = loss_config.loss_ctx_cls.build_batches([loss_ctx])[0]
        output = model(seq_ctx, {"lm": loss_ctx})
        assert output.loss is not None
        output.loss.backward()

        assert torch.isfinite(output.loss)
        assert any(parameter.grad is not None for parameter in model.parameters())
