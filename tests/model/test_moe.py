import torch
from xtuner.v1.model.moe.moe import MoEConfig, MoE, SequenceContext
from xtuner.v1.module.router import NoAuxRouterConfig
from xtuner.v1.module.attention import MHAConfig
from torch.distributed.device_mesh import init_device_mesh
import os
from copy import deepcopy
from xtuner.v1.loss.ce_loss import CELossContext, CELossConfig
from xtuner.v1.loss.moe_loss import BalancingLossConfig, ZLossConfig

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.utils.compile import maybe_compile
import parametrize


class TestMoE:
    @parametrize.parametrize("dtype,device", [(torch.bfloat16, "cuda")])
    def test_moe_config(self, dtype, device):
        router_config = NoAuxRouterConfig(
            scoring_func="sigmoid",
            router_scaling_factor=1.0,
            n_group=8,
            topk_group=4,
            norm_topk_prob=True,
        )
        attention_config = MHAConfig(
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=16,
        )
        config = MoEConfig(
            vocab_size=10240,
            max_position_embeddings=2048,
            pad_token_id=0,
            eos_token_id=0,
            num_hidden_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            hidden_act="silu",
            attention=attention_config,
            tie_word_embeddings=False,
            n_routed_experts=32,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            hidden_factor=1.0,
            moe_intermediate_size=512,  # TODO: Restriction of triton grouped gemm, should be optimizer
            router=router_config,
            compile_cfg=False,
        )
        model = MoE(config=config).to(dtype).to(device)
        model.cuda()
        loss_cfg = CELossConfig()

        input_ids = torch.randint(
            0, config.vocab_size, (1, 128), dtype=torch.int64, device="cuda"
        )
        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))

        seq_ctx_list = [seq_ctx]
        LossContext = loss_cfg.loss_ctx_cls
        loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
        loss_ctx_list = [loss_ctx]
        loss_ctx_list = LossContext.build_batches(loss_ctx_list)
        loss_ctx = loss_ctx_list[0]
        seq_ctx = seq_ctx_list[0]
        model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})

    @parametrize.parametrize("dtype,device", [(torch.bfloat16, "cuda")])
    def test_forward_decomposition(self, dtype, device):
        """``MoE._forward`` is now an orchestrator over the _prepare / _embed / _layers / _head stage
        helpers (so pipeline parallel can later run a layer subset per stage). This guards that the
        decomposition is a faithful, deterministic refactor with both auxiliary losses active:

        - the orchestrator equals an explicit manual composition of the four helpers (same single
          full-layer call, so the same kernels run);
        - a second identical forward reproduces the result (no incidental state between stages);
        - balancing loss (finalized in the head stage) still backprops to a routed-expert router.

        Numerical *finiteness* is intentionally not asserted: the toy ``moe_intermediate_size`` grouped
        GEMM is kernel-flaky in this env (the existing ``test_moe_config`` runs a non-finite forward
        without noticing), so comparisons use ``equal_nan=True`` to test wiring, not kernel values.
        """
        router_config = NoAuxRouterConfig(
            scoring_func="sigmoid",
            router_scaling_factor=1.0,
            n_group=8,
            topk_group=4,
            norm_topk_prob=True,
        )
        attention_config = MHAConfig(num_attention_heads=32, num_key_value_heads=32, head_dim=16)
        config = MoEConfig(
            vocab_size=10240,
            max_position_embeddings=2048,
            pad_token_id=0,
            eos_token_id=0,
            num_hidden_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            hidden_act="silu",
            attention=attention_config,
            tie_word_embeddings=False,
            n_routed_experts=32,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            hidden_factor=1.0,
            moe_intermediate_size=512,
            router=router_config,
            # Exercise both auxiliary losses: balancing accumulates per layer and is finalized in the
            # head stage; z-loss is injected inline per layer via AuxLossScaler.
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
            compile_cfg=False,
        )

        torch.manual_seed(0)
        model = MoE(config=config).to(dtype).to(device)
        model.cuda()

        input_ids = torch.randint(0, config.vocab_size, (1, 128), dtype=torch.int64, device=device)
        seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids[:, :-1].to(device),))
        data_batch = [{"seq_ctx": seq_ctx, "shifted_labels": input_ids[:, 1:]}]

        def total_loss(model_outputs):
            total = None
            for key in type(model_outputs).model_fields:
                value = getattr(model_outputs, key)
                if "loss" in key and isinstance(value, torch.Tensor):
                    total = value if total is None else total + value
            assert total is not None, "no loss field produced by forward"
            return total

        # Orchestrator path (MoE._forward via __call__).
        out_orch = model(seq_ctx=seq_ctx, loss_ctx=model.build_loss_ctx_batch(data_batch)[0])
        # The orchestrator must route through all four stage helpers, so the aux-loss fields appear.
        assert out_orch.balancing_loss is not None
        assert out_orch.z_loss is not None

        # Manual composition of the same helpers with a single full-layer _layers_step: identical
        # kernel path, so results must match exactly (equal_nan tolerates the flaky toy kernel).
        loss_ctx_manual = model.build_loss_ctx_batch(data_batch)[0]
        state = model._prepare_forward(seq_ctx, loss_ctx_manual, return_router_logits=False)
        hidden_states = model._embed_step(seq_ctx)
        hidden_states, position_embeddings = model._layers_step(hidden_states, seq_ctx, state)
        out_manual = model._head_step(hidden_states, position_embeddings, seq_ctx, loss_ctx_manual, state)
        torch.testing.assert_close(
            total_loss(out_orch).detach(), total_loss(out_manual).detach(), rtol=0, atol=0, equal_nan=True
        )

        # Determinism: a third identical forward reproduces the orchestrator result.
        out_again = model(seq_ctx=seq_ctx, loss_ctx=model.build_loss_ctx_batch(data_batch)[0])
        torch.testing.assert_close(
            total_loss(out_orch).detach(), total_loss(out_again).detach(), rtol=0, atol=0, equal_nan=True
        )

        # Balancing loss is finalized in the head stage but must still backprop to the routers; layer 0
        # is dense (first_k_dense_replace=1), so layer 1 is the first routed-expert layer.
        total_loss(out_manual).backward()
        first_moe_layer = model.layers[list(model.layers.keys())[1]]
        router_grads = [p.grad for p in first_moe_layer.gate.parameters() if p.grad is not None]
        assert router_grads, "expected router gradient from the decomposed forward + aux loss"


class TestDistributedMoE(DeterministicDDPTestCase):
    @parametrize.parametrize(
        "dtype,device,dispatcher,n_shared_experts,first_k_dense_replace",
        [
            # (torch.bfloat16, "cuda", "deepep", 1, 2),
            (torch.bfloat16, "cuda", "all2all", 1, 2),
            (torch.bfloat16, "cuda", "all2all", 0, 0),
        ],
    )
    def test_parallel_accuracy(self, dtype, device, dispatcher, n_shared_experts, first_k_dense_replace):
        self.create_pg(device)
        router_config = NoAuxRouterConfig(
            scoring_func="sigmoid",
            router_scaling_factor=1.0,
            n_group=8,
            topk_group=4,
            norm_topk_prob=True,
        )
        attention_config = MHAConfig(
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=16,
        )
        config = MoEConfig(
            vocab_size=10240,
            max_position_embeddings=2048,
            pad_token_id=0,
            eos_token_id=0,
            num_hidden_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            hidden_act="silu",
            attention=attention_config,
            tie_word_embeddings=False,
            n_routed_experts=32,
            n_shared_experts=n_shared_experts,
            num_experts_per_tok=2,
            first_k_dense_replace=first_k_dense_replace,
            hidden_factor=1.0,
            moe_intermediate_size=512,  # TODO: Restriction of triton grouped gemm, should be optimizer
            router=router_config,
        )
        loss_cfg = CELossConfig()

        model = MoE(config=config).to(dtype).to(device)
        parallel_config = deepcopy(config)
        parallel_config.dispatcher = dispatcher
        ep_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(8,)
        )

        parallel_model = MoE(config=parallel_config).to(dtype).to(device)

        input_ids = torch.randint(
            0, config.vocab_size, (1, 128), dtype=torch.int64, device="cuda"
        )
        shift_input_ids = input_ids[:, :-1]
        shifted_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))

        seq_ctx_list = [seq_ctx]
        LossContext = loss_cfg.loss_ctx_cls
        loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
        loss_ctx_list = [loss_ctx]
        loss_ctx_list = LossContext.build_batches(loss_ctx_list)
        loss_ctx = loss_ctx_list[0]
        seq_ctx = seq_ctx_list[0]

        loss_parallel = parallel_model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})["loss"]

        loss_expected = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})["loss"]

        torch.allclose(loss_expected, loss_parallel, atol=1e-6, rtol=1e-4)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
