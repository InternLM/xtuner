import torch
from xtuner.v1.model.moe.moe import MoEConfig, MoE, SequenceContext
from xtuner.v1.module.router import NoAuxRouterConfig
from xtuner.v1.module.attention import MHAConfig
from torch.distributed.device_mesh import init_device_mesh
import os
from copy import deepcopy
from xtuner.v1.loss import CELossContext

from torch.testing._internal.common_distributed import DistributedTestBase
import parametrize


class TestMoE:
    @parametrize.parametrize("dtype,device", [(torch.bfloat16, "cuda")])
    def test_moe_config(self, dtype, device):
        router_config = NoAuxRouterConfig(
            scoring_func="sigmoid",
            router_scaling_factor=1.0,
            routed_scaling_factor=1.0,
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
            padding_idx=0,
            num_hidden_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            hidden_act="silu",
            attention=attention_config,
            tie_word_embeddings=False,
            training_dtype="bf16",
            chunked_loss=False,
            n_routed_experts=32,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            hidden_factor=1.0,
            moe_intermediate_size=256,  # grouped linear kernel need this to be multiple of 256
            router=router_config,
        )
        model = MoE(config=config).to(dtype).to(device)
        model.cuda()

        input_ids = torch.randint(
            0, config.vocab_size, (1, 128), dtype=torch.int64, device="cuda"
        )
        shift_input_ids = input_ids[:, :-1]
        shift_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))

        data_batch = [{'seq_ctx': seq_ctx, 'labels': shift_labels}]
        loss_ctx = CELossContext()
        data_batch = loss_ctx.build_list_ctx(data_batch, grad_accumulation_steps=1)[0]
        model(
            seq_ctx=data_batch['seq_ctx'],
            loss_ctx=data_batch['loss_ctx'],
        )


class TestDistributedMoE(DistributedTestBase):
    @parametrize.parametrize(
        "dtype,device,dispatcher,n_shared_experts,first_k_dense_replace",
        [
            (torch.bfloat16, "cuda", "deepep", 1, 2),
            (torch.bfloat16, "cuda", "all2all", 1, 2),
            (torch.bfloat16, "cuda", "all2all", 0, 0),
        ],
    )
    def test_parralel_accuracy(self, dtype, device, dispatcher, n_shared_experts, first_k_dense_replace):
        self.create_pg(device)
        router_config = NoAuxRouterConfig(
            scoring_func="sigmoid",
            router_scaling_factor=1.0,
            routed_scaling_factor=1.0,
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
            padding_idx=0,
            num_hidden_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            hidden_act="silu",
            attention=attention_config,
            tie_word_embeddings=False,
            training_dtype="bf16",
            chunked_loss=False,
            n_routed_experts=32,
            n_shared_experts=n_shared_experts,
            num_experts_per_tok=2,
            first_k_dense_replace=first_k_dense_replace,
            hidden_factor=1.0,
            moe_intermediate_size=256,  # grouped linear kernel need this to be multiple of 256
            router=router_config,
        )

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
        shift_labels = input_ids[:, 1:]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        data_batch = [{'seq_ctx': seq_ctx, 'labels': shift_labels}]
        loss_ctx = CELossContext()
        data_batch = loss_ctx.build_list_ctx(data_batch, grad_accumulation_steps=1)[0]

        loss_parallel = parallel_model(
            seq_ctx=data_batch['seq_ctx'],
            loss_ctx=data_batch['loss_ctx'],
        )["loss"]

        loss_expected = model(
            seq_ctx=data_batch['seq_ctx'],
            loss_ctx=data_batch['loss_ctx'],
        )["loss"]

        torch.allclose(loss_expected, loss_parallel, atol=1e-6, rtol=1e-4)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
