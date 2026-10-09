"""EP load-imbalance metrics reported by ``MoE.post_micro_batch_forward``.

Routing is pinned per layer through the NoAux router's selection bias, so every rank's received rows are known
exactly. World = 4 GPUs = 2 independent EP groups of size 2 (experts 0,1 on ep_rank 0; experts 2,3 on ep_rank 1).
Layer 0 is activation-checkpointed and replayed in backward, so a double count shows up in the expected values.
"""

import parametrize
import torch

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine, TrainStepInfo
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import NoAuxRouterConfig


_EP_SIZE = 2
_SEQ_LEN = 32
_VOCAB_SIZE = 128

# Experts picked by every token, per layer.
# Aligned-then-balanced: ep_rank 0 hosts both experts of layer 0, layer 1 spreads one copy to each rank.
_ALIGNED_THEN_BALANCED = ((0, 1), (0, 2))
# Opposite hot ranks: each rank is flooded in one layer, so step sums look balanced but every layer waits.
_OPPOSITE_HOT_RANKS = ((0, 1), (2, 3))


class TestEPLoadMetrics(DeterministicDDPTestCase):
    @parametrize.parametrize(
        "routing,intra_layer_micro_batch,expected",
        [
            # load = ep_rank 0: (2 + 1) / 2, ep_rank 1: (0 + 1) / 2; peak = layer 0; straggler = (2 + 1) / 2.
            (_ALIGNED_THEN_BALANCED, 1, {"load": (1.5, 0.5), "peak": (2.0, 1.0), "straggler": 1.5}),
            (_ALIGNED_THEN_BALANCED, 2, {"load": (1.5, 0.5), "peak": (2.0, 1.0), "straggler": 1.5}),
            # load = (2 + 0) / 2 on both ranks, yet each layer waits for a rank holding twice its share.
            (_OPPOSITE_HOT_RANKS, 1, {"load": (1.0, 1.0), "peak": (2.0, 2.0), "straggler": 2.0}),
        ],
    )
    def test_ratios_match_pinned_routing(
        self,
        routing: tuple[tuple[int, int], ...],
        intra_layer_micro_batch: int,
        expected: dict,
    ) -> None:
        self.create_pg("cuda")
        engine = _build_engine(n_layers=len(routing), intra_layer_micro_batch=intra_layer_micro_batch)
        _pin_routing(engine, routing)

        logs_info = _train_step(engine, n_micro_batches=2)["logs_info"]

        ep_rank = self.rank % _EP_SIZE
        assert logs_info["ep_load_ratio"] == expected["load"][ep_rank]
        assert logs_info["ep_load_peak_ratio"] == expected["peak"][ep_rank]
        assert logs_info["ep_load_ratio_max"] == max(expected["load"])
        assert logs_info["ep_load_peak_ratio_max"] == max(expected["peak"])
        # argmax picks the first maximal global rank; ep_rank 0 of EP group 0 is global rank 0.
        assert logs_info["ep_load_ratio_max_rank"] == 0.0
        assert logs_info["ep_load_peak_ratio_max_rank"] == 0.0
        assert logs_info["ep_straggler_ratio"] == expected["straggler"]

    def test_forward_outside_train_step_is_not_counted(self) -> None:
        self.create_pg("cuda")
        engine = _build_engine(n_layers=2, intra_layer_micro_batch=1)
        _pin_routing(engine, _OPPOSITE_HOT_RANKS)
        # A forward-only pass (RL old-logprob style) under the aligned routing must not leak into the next step.
        with torch.no_grad():
            engine.model(**_make_items(n_micro_batches=1)[0])
        _pin_routing(engine, _ALIGNED_THEN_BALANCED)

        logs_info = _train_step(engine, n_micro_batches=2)["logs_info"]

        assert logs_info["ep_load_ratio"] == (1.5, 0.5)[self.rank % _EP_SIZE]

    @property
    def world_size(self) -> int:
        return 4


def _build_engine(n_layers: int, intra_layer_micro_batch: int) -> TrainEngine:
    model_cfg = Qwen3MoEConfig(
        vocab_size=_VOCAB_SIZE,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_hidden_layers=n_layers,
        hidden_size=128,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=4, num_key_value_heads=2, head_dim=32, qk_norm=True),
        tie_word_embeddings=False,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=64,
        router=NoAuxRouterConfig(
            scoring_func="sigmoid", router_scaling_factor=1.0, n_group=1, topk_group=1, norm_topk_prob=True
        ),
        ep_size=_EP_SIZE,
        dispatcher="all2all",
        compile_cfg=False,
        balancing_loss_cfg=None,
        z_loss_cfg=None,
    )
    engine = TrainEngine(
        model_cfg=model_cfg,
        optim_cfg=AdamWConfig(),
        # recompute_ratio=1.0 checkpoints every layer but the last one.
        fsdp_cfg=FSDPConfig(ep_size=_EP_SIZE, recompute_ratio=1.0, cpu_offload=False),
        intra_layer_micro_batch=intra_layer_micro_batch,
    )
    engine.init_model_weights()
    return engine


def _pin_routing(engine: TrainEngine, routing: tuple[tuple[int, int], ...]) -> None:
    # Sigmoid scores lie in (0, 1), so a +10 selection bias makes the pinned experts every token's top-k.
    for name, buffer in engine.model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            layer_idx = int(name.split("layers.")[1].split(".")[0])
            buffer.zero_()
            buffer[list(routing[layer_idx])] = 10.0


def _make_items(n_micro_batches: int) -> list[ModelItem]:
    loss_cfg = CELossConfig()
    seq_ctx_list, loss_ctx_list = [], []
    for _ in range(n_micro_batches):
        ids = torch.randint(0, _VOCAB_SIZE, (1, _SEQ_LEN), device="cuda")
        seq_ctx_list.append(SequenceContext.from_input_ids((ids[:, :-1],), device="cuda"))
        loss_ctx_list.append(loss_cfg.build(data={"shifted_labels": ids[:, 1:]}, sp_mesh=None))
    loss_ctx_list = loss_cfg.loss_ctx_cls.build_batches(loss_ctx_list)
    return [
        ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx}) for seq_ctx, loss_ctx in zip(seq_ctx_list, loss_ctx_list)
    ]


def _train_step(engine: TrainEngine, n_micro_batches: int) -> TrainStepInfo:
    return engine.train_step(_make_items(n_micro_batches))
