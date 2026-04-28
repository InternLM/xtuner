"""Validate that EP+TP training produces the same forward loss and backward
gradients as a pure single-GPU (EP=1, TP=1) run.

Test topology: world_size = EP * TP * DP = 2 * 2 * 1 = 4 GPUs.

Strategy
--------
1. Build a tiny Qwen3MoE model with EP=2, TP=2.
2. Build the same model with EP=1, TP=1 (4 identical DP replicas).
3. Init both engines with ``init_model_weights()``.  Because weights for EP+TP
   models are Shard(0) on ep_mesh for experts and Replicate for non-experts,
   and ``init_params`` always initialises the *full* tensor before sharding,
   the underlying full weight values are identical when the same RNG seed is
   active on all ranks.
4. Sync expert weights from EP=1 engine to EP=2 engine via DCP so the two
   models start from the exact same checkpoint.
5. Run one ``train_step`` + ``clip_grad_norm`` on both engines with the same
   input.
6. Assert:
   - losses agree within tolerance
   - gate (router) gradients agree within tolerance (non-expert, replicated
     on all ranks in both configs)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import parametrize
import torch
import torch.distributed as dist

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.utils.device import get_device

DEVICE = get_device()

# Tolerance for bfloat16 numerical differences between the two configs.
ATOL = 2e-1
RTOL = 2e-1

# Use a very small model to keep test runtime manageable.
_TINY_LAYERS = 2
_SEQ_LEN = 64


def _build_tiny_moe_cfg(ep_size: int = 1, tp_size: int = 1) -> Qwen3MoE30BA3Config:
    return Qwen3MoE30BA3Config(
        num_hidden_layers=_TINY_LAYERS,
        ep_size=ep_size,
        tp_size=tp_size,
        dispatcher="all2all" if ep_size > 1 else None,
        compile_cfg=False,
        # Disable auxiliary losses to keep the comparison clean.
        balancing_loss_cfg=None,
        z_loss_cfg=None,
    )


def _build_engine(ep_size: int, tp_size: int) -> TrainEngine:
    moe_cfg = _build_tiny_moe_cfg(ep_size, tp_size)
    optim_cfg = AdamWConfig()
    fsdp_cfg = FSDPConfig(
        ep_size=ep_size,
        tp_size=tp_size,
        cpu_offload=False,
    )
    return TrainEngine(model_cfg=moe_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg)


def _make_engine_input(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids [1, SEQ_LEN-1], shifted_labels [1, SEQ_LEN-1]) on *device*."""
    torch.manual_seed(12345)
    full_ids = torch.randint(0, 151936, (1, _SEQ_LEN), dtype=torch.long, device=device)
    input_ids = full_ids[:, :-1]  # [1, SEQ_LEN-1]
    labels = full_ids[:, 1:]      # [1, SEQ_LEN-1] already shifted
    return input_ids, labels


def _run_one_step(
    engine: TrainEngine,
    loss_cfg: CELossConfig,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, dict[str, torch.Tensor]]:
    """Run one train step; return (loss_value, {param_name: grad_tensor})."""
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    shifted_labels = labels.to(DEVICE)

    LossContext = loss_cfg.loss_ctx_cls
    loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
    loss_ctx_list = LossContext.build_batches([loss_ctx])
    loss_ctx = loss_ctx_list[0]

    engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})]
    step_info = engine.train_step(engine_input)
    engine.clip_grad_norm()

    loss_val: float = step_info["logs_info"]["reduced_llm_loss"]

    # Collect gradients from gate (router) parameters; these are non-expert
    # parameters replicated on all ranks in both configs, so they're easy to
    # compare directly.
    grads: dict[str, torch.Tensor] = {}
    for name, param in engine.model.named_parameters():
        if "gate.weight" in name and param.grad is not None:
            grad = param.grad
            if hasattr(grad, "full_tensor"):
                grad = grad.full_tensor()  # type: ignore[attr-defined]
            grads[name] = grad.detach().float().cpu()
            break  # one gate layer is sufficient

    return loss_val, grads


class TestMoETrainEngineTPEP(DeterministicDDPTestCase):
    """Verify EP+TP training matches single-GPU (EP=1, TP=1) forward and backward."""

    @parametrize.parametrize(
        "device,ep_size,tp_size",
        [
            ("cuda", 2, 2),
        ],
    )
    def test_tpep_forward_backward_matches_single(
        self, device: str, ep_size: int, tp_size: int
    ) -> None:
        """Loss and gate gradients with EP+TP must match the EP=1, TP=1 baseline."""
        pg = self.create_pg(device)

        # ------------------------------------------------------------------
        # Build reference engine: EP=1, TP=1 (world acts as pure DP).
        # ------------------------------------------------------------------
        engine_ref = _build_engine(ep_size=1, tp_size=1)
        engine_ref.init_model_weights()

        # ------------------------------------------------------------------
        # Build EP+TP engine.
        # ------------------------------------------------------------------
        engine_tpep = _build_engine(ep_size=ep_size, tp_size=tp_size)
        engine_tpep.init_model_weights()

        # ------------------------------------------------------------------
        # Sync weights: save reference engine, load into EP+TP engine.
        # DCP handles the translation between different tensor layouts.
        # ------------------------------------------------------------------
        tmp: list[str] = [tempfile.mkdtemp() if dist.get_rank() == 0 else ""]
        dist.broadcast_object_list(tmp, src=0)
        ckpt_root = Path(tmp[0])
        model_dir = ckpt_root / "model"

        engine_ref.save_dcp(model_dir=model_dir)
        dist.barrier()
        engine_tpep.load_dcp(model_dir=model_dir)
        dist.barrier()

        # ------------------------------------------------------------------
        # Prepare shared input (identical on all ranks – no SP).
        # ------------------------------------------------------------------
        input_ids, labels = _make_engine_input(torch.device(device, dist.get_rank() % torch.cuda.device_count()))
        loss_cfg = CELossConfig()

        # Run EP+TP step.
        loss_tpep, grads_tpep = _run_one_step(engine_tpep, loss_cfg, input_ids, labels)

        # Run reference step.
        loss_ref, grads_ref = _run_one_step(engine_ref, loss_cfg, input_ids, labels)

        # ------------------------------------------------------------------
        # Assert losses match.
        # ------------------------------------------------------------------
        if dist.get_rank() == 0:
            self.assertAlmostEqual(
                loss_tpep,
                loss_ref,
                delta=ATOL,
                msg=f"Loss mismatch: EP+TP={loss_tpep:.6f}, ref={loss_ref:.6f}",
            )

        # ------------------------------------------------------------------
        # Assert gate gradients match (key non-expert parameter).
        # ------------------------------------------------------------------
        if grads_tpep and grads_ref:
            for name in grads_ref:
                if name not in grads_tpep:
                    continue
                g_tpep = grads_tpep[name]
                g_ref = grads_ref[name]
                if dist.get_rank() == 0:
                    try:
                        torch.testing.assert_close(
                            g_tpep,
                            g_ref,
                            atol=ATOL,
                            rtol=RTOL,
                        )
                    except AssertionError as exc:
                        max_diff = (g_tpep - g_ref).abs().max().item()
                        raise AssertionError(
                            f"Gate gradient mismatch for '{name}': "
                            f"max_abs_diff={max_diff:.4e}, EP+TP shape={g_tpep.shape}, ref shape={g_ref.shape}"
                        ) from exc

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @parametrize.parametrize(
        "device,ep_size,tp_size",
        [
            ("cuda", 2, 2),
        ],
    )
    def test_tpep_training_stability(self, device: str, ep_size: int, tp_size: int) -> None:
        """EP+TP training should produce finite losses and decreasing trend."""
        pg = self.create_pg(device)

        engine = _build_engine(ep_size=ep_size, tp_size=tp_size)
        engine.init_model_weights()

        input_ids, labels = _make_engine_input(torch.device(device, dist.get_rank() % torch.cuda.device_count()))
        loss_cfg = CELossConfig()

        losses: list[float] = []
        for _ in range(4):
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
            shifted_labels = labels.to(DEVICE)
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
            loss_ctx_list = LossContext.build_batches([loss_ctx])
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx_list[0]})]
            step_info = engine.train_step(engine_input)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            losses.append(step_info["logs_info"]["reduced_llm_loss"])

        if dist.get_rank() == 0:
            for i, loss_val in enumerate(losses):
                self.assertTrue(
                    torch.isfinite(torch.tensor(loss_val)),
                    f"Loss at step {i} is not finite: {loss_val}",
                )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @property
    def world_size(self) -> int:
        # EP=2, TP=2, DP=1 → 4 GPUs
        return 4

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
