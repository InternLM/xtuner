"""Critic model unit tests on Qwen3.5-VL (compose) models.

Coverage:
- value_head models reject MTP
- Qwen3.5-VL value-head HF keys are top-level (value_head.weight, no language_model. prefix)
- SFT value loss (returns only): forward, loss value, and backward
- PPO critic value loss (returns + old_values): forward, loss value, and backward
- load a Qwen3.5-VL actor HF checkpoint into a critic, save_hf, and reload
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.head import ValueHead
from xtuner.v1.module.mtp.config import MTPConfig
from xtuner.v1.rl.loss.critic_loss import CriticLossConfig
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
CLIPRANGE_VALUE = 0.2
QWEN3_5_MOE_PATH = os.environ.get("QWEN3_5_MOE_PATH")


def _critic_model_cfg() -> Qwen3_5_VLMoE35BA3Config:
    model_cfg = Qwen3_5_VLMoE35BA3Config(
        freeze_vision=True,
        freeze_projector=True,
        only_llm_forward=True,
        compile_cfg=False,
    )
    model_cfg.text_config.head_type = "value_head"
    return model_cfg


def _mse_value_loss(
    vpred: torch.Tensor,
    returns: torch.Tensor,
    loss_weight: torch.Tensor,
) -> torch.Tensor:
    """Hand-computed SFT critic loss used to document / check CriticLossContext.

    Args:
        vpred (torch.Tensor): Current value predictions.
        returns (torch.Tensor): Per-token regression targets.
        loss_weight (torch.Tensor): Normalized per-token weights.

    Returns:
        torch.Tensor: Scalar sum_t w_t * 0.5 * (v - R)^2.
    """
    vpred = vpred.squeeze(-1) if vpred.dim() > 0 and vpred.size(-1) == 1 else vpred
    returns = returns.squeeze(-1) if returns.dim() > 0 and returns.size(-1) == 1 else returns
    loss_weight = loss_weight.squeeze(-1) if loss_weight.dim() > 0 and loss_weight.size(-1) == 1 else loss_weight
    per_token = 0.5 * (vpred - returns) ** 2
    return (per_token * loss_weight).sum()


def _clipped_value_loss(
    vpred: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    loss_weight: torch.Tensor,
    cliprange_value: float,
) -> torch.Tensor:
    """Hand-computed PPO critic loss used to document / check CriticLossContext.

    Args:
        vpred (torch.Tensor): Current value predictions.
        old_values (torch.Tensor): Frozen values used for clipping.
        returns (torch.Tensor): Per-token regression targets.
        loss_weight (torch.Tensor): Normalized per-token weights.
        cliprange_value (float): Clip range eps.

    Returns:
        torch.Tensor: Scalar sum_t w_t * 0.5 * max((v-R)^2, (v_clip-R)^2).
    """
    vpred = vpred.squeeze(-1) if vpred.dim() > 0 and vpred.size(-1) == 1 else vpred
    old_values = old_values.squeeze(-1) if old_values.dim() > 0 and old_values.size(-1) == 1 else old_values
    returns = returns.squeeze(-1) if returns.dim() > 0 and returns.size(-1) == 1 else returns
    loss_weight = loss_weight.squeeze(-1) if loss_weight.dim() > 0 and loss_weight.size(-1) == 1 else loss_weight
    v_clip = old_values + (vpred - old_values).clamp(-cliprange_value, cliprange_value)
    per_token = 0.5 * torch.maximum((vpred - returns) ** 2, (v_clip - returns) ** 2)
    return (per_token * loss_weight).sum()


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


def _local_bf16(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().to(torch.bfloat16).contiguous().cpu()


def _text_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.language_model


def _critic_engine(model_cfg: Qwen3_5_VLMoE35BA3Config) -> TrainEngine:
    ep_size = dist.get_world_size()
    model_cfg.text_config.ep_size = ep_size
    return TrainEngine(
        model_cfg=model_cfg,
        optim_cfg=AdamWConfig(lr=1e-4, foreach=False),
        fsdp_cfg=FSDPConfig(
            cpu_offload=False,
            torch_compile=False,
            fp32_head=True,
            recompute_ratio=0.0,
            ep_size=ep_size,
        ),
    )


class TestCriticSFT(DeterministicDDPTestCase):
    def test_critic_rejects_mtp(self):
        pg = self.create_pg(str(DEVICE))
        with self.assertRaisesRegex(ValueError, "does not support MTP"):
            Qwen3_5_VLTextMoE35BA3BConfig(head_type="value_head", mtp_config=MTPConfig(num_layers=1))
        try:
            dist.destroy_process_group(pg)
        except (AssertionError, ValueError):
            pass

    def test_critic_qwen35_value_head_hf_keys(self):
        """Qwen3.5-VL maps runtime lm_head.weight to top-level value_head.weight."""
        pg = self.create_pg(str(DEVICE))
        with torch.device("meta"):
            model = _critic_model_cfg().build()
        self.assertEqual(
            model.language_model._to_hf_key_list("lm_head.weight"),
            ["value_head.weight"],
        )
        try:
            dist.destroy_process_group(pg)
        except (AssertionError, ValueError):
            pass

    def _make_batch(self, model_cfg: Qwen3_5_VLMoE35BA3Config, loss_cfg: CriticLossConfig):
        seq_len = 32
        prompt_len = 8
        vocab_size = model_cfg.text_config.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=DEVICE)
        shifted_labels = input_ids.clone()
        shifted_labels[:, :prompt_len] = loss_cfg.ignore_idx
        returns = torch.zeros(1, seq_len, device=DEVICE, dtype=torch.float32)
        returns[:, prompt_len:] = 1.0
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device=str(DEVICE))
        return seq_ctx, shifted_labels, returns

    def _assert_loss_matches(self, actual: torch.Tensor, expected_local: torch.Tensor, name: str):
        expected = expected_local.clone()
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)
        self.assertTrue(
            abs(actual.item() - expected.item()) < 1e-4,
            f"{name} mismatch: got {actual.item()}, expected {expected.item()}",
        )

    def _assert_head_backward(self, engine: TrainEngine, loss: torch.Tensor, *, require_nonzero: bool = True):
        engine.optimizer.zero_grad()
        loss.backward()
        head_grad = _text_module(engine.model).lm_head.weight.grad
        self.assertIsNotNone(head_grad)
        assert head_grad is not None
        head_grad = _full_tensor(head_grad)
        self.assertTrue(torch.isfinite(head_grad).all())
        if require_nonzero:
            self.assertTrue(head_grad.abs().sum().item() > 0)

    @unittest.skipUnless(QWEN3_5_MOE_PATH, "QWEN3_5_MOE_PATH is not set")
    def test_critic_sft_loss(self):
        """SFT path: only returns is provided, loss is unclipped MSE."""
        pg = self.create_pg(str(DEVICE))
        torch.manual_seed(0)

        model_cfg = _critic_model_cfg()
        loss_cfg = CriticLossConfig()
        engine = _critic_engine(model_cfg)
        engine.from_hf(QWEN3_5_MOE_PATH, strict=False)

        lm_head = _text_module(engine.model).lm_head
        head_weight = _full_tensor(lm_head.weight)
        self.assertEqual(tuple(head_weight.shape), (1, model_cfg.text_config.hidden_size))
        self.assertEqual(lm_head.out_features, 1)
        self.assertIsInstance(lm_head, ValueHead)

        seq_ctx, shifted_labels, returns = self._make_batch(model_cfg, loss_cfg)
        engine.model.train()
        loss_ctx = loss_cfg.build(
            data={"shifted_labels": shifted_labels, "returns": returns},
            sp_mesh=None,
        )
        assert loss_ctx is not None
        self.assertIsNone(loss_ctx.loss_kwargs.old_values)
        loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
        out = engine.model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        assert out.loss is not None
        assert out.logits is not None
        assert loss_ctx.loss_kwargs.loss_weight is not None
        expected_local = _mse_value_loss(
            out.logits.detach().float(),
            returns,
            loss_ctx.loss_kwargs.loss_weight,
        )
        self._assert_loss_matches(out.loss, expected_local, "sft mse loss")
        self._assert_head_backward(engine, out.loss)

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except (AssertionError, ValueError):
            pass

    @unittest.skipUnless(QWEN3_5_MOE_PATH, "QWEN3_5_MOE_PATH is not set")
    def test_critic_clipped_value_loss(self):
        """PPO critic path: old_values is provided, loss is clipped value loss."""
        pg = self.create_pg(str(DEVICE))
        torch.manual_seed(0)

        model_cfg = _critic_model_cfg()
        loss_cfg = CriticLossConfig(cliprange_value=CLIPRANGE_VALUE)
        engine = _critic_engine(model_cfg)
        engine.from_hf(QWEN3_5_MOE_PATH, strict=False)

        seq_ctx, shifted_labels, returns = self._make_batch(model_cfg, loss_cfg)
        engine.model.train()
        with torch.no_grad():
            freeze_out = engine.model(seq_ctx=seq_ctx, loss_ctx=None)
            assert freeze_out.logits is not None
            # v_old is shifted away from R=1 so the clipped term is the larger square.
            old_values = freeze_out.logits.float().squeeze(-1).detach() - 1.0

        loss_ctx = loss_cfg.build(
            data={
                "shifted_labels": shifted_labels,
                "returns": returns,
                "old_values": old_values,
            },
            sp_mesh=None,
        )
        assert loss_ctx is not None
        self.assertIsNotNone(loss_ctx.loss_kwargs.old_values)
        loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
        out = engine.model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        assert out.loss is not None
        assert out.logits is not None
        assert loss_ctx.loss_kwargs.loss_weight is not None

        mse_local = _mse_value_loss(
            out.logits.detach().float(),
            returns,
            loss_ctx.loss_kwargs.loss_weight,
        )
        expected_local = _clipped_value_loss(
            out.logits.detach().float(),
            old_values,
            returns,
            loss_ctx.loss_kwargs.loss_weight,
            CLIPRANGE_VALUE,
        )
        self.assertGreater(
            abs(expected_local.item() - mse_local.item()),
            1e-6,
            "clipped value loss should differ from SFT MSE when clip is active",
        )
        self._assert_loss_matches(out.loss, expected_local, "clipped value loss")
        # Saturated clip can zero dL/dv; only require a finite backward.
        self._assert_head_backward(engine, out.loss, require_nonzero=False)

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except (AssertionError, ValueError):
            pass

    @unittest.skipUnless(QWEN3_5_MOE_PATH, "QWEN3_5_MOE_PATH is not set")
    def test_critic_load_actor_and_save_hf(self):
        """Load Qwen3.5-VL actor HF weights into a critic, save to a temp dir, then reload."""
        pg = self.create_pg(str(DEVICE))

        model_cfg = _critic_model_cfg()
        engine = _critic_engine(model_cfg)
        engine.from_hf(QWEN3_5_MOE_PATH, strict=False)

        text = _text_module(engine.model)
        self.assertIsInstance(text.lm_head, ValueHead)
        self.assertEqual(text.lm_head.out_features, 1)
        self.assertEqual(tuple(_full_tensor(text.lm_head.weight).shape), (1, model_cfg.text_config.hidden_size))

        saved_head = _local_bf16(text.lm_head.weight)
        saved_embed = _local_bf16(text.embed_tokens.weight)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = [tmpdir]
            dist.broadcast_object_list(save_dir, src=0)
            save_path = Path(save_dir[0])
            engine.save_hf(str(save_path))
            dist.barrier()

            if dist.get_rank() == 0:
                index = json.loads((save_path / "model.safetensors.index.json").read_text())
                self.assertEqual(index["metadata"]["head_type"], "value_head")
                self.assertIn("value_head.weight", index["weight_map"])
                self.assertNotIn("lm_head.weight", index["weight_map"])
                self.assertNotIn("language_model.value_head.weight", index["weight_map"])
                self.assertNotIn("language_model.lm_head.weight", index["weight_map"])

            dist.barrier()
            del engine
            torch.cuda.empty_cache()

            restored = _critic_engine(_critic_model_cfg())
            restored.from_hf(save_path, strict=True)

            restored_text = _text_module(restored.model)
            restored_head = _local_bf16(restored_text.lm_head.weight)
            restored_embed = _local_bf16(restored_text.embed_tokens.weight)
            torch.testing.assert_close(restored_head, saved_head, atol=0, rtol=0)
            torch.testing.assert_close(restored_embed, saved_embed, atol=0, rtol=0)

            del restored
            torch.cuda.empty_cache()
            dist.barrier()  # keep tmpdir until all ranks finish from_hf

        try:
            dist.destroy_process_group(pg)
        except (AssertionError, ValueError):
            pass

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
