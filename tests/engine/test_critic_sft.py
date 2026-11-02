"""Critic 模型单测(Qwen3.5-VL compose 模型)。

覆盖的行为:
- TestCriticCustomHFKeyMapping(CPU,无需 GPU/权重)
    - test_loaded_mapped_critic_head_is_not_reinitialized: hf_key_mapping 映射的 value head strict 加载后保留 checkpoint 值
    - test_actor_to_critic_strict_load_initializes_head: actor checkpoint strict 加载进 critic,缺失头随机初始化
    - test_backbone_missing_still_strict_error: head 豁免不掩盖 backbone 缺失,strict 仍报错
- TestCriticSFT(8 卡 + QWEN3_5_MOE_PATH)
    - test_critic_rejects_mtp: value_head 配置拒绝 MTP
    - test_critic_qwen35_value_head_hf_keys: value head 落盘为顶层 value_head.weight(无 language_model. 前缀)
    - test_critic_accepts_xtuner_actor_checkpoint_head_metadata: checkpoint head 元数据的角色校验(actor→critic 放行,critic→actor 拒绝)
    - test_critic_sft_loss: SFT MSE 路径,从 build_loss_ctx_batch 公开入口进入的 forward/backward 与数值
    - test_critic_clipped_value_loss: PPO clipped 路径,同上
    - test_critic_load_actor_and_save_hf: actor 权重加载 → save_hf → strict 重载一致性
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from pydantic import Field
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.critic_loss import CriticLossConfig
from xtuner.v1.model.base import BaseModel, HeadConfig, XTunerBaseModelConfig
from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.head import ValueHead
from xtuner.v1.module.mtp.config import MTPConfig
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
CLIPRANGE_VALUE = 0.2
QWEN3_5_MOE_PATH = os.environ.get("QWEN3_5_MOE_PATH")


class _MappedCriticConfig(XTunerBaseModelConfig):
    hidden_size: int = 4
    head_config: HeadConfig = Field(default_factory=lambda: HeadConfig(type="value_head"))
    hf_key_mapping: dict[str, str] | None = {r"^value_head\.weight$": "critic.weight"}


class _MappedCriticModel(BaseModel):
    def __init__(self, config: _MappedCriticConfig):
        super().__init__(config)
        self.backbone = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lm_head = torch.nn.Linear(config.hidden_size, 1, bias=False)
        self._init_load_spec()

    def _model_hf_keys(self, key: str) -> list[str]:
        return [key]


def _write_mapped_ckpt(tmpdir: str, tensors: dict) -> None:
    index = {"weight_map": {key: "model.safetensors" for key in tensors}}
    save_file(tensors, str(Path(tmpdir) / "model.safetensors"))
    (Path(tmpdir) / "model.safetensors.index.json").write_text(json.dumps(index))


def _critic_model_cfg() -> Qwen3_5_VLMoE35BA3Config:
    model_cfg = Qwen3_5_VLMoE35BA3Config(
        freeze_vision=True,
        freeze_projector=True,
        only_llm_forward=True,
        compile_cfg=False,
    )
    model_cfg.text_config.head_config.type = "value_head"
    return model_cfg


class TestCriticCustomHFKeyMapping(unittest.TestCase):
    """Critic head 的 HF key 映射与 strict 加载契约(自定义 hf_key_mapping,无需 GPU)。"""

    def test_loaded_mapped_critic_head_is_not_reinitialized(self):
        # 经 hf_key_mapping 映射的 value head 在 strict 加载后必须保留 checkpoint 值,不被随机初始化覆盖
        model = _MappedCriticModel(_MappedCriticConfig())
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

        # 公开 to_hf_key_list 已包含 critic 改名(hf_key_mapping 仍在 load spec 阶段单独应用)
        self.assertEqual(model.to_hf_key_list("lm_head.weight"), ["value_head.weight"])

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_mapped_ckpt(
                tmpdir,
                {"critic.weight": expected, "backbone.weight": torch.zeros(4, 4)},
            )
            loaded_keys, unloaded_keys, missing_keys = model.from_hf(tmpdir, strict=True)

        self.assertIn("lm_head.weight", loaded_keys)
        self.assertIn("backbone.weight", loaded_keys)
        self.assertNotIn("lm_head.weight", unloaded_keys)
        self.assertNotIn("critic.weight", missing_keys)
        torch.testing.assert_close(model.lm_head.weight, expected)

    def test_actor_to_critic_strict_load_initializes_head(self):
        # actor checkpoint strict 加载进 critic:缺失的 value head 被随机初始化,backbone 正常加载,key 对账正确
        model = _MappedCriticModel(_MappedCriticConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_mapped_ckpt(
                tmpdir,
                {"backbone.weight": torch.ones(4, 4), "lm_head.weight": torch.zeros(4, 4)},
            )
            loaded_keys, unloaded_keys, missing_keys = model.from_hf(tmpdir, strict=True)

        self.assertIn("backbone.weight", loaded_keys)
        self.assertIn("lm_head.weight", loaded_keys)
        self.assertNotIn("lm_head.weight", unloaded_keys)
        self.assertNotIn("critic.weight", missing_keys)
        self.assertNotEqual(model.lm_head.weight.abs().sum().item(), 0.0)

    def test_backbone_missing_still_strict_error(self):
        # head 豁免不得掩盖 backbone 缺失:strict 加载仍必须报错
        model = _MappedCriticModel(_MappedCriticConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_mapped_ckpt(tmpdir, {"lm_head.weight": torch.zeros(4, 4)})
            with self.assertRaisesRegex(RuntimeError, "Missing keys in checkpoint"):
                model.from_hf(tmpdir, strict=True)


def _mse_value_loss(
    vpred: torch.Tensor,
    returns: torch.Tensor,
    loss_weight: torch.Tensor,
) -> torch.Tensor:
    """Hand-computed SFT critic loss used to document / check CriticLossContext.

    Args:
        vpred (torch.Tensor): Current value predictions, [B, S, 1].
        returns (torch.Tensor): Per-token regression targets.
        loss_weight (torch.Tensor): Normalized per-token weights.

    Returns:
        torch.Tensor: Scalar sum_t w_t * 0.5 * (v - R)^2.
    """
    v = vpred.squeeze(-1)  # drop the head output dim only; it is always 1
    returns = returns.reshape_as(v)
    loss_weight = loss_weight.reshape_as(v)
    per_token = 0.5 * (v - returns) ** 2
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
        vpred (torch.Tensor): Current value predictions, [B, S, 1].
        old_values (torch.Tensor): Frozen values used for clipping.
        returns (torch.Tensor): Per-token regression targets.
        loss_weight (torch.Tensor): Normalized per-token weights.
        cliprange_value (float): Clip range eps.

    Returns:
        torch.Tensor: Scalar sum_t w_t * 0.5 * max((v-R)^2, (v_clip-R)^2).
    """
    v = vpred.squeeze(-1)  # drop the head output dim only; it is always 1
    old_values = old_values.reshape_as(v)
    returns = returns.reshape_as(v)
    loss_weight = loss_weight.reshape_as(v)
    v_clip = old_values + (v - old_values).clamp(-cliprange_value, cliprange_value)
    per_token = 0.5 * torch.maximum((v - returns) ** 2, (v_clip - returns) ** 2)
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
            fp32_value_head=True,
            recompute_ratio=0.0,
            ep_size=ep_size,
        ),
    )


class TestCriticSFT(DeterministicDDPTestCase):
    def test_critic_rejects_mtp(self):
        pg = self.create_pg(str(DEVICE))
        with self.assertRaisesRegex(ValueError, "does not support MTP"):
            Qwen3_5_VLTextMoE35BA3BConfig(head_config=HeadConfig(type="value_head"), mtp_config=MTPConfig(num_layers=1))
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
            model.language_model.to_hf_key_list("lm_head.weight"),
            ["value_head.weight"],
        )
        try:
            dist.destroy_process_group(pg)
        except (AssertionError, ValueError):
            pass

    def test_head_checkpoint_role_inferred_from_keys(self):
        # 角色由 checkpoint 实际 keys 推断:含 value_head 权重的 ckpt 不能加载进 actor;
        # actor ckpt(无 value_head 权重)初始化 critic 放行
        pg = self.create_pg(str(DEVICE))
        with torch.device("meta"):
            model = _critic_model_cfg().build()

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "model.safetensors.index.json"
            # actor checkpoint:无 value_head 权重,critic 模型放行(头稍后随机初始化)
            index_path.write_text(json.dumps({"weight_map": {"lm_head.weight": "model.safetensors"}}))
            model.language_model._validate_head_checkpoint(Path(tmpdir))

            # critic checkpoint:含 value_head 权重,actor 模型必须拒绝(没有词表头)
            index_path.write_text(json.dumps({"weight_map": {"value_head.weight": "model.safetensors"}}))
            model.language_model.config.head_config.type = "lm_head"
            with self.assertRaisesRegex(RuntimeError, "Cannot load a checkpoint containing value_head"):
                model.language_model._validate_head_checkpoint(Path(tmpdir))

        try:
            dist.destroy_process_group(pg)
        except (AssertionError, ValueError):
            pass

    def _make_batch(self, model_cfg: Qwen3_5_VLMoE35BA3Config, ignore_idx: int = -100):
        seq_len = 32
        prompt_len = 8
        vocab_size = model_cfg.text_config.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=DEVICE)
        shifted_labels = input_ids.clone()
        shifted_labels[:, :prompt_len] = ignore_idx
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
        # SFT MSE 路径:只有 returns 时,经模型公开入口组装并前向/反向,loss 与手算 oracle 一致
        """SFT path: only returns is provided, loss is unclipped MSE."""
        pg = self.create_pg(str(DEVICE))
        torch.manual_seed(0)

        model_cfg = _critic_model_cfg()
        model_cfg.text_config.lm_loss_cfg = CriticLossConfig()
        engine = _critic_engine(model_cfg)
        engine.from_hf(QWEN3_5_MOE_PATH, strict=True)

        lm_head = _text_module(engine.model).lm_head
        head_weight = _full_tensor(lm_head.weight)
        self.assertEqual(tuple(head_weight.shape), (1, model_cfg.text_config.hidden_size))
        self.assertEqual(lm_head.out_features, 1)
        self.assertIsInstance(lm_head, ValueHead)

        seq_ctx, shifted_labels, returns = self._make_batch(model_cfg)
        engine.model.train()
        # Go through the public batch entry so the model-side call protocol
        # (cu_seq_lens_list / sp_mesh keywords) stays covered by CI.
        loss_ctx_batch = engine.model.build_loss_ctx_batch(
            [{"seq_ctx": seq_ctx, "shifted_labels": shifted_labels, "returns": returns}], None
        )
        loss_ctx = loss_ctx_batch[0]["lm"]
        assert loss_ctx is not None
        self.assertIsNone(loss_ctx.loss_kwargs.old_values)
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
        # PPO critic 路径:提供 old_values 时,经模型公开入口组装,clipped loss 与手算 oracle 一致且区别于 MSE
        """PPO critic path: old_values is provided, loss is clipped value loss."""
        pg = self.create_pg(str(DEVICE))
        torch.manual_seed(0)

        model_cfg = _critic_model_cfg()
        model_cfg.text_config.lm_loss_cfg = CriticLossConfig(cliprange_value=CLIPRANGE_VALUE)
        engine = _critic_engine(model_cfg)
        engine.from_hf(QWEN3_5_MOE_PATH, strict=True)

        seq_ctx, shifted_labels, returns = self._make_batch(model_cfg)
        engine.model.train()
        with torch.no_grad():
            freeze_out = engine.model(seq_ctx=seq_ctx, loss_ctx=None)
            assert freeze_out.logits is not None
            # v_old is shifted away from R=1 so the clipped term is the larger square.
            old_values = freeze_out.logits.float().squeeze(-1).detach() - 1.0

        # Go through the public batch entry so the model-side call protocol
        # (cu_seq_lens_list / sp_mesh keywords) stays covered by CI.
        loss_ctx_batch = engine.model.build_loss_ctx_batch(
            [
                {
                    "seq_ctx": seq_ctx,
                    "shifted_labels": shifted_labels,
                    "returns": returns,
                    "old_values": old_values,
                }
            ],
            None,
        )
        loss_ctx = loss_ctx_batch[0]["lm"]
        assert loss_ctx is not None
        self.assertIsNotNone(loss_ctx.loss_kwargs.old_values)
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
        engine.from_hf(QWEN3_5_MOE_PATH, strict=True)

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
                self.assertNotIn("metadata", index)
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
