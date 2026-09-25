"""GPU gates for the decoupled EP/FSDP layout (``FSDPConfig.decouple_ep_fsdp``).

Every test trains the same tiny random Qwen3-MoE on the same token stream under several EP/FSDP
layouts and fails when the decoupled layout departs from the legacy one:

* ``TestDecoupledEpFsdpNumerics`` (L1 / L2): loss curves, total gradient norms, per-parameter
  gradient norms at step 0 (identical weights and data, so they catch a wrong expert-gradient scale
  that AdamW would hide from the loss curve; after the first update bf16 weight differences flip
  individual top-k routing decisions and router gradients legitimately differ by percents) and
  per-rank parameter memory, for ``efsdp == 1``, ``efsdp > 1`` and HSDP + EP.
* ``TestDecoupledEpFsdpCheckpoint`` (L3): bit-exact HF export right after ``from_hf``, DCP resume,
  HF export after resume and cross-layout DCP resharding.

The observed noise floor between layouts is ~1e-5 on losses and ≤ 2e-3 on per-parameter grad
norms (see ``docs/design/decouple_ep_fsdp.md`` §5); the gates sit 1-2 orders of magnitude above
it and orders of magnitude below any layout bug seen so far.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from xtuner._testing import DeterministicDDPTestCase
from xtuner._testing.decoupled_ep_fsdp import (
    LayoutMode,
    assert_hf_dirs_close,
    build_engine,
    build_hf_checkpoint,
    load_hf_dir,
    max_rel_diff,
    parse_mode,
    release,
    run_mode,
    train,
)
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig


SEQ_LEN = 2048
LR = 1e-4
# Loss curves of two layouts: max relative difference over all steps (noise ~2e-5).
LOSS_RTOL = 1e-3
# Total grad norms per step and per-parameter grad norms at step 0 (noise ≤ 2e-3 dense, ≤ 2e-5 expert).
GRAD_NORM_RTOL = 1e-2
# Per-rank parameter memory vs. the value implied by the layout (FSDP padding slack).
MEMORY_RTOL = 0.25
# HF exports: 1 bf16 ulp is 2**-7 relative; resumed vs. continuous differs by single ulps on a
# subset of elements, exports of different layouts by a few ulps.
HF_RESUME_RTOL, HF_RESUME_ATOL = 2**-6, 1e-3
HF_LAYOUT_RTOL, HF_LAYOUT_ATOL = 2**-6, 2e-3


class _SharedTmpDirMixin:
    """Temp directory created by rank 0 and shared with every rank of the process group."""

    def _make_shared_tmpdir(self) -> Path:
        holder = [tempfile.mkdtemp(prefix="decoupled_ep_fsdp_") if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(holder, src=0)
        assert holder[0] is not None
        return Path(holder[0])

    @staticmethod
    def _cleanup_shared_tmpdir(path: Path) -> None:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(path, ignore_errors=True)


def _build_tiny_checkpoint(root: Path) -> Path:
    hf_dir = root / "tiny_qwen3_moe_hf"
    if dist.get_rank() == 0:
        build_hf_checkpoint(hf_dir, seed=0, model_size="tiny")
    dist.barrier()
    return hf_dir


@unittest.skipUnless(torch.cuda.device_count() >= 8, "requires 8 CUDA devices")
class TestDecoupledEpFsdpNumerics(_SharedTmpDirMixin, DeterministicDDPTestCase):
    def test_ep8_decoupled_matches_legacy_ep8_and_ep1(self) -> None:
        # L1: `efsdp == 1`; dense params go from replicated over EP to sharded over all 8 ranks.
        results = self._run_layouts(("A:ep=1", "B:ep=8", "C:ep=8,decouple=1"), steps=20)
        self._assert_training_matches(results["C"], results["B"])
        self._assert_training_matches(results["C"], results["A"])
        # 中文注释：每卡参数显存断言。记 D / E 为 dense / expert 参数总量：
        #   B（legacy，ep=8）：mesh 为 (fsdp=1, ep=8)，dense 在 ep 维整份复制 → 每卡 D；expert 按 ep 分 8 份 → E/8。
        #   C（解耦，ep=8）：dense 在整个 dp_shard=8 上分片 → D/8；expert 按 ep×efsdp=8×1 分片 → E/8。
        # 所以 C 的 dense 应是 B 的 1/8，expert 相同。loss / 梯度对比区分不了"dense 真的分片了"和
        # "dense 仍在 EP 维复制"（两者数值完全一样），只有这条显存断言能证明解耦布局确实生效。
        self._assert_param_memory(results["C"], results["B"], dense_ratio=1 / 8, expert_ratio=1.0)

    def test_efsdp_and_hsdp_layouts_match_legacy_ep4(self) -> None:
        # L2 on 8 ranks: `efsdp == 2` (C4), HSDP + EP with `efsdp == 1` (H41) and `efsdp == 2` (H22).
        results = self._run_layouts(
            ("B4:ep=4", "C4:ep=4,decouple=1", "H41:ep=4,decouple=1,hsdp=4", "H22:ep=2,decouple=1,hsdp=4"),
            steps=10,
        )
        for name in ("C4", "H41", "H22"):
            self._assert_training_matches(results[name], results["B4"])
        # Per-rank shards. Legacy ep=4 on 8 ranks: dense over fsdp=2 (D/2), experts over ep=4 then
        # fsdp=2 (E/8). Decoupled: dense over dp_shard (D/dp_shard), experts over ep * efsdp ==
        # dp_shard (E/dp_shard); dp_shard is 8 for C4 and 4 for the HSDP layouts H41 / H22.
        #
        # 中文注释：ratio = 被测布局的每卡参数量 / 参考布局 B4 的每卡参数量，按各布局的分片数推导：
        #   B4 （legacy，ep=4）      ：mesh (fsdp=2, ep=4)；dense 在 ep 维复制、在 fsdp 维分 2 份 → D/2；
        #                              expert 先按 ep 分 4 份、再按 fsdp 分 2 份 → E/8。
        #   C4 （解耦，ep=4）        ：dp_shard=8，efsdp=8/4=2；dense → D/8；expert 按 ep×efsdp=8 → E/8。
        #   H41（解耦，ep=4，hsdp=4）：dp_shard=4，replicate=2，efsdp=1；dense → D/4；expert 按 4×1 → E/4。
        #   H22（解耦，ep=2，hsdp=4）：dp_shard=4，replicate=2，efsdp=2；dense → D/4；expert 按 2×2 → E/4。
        # 于是 C4 vs B4：dense (D/8)/(D/2)=1/4，expert (E/8)/(E/8)=1；
        #      H41、H22 vs B4：dense (D/4)/(D/2)=1/2，expert (E/4)/(E/8)=2。
        # 写成 (1/8)/(1/2) 而不是 0.25，是为了让"每卡份额 / 参考份额"的推导过程留在代码里。
        self._assert_param_memory(results["C4"], results["B4"], dense_ratio=(1 / 8) / (1 / 2), expert_ratio=1.0)
        self._assert_param_memory(results["H41"], results["B4"], dense_ratio=(1 / 4) / (1 / 2), expert_ratio=2.0)
        self._assert_param_memory(results["H22"], results["B4"], dense_ratio=(1 / 4) / (1 / 2), expert_ratio=2.0)

    @property
    def world_size(self) -> int:
        return 8

    def _run_layouts(self, specs: tuple[str, ...], steps: int) -> dict[str, dict[str, Any]]:
        self.create_pg("cuda")
        root = self._make_shared_tmpdir()
        try:
            hf_dir = _build_tiny_checkpoint(root)
            results: dict[str, dict[str, Any]] = {}
            for spec in specs:
                mode = parse_mode(spec)
                results[mode["name"]] = run_mode(
                    mode,
                    hf_dir,
                    steps=steps,
                    seq_len=SEQ_LEN,
                    lr=LR,
                    dispatcher="all2all",
                    fp8=False,
                    grad_norm_steps=(0,),
                    tag=self._testMethodName,
                )
                dist.barrier()
            return results
        finally:
            self._cleanup_shared_tmpdir(root)

    def _assert_training_matches(self, result: dict[str, Any], reference: dict[str, Any]) -> None:
        name, ref_name = result["mode"]["name"], reference["mode"]["name"]
        loss_diff = max_rel_diff(result["losses"], reference["losses"])
        self.assertLessEqual(loss_diff, LOSS_RTOL, f"{name} vs {ref_name}: loss curves differ by {loss_diff:.2e}")
        grad_diff = max_rel_diff(result["grad_norms"], reference["grad_norms"])
        self.assertLessEqual(grad_diff, GRAD_NORM_RTOL, f"{name} vs {ref_name}: grad norms differ by {grad_diff:.2e}")
        for step, ref_norms in reference["param_grad_norms"].items():
            norms = result["param_grad_norms"][step]
            self.assertEqual(set(norms), set(ref_norms), f"{name} vs {ref_name}: parameter sets differ at step {step}")
            worst = max(ref_norms, key=lambda key: abs(norms[key] - ref_norms[key]) / max(abs(ref_norms[key]), 1e-12))
            diff = abs(norms[worst] - ref_norms[worst]) / max(abs(ref_norms[worst]), 1e-12)
            self.assertLessEqual(
                diff,
                GRAD_NORM_RTOL,
                f"{name} vs {ref_name}: grad norm of {worst} differs by {diff:.2e} at step {step} "
                f"({norms[worst]:.6g} vs {ref_norms[worst]:.6g})",
            )

    def _assert_param_memory(
        self, result: dict[str, Any], reference: dict[str, Any], dense_ratio: float, expert_ratio: float
    ) -> None:
        # 中文注释：比较两个布局训练时记录的"每卡本地参数字节数"（`param_memory`：对每个参数取
        # DTensor 的本地分片求和，按名字里是否含 `.experts` 分成 expert / dense 两类）。
        # 断言 result 的值 ≈ reference 的值 × ratio，容差 MEMORY_RTOL（±25%）留给 FSDP 的 padding。
        # 期望比值由调用方按布局的分片数给出，见各调用点的推导。
        name, ref_name = result["mode"]["name"], reference["mode"]["name"]
        for key, ratio in (("dense_param_mib", dense_ratio), ("expert_param_mib", expert_ratio)):
            expected = reference["memory"][key] * ratio
            actual = result["memory"][key]
            self.assertAlmostEqual(
                actual,
                expected,
                delta=expected * MEMORY_RTOL,
                msg=f"{name} vs {ref_name}: {key} {actual:.1f} MiB, expected {expected:.1f} MiB per rank",
            )


@unittest.skipUnless(torch.cuda.device_count() >= 8, "requires 8 CUDA devices")
class TestDecoupledEpFsdpCheckpoint(_SharedTmpDirMixin, DeterministicDDPTestCase):
    def test_hf_export_dcp_resume_and_cross_layout_reshard(self) -> None:
        self.create_pg("cuda")
        root = self._make_shared_tmpdir()
        try:
            hf_dir = _build_tiny_checkpoint(root)
            vocab_size = Qwen3MoEConfig.from_hf(hf_dir).vocab_size
            src_bf16 = root / "source_bf16"
            if dist.get_rank() == 0:
                # `save_hf` writes bf16; compare against the bf16-cast source.
                src_bf16.mkdir()
                tensors = {k: v.to(torch.bfloat16).contiguous() for k, v in load_hf_dir(hf_dir).items()}
                save_file(tensors, str(src_bf16 / "model.safetensors"))
            dist.barrier()

            modes = [
                parse_mode(s)
                for s in ("A:ep=1", "C:ep=8,decouple=1", "C4:ep=4,decouple=1", "H41:ep=4,decouple=1,hsdp=4")
            ]
            continuous: dict[str, list[float]] = {}
            for mode in modes:
                continuous[mode["name"]] = self._check_round_trip(
                    mode, hf_dir, src_bf16, root / mode["name"], vocab_size
                )

            # Step-10 exports of every layout vs. the ep=1 baseline (same data, same optimizer).
            for mode in modes[1:]:
                if dist.get_rank() == 0:
                    assert_hf_dirs_close(
                        root / "A" / "hf_step10",
                        root / mode["name"] / "hf_step10",
                        rtol=HF_LAYOUT_RTOL,
                        atol=HF_LAYOUT_ATOL,
                        label=f"hf_step10 {mode['name']} vs A",
                    )
                dist.barrier()

            # DCP resharding across layouts: the step-5 checkpoint of `src` continues under `dst`.
            for src, dst in (("C", "A"), ("A", "C"), ("C4", "H41")):
                dst_mode = next(m for m in modes if m["name"] == dst)
                engine = build_engine(dst_mode, hf_dir, tag=f"cross_{src}")
                engine.load_dcp(root / src / "dcp_step5")
                losses = train(engine, range(5, 10), vocab_size, SEQ_LEN)
                release(engine)
                diff = max_rel_diff(losses, continuous[src])
                self.assertLessEqual(diff, LOSS_RTOL, f"cross load {src} -> {dst}: losses differ by {diff:.2e}")
                dist.barrier()
        finally:
            self._cleanup_shared_tmpdir(root)

    @property
    def world_size(self) -> int:
        return 8

    def _check_round_trip(
        self, mode: LayoutMode, hf_dir: Path, src_bf16: Path, mode_dir: Path, vocab_size: int
    ) -> list[float]:
        name = mode["name"]
        engine = build_engine(mode, hf_dir, tag="main")
        engine.from_hf(hf_path=hf_dir, strict=True)
        engine.save_hf(str(mode_dir / "hf_step0"))
        dist.barrier()
        if dist.get_rank() == 0:
            assert_hf_dirs_close(
                src_bf16, mode_dir / "hf_step0", rtol=0.0, atol=0.0, label=f"{name} hf_step0 vs source"
            )

        train(engine, range(0, 5), vocab_size, SEQ_LEN)
        engine.save_dcp(mode_dir / "dcp_step5")
        dist.barrier()
        losses_continuous = train(engine, range(5, 10), vocab_size, SEQ_LEN)
        engine.save_hf(str(mode_dir / "hf_step10"))
        dist.barrier()
        release(engine)

        resumed = build_engine(mode, hf_dir, tag="resume")
        resumed.load_dcp(mode_dir / "dcp_step5")
        losses_resumed = train(resumed, range(5, 10), vocab_size, SEQ_LEN)
        resumed.save_hf(str(mode_dir / "hf_step10_resumed"))
        dist.barrier()
        release(resumed)

        diff = max_rel_diff(losses_resumed, losses_continuous)
        self.assertLessEqual(diff, LOSS_RTOL, f"{name}: resumed losses differ from the continuous run by {diff:.2e}")
        if dist.get_rank() == 0:
            assert_hf_dirs_close(
                mode_dir / "hf_step10",
                mode_dir / "hf_step10_resumed",
                rtol=HF_RESUME_RTOL,
                atol=HF_RESUME_ATOL,
                label=f"{name} hf_step10 resumed vs continuous",
            )
        dist.barrier()
        return losses_continuous


class TestHfDirComparison:
    """CPU checks of the checkpoint comparison the GPU gates rely on."""

    @staticmethod
    def _write(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path / "model.safetensors"))
        return path

    def test_bit_exact_passes_and_one_ulp_fails(self, tmp_path: Path) -> None:
        base = {"w": torch.linspace(-1, 1, 64, dtype=torch.bfloat16)}
        lhs = self._write(tmp_path / "lhs", base)
        rhs = self._write(tmp_path / "rhs", {"w": base["w"].clone()})
        assert_hf_dirs_close(lhs, rhs, rtol=0.0, atol=0.0, label="same")

        nudged = base["w"].clone()
        # One bf16 ulp: bump the bit pattern (a float32 `nextafter` would round back to the same bf16).
        nudged[3] = (nudged[3].view(torch.int16) + 1).view(torch.bfloat16)
        rhs_ulp = self._write(tmp_path / "rhs_ulp", {"w": nudged})
        with pytest.raises(AssertionError, match="one ulp: w"):
            assert_hf_dirs_close(lhs, rhs_ulp, rtol=0.0, atol=0.0, label="one ulp")
        # The layout tolerance admits a few ulps.
        assert_hf_dirs_close(lhs, rhs_ulp, rtol=HF_LAYOUT_RTOL, atol=HF_LAYOUT_ATOL, label="tolerant")

    def test_key_and_shape_mismatches_fail(self, tmp_path: Path) -> None:
        lhs = self._write(tmp_path / "lhs", {"w": torch.ones(4, dtype=torch.bfloat16)})
        with pytest.raises(AssertionError, match="key sets differ"):
            assert_hf_dirs_close(
                lhs, self._write(tmp_path / "keys", {"v": torch.ones(4, dtype=torch.bfloat16)}), 0, 0, "keys"
            )
        with pytest.raises(AssertionError, match="expected \\(4,\\)"):
            assert_hf_dirs_close(
                lhs, self._write(tmp_path / "shape", {"w": torch.ones(8, dtype=torch.bfloat16)}), 0, 0, "shape"
            )

    def test_max_rel_diff(self) -> None:
        assert max_rel_diff([1.0, 2.0], [1.0, 2.0]) == 0.0
        assert max_rel_diff([1.0, 2.2], [1.0, 2.0]) == pytest.approx(0.1)
        with pytest.raises(ValueError, match="length mismatch"):
            max_rel_diff([1.0], [1.0, 2.0])
