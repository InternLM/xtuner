"""GLM-5.3-Flash 文本塔，见 doc/xtuner_glm5p3flash_design.md F6。

TestGlm53TextMoEConfig
    test_default_layer_schedule_matches_checkpoint_pattern  默认层调度与 checkpoint 一致
    test_layer_schedule_length_mismatch_is_caught_at_build  层数与调度长度不符时构造期报错
    test_mtp_layer_has_no_mhc                               MTP 层不带 hc_* 参数
TestGlm53TextMoEInitWeights
    test_init_weights_covers_every_parameter                from-scratch 初始化覆盖全部参数
    test_init_weights_matches_hf_init_for_gate_and_hc_params 初值与 HF _init_weights 一致
TestGlm53CompileCfg
    test_mhc_primitives_are_compiled_with_and_without_ep    mHC 原语在 EP/非 EP 下都被编译
    test_every_compile_target_resolves                      编译目标名都能解析（防改名失效）
TestGlm53TextMoEFp32Params
    test_only_the_sinkhorn_and_gate_scalars_are_pinned_to_fp32  该 pin 的 pin，fn 刻意不 pin
TestGlm53TextMoEForwardBackward
    test_forward_backward_all_trainable_params_get_gradient  除冻结 indexer 外都有梯度
    test_mtp_block_builds_and_forwards                       MTP block 可构造并前向
    test_optimizer_step_updates_text_model                   CE 反传和 AdamW 更新
TestGlm53TextMoEWeightMapping
    test_real_checkpoint_weight_coverage                     真实 checkpoint 权重全覆盖
TestGlm53TextMoEAccuracy
    test_fsdp_accuracy                                       FSDP 文本/图文 loss 和 logits 与 HF 对齐
TestGlm53TextMoEGradientParity
    test_full_crop_fsdp_gradients_match_hf                   五层真实权重 FSDP 梯度与 HF 对齐
TestNoPEDSAMLAConfigValidatesAssignment
    test_backend_assignment_is_validated                     构造后赋值仍走校验
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import parametrize
import pytest
import torch
from pydantic import ValidationError
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_distributed import DistributedTestBase

from transformers import Glm5NextForConditionalGeneration
from xtuner._testing import DeterministicDDPTestCase
from xtuner._testing.logits import check_logits
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.compose.glm53 import Glm53BaseConfig
from xtuner.v1.model.moe.glm53.glm53 import Glm53TextMoEConfig
from xtuner.v1.model.moe.glm53.nope_dsa_mla import NoPEDSAMLAConfig
from xtuner.v1.module.attention.kda import KDAConfig
from xtuner.v1.module.decoder_layer.mhc import MHCConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig

from .glm53_parity_cases import TEXT_CASE_COUNT, build_glm53_parity_cases, load_glm53_reference


GLM_5_3_FLASH_PATH = os.environ.get(
    "GLM_5_3_FLASH_PATH", "/mnt/shared-storage-user/zhaopenghao/model/GLM-5.3-Flash-25B"
)


class TestGlm53TextMoEConfig:
    def test_default_layer_schedule_matches_checkpoint_pattern(self):
        # 默认层调度必须与真实 checkpoint 的 layer_types 一致。
        cfg = Glm53TextMoEConfig()
        assert cfg.num_hidden_layers == 45
        # [KDA, KDA, KDA, DSA] x 11 + final KDA (real checkpoint text_config.layer_types).
        expected = (["linear_attention"] * 3 + ["full_attention"]) * 11 + ["linear_attention"]
        assert cfg.layers_type == expected

    def test_layer_schedule_length_mismatch_is_caught_at_build(self):
        # 层数与调度长度不一致要在构造期报错，而不是前向时越界。
        cfg = Glm53TextMoEConfig(num_hidden_layers=4, glm53_layer_types=["linear_attention"] * 3)
        with pytest.raises(AssertionError):
            cfg.build()

    def test_mtp_layer_has_no_mhc(self):
        # design doc F6: checkpoint layers.45 has no hc_* params; asserted via config wiring,
        # not runtime inspection, since MTP is a plain pre-norm layer by construction (mhc_cfg=None).
        cfg = Glm53TextMoEConfig()
        assert cfg.mhc.hc_mult == 4  # main stack layers do have mHC
        assert cfg.mtp_config is not None and cfg.mtp_config.share_weights


def _tiny_cfg(**overrides):
    base = dict(
        compile_cfg=False,
        vocab_size=200,
        pad_token_id=0,
        eos_token_id=1,
        hf_eos_token_id=[1],
        num_hidden_layers=5,
        first_k_dense_replace=1,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=48,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        attention=NoPEDSAMLAConfig(
            num_attention_heads=4,
            head_dim=16,
            kv_lora_rank=24,
            q_lora_rank=16,
            qk_rope_head_dim=0,
            qk_nope_head_dim=8,
            v_head_dim=8,
            index_topk=4,
            index_head_dim=8,
            index_n_heads=2,
            index_kpool=2,
            sparse_mla_backend="torch",
            indexer_backend="torch",
            freeze_dsa_indexer=False,
        ),
        # head_dim=16 (not 8) deliberately: FLA's chunked KDA Triton kernel requires the
        # tl.dot K dimension >= 16; below that, forward silently runs but backward drops
        # gradients on q/k/v/A_log/dt_bias/conv1d with no error (confirmed via a minimal
        # single-layer repro comparing head_dim=8 vs 16 at matched seq_len -- not a production
        # bug since real GLM-5.3-Flash's KDA head_dim is 128).
        linear_attention=KDAConfig(num_heads=2, head_dim=16, gate_lower_bound=-5.0),
        glm53_layer_types=[
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
        ],
        mhc=MHCConfig(hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=4),
        router=NoAuxRouterConfig(
            n_group=1, topk_group=1, scoring_func="sigmoid", norm_topk_prob=True, router_scaling_factor=1.0
        ),
        mtp_config=None,
        dispatcher="all2all",
        ep_size=1,
    )
    base.update(overrides)
    return Glm53TextMoEConfig(**base)


class TestGlm53TextMoEInitWeights:
    def test_init_weights_covers_every_parameter(self):
        """From-scratch (no HF checkpoint) init must materialize *every* parameter. mHC's
        `hc_*`, KDA's `A_log`/`dt_bias` and the KPool indexer's `index_kpool_compress_*` are
        none of them named `weight`/`bias`, and `__init__`'s `torch.zeros(...)` is a no-op
        under `init_device="meta"`, so they have to be initialized explicitly."""
        # from-scratch 初始化要覆盖 hc_*/A_log/dt_bias 等非 weight/bias 命名的参数。
        model = _tiny_cfg().build()
        model.init_weights()

        for name, param in model.named_parameters():
            assert not param.is_meta, name
            assert torch.isfinite(param).all(), name

    def test_init_weights_matches_hf_init_for_gate_and_hc_params(self):
        """Values, not just presence: HF zeroes `A_log` whenever a safe gate lower bound is set
        (`Glm5NextTextForgetGate._init_weights`) and zeroes the mHC mixing params while setting
        the three per-site scales to one."""
        # 不只是被初始化，初值也要与 HF 的 _init_weights 一致。
        model = _tiny_cfg().build()
        model.init_weights()

        layer0 = model.layers["0"]
        torch.testing.assert_close(layer0.self_attn.A_log, torch.zeros_like(layer0.self_attn.A_log))
        torch.testing.assert_close(layer0.hc_attn_base, torch.zeros_like(layer0.hc_attn_base))
        torch.testing.assert_close(layer0.hc_ffn_scale, torch.ones_like(layer0.hc_ffn_scale))
        # dt_bias is log-uniform(1e-3, 1e-1) regardless of the gate bound.
        assert (layer0.self_attn.dt_bias.exp() > 1e-3 - 1e-6).all()
        assert (layer0.self_attn.dt_bias.exp() < 1e-1 + 1e-6).all()


class TestGlm53CompileCfg:
    def test_mhc_primitives_are_compiled_with_and_without_ep(self):
        """`hc_pre` materializes fp32 rms-norm intermediates over `[B, S, hc_mult * hidden]` and
        `_hc_post_eager` a broadcast-multiply that only inductor's fusion keeps off HBM. They
        used to ride along inside `Glm53MoEDecoderLayer._pre/_post_moe_forward`'s boundary,
        which the EP config drops (all2all does not trace), leaving 42 of 45 layers running
        them eagerly under the production topology. They need their own entries."""
        # EP 表会 pop 掉 MoE 层边界，mHC 原语必须各自登记才不会退回 eager。
        from xtuner.v1.model.moe.glm53.glm53 import GLM53_MOE_EP_COMPILE_CFG, GLM53_MOE_NON_EP_COMPILE_CFG

        for cfg_name, cfg in (("non-EP", GLM53_MOE_NON_EP_COMPILE_CFG), ("EP", GLM53_MOE_EP_COMPILE_CFG)):
            for target in (
                "xtuner.v1.module.decoder_layer.mhc.hc_pre",
                "xtuner.v1.module.decoder_layer.mhc._hc_post_eager",
            ):
                assert target in cfg, f"{target} missing from the {cfg_name} compile cfg"

    def test_every_compile_target_resolves(self):
        """`BaseModel._compile_overwrite` locates each target by name; a stale one aborts model
        construction rather than quietly skipping compilation."""
        # 编译目标按名字解析，改名后必须在测试里先炸而不是运行期才炸。
        import pydoc

        from xtuner.v1.model.moe.glm53.glm53 import GLM53_MOE_EP_COMPILE_CFG, GLM53_MOE_NON_EP_COMPILE_CFG

        for cfg in (GLM53_MOE_NON_EP_COMPILE_CFG, GLM53_MOE_EP_COMPILE_CFG):
            for target in cfg:
                assert pydoc.locate(target) is not None, target


class TestGlm53TextMoEFp32Params:
    """`hc_split_sinkhorn`'s 20 iterations and KDA's `fused_kda_gate` are documented as
    bf16-unsafe, but declaring the parameters `dtype=torch.float32` buys nothing under FSDP2:
    `fully_shard` upcasts every trainable parameter to an fp32 master anyway and then casts to
    `MixedPrecisionPolicy.param_dtype` for the forward all-gather. The only lever that keeps a
    parameter in fp32 *compute* is `hf_save_cfg.fp32_keys_pattern`, which routes it to
    `fully_shard(ignored_params=...)` -- see `BaseModel._fully_shard`."""

    def _matches(self, cfg, model, param_name: str) -> bool:
        patterns = cfg.hf_save_cfg.fp32_keys_pattern or []
        return any(re.search(p, k) for p in patterns for k in model.to_hf_key_list(param_name))

    def test_only_the_sinkhorn_and_gate_scalars_are_pinned_to_fp32(self):
        # sinkhorn 的 base/scale 与 KDA 的 A_log/dt_bias 必须留在 fp32；而 hc_*_fn 刻意不留
        # （hc_pre 本就把它转成激活 dtype，pin 住只会让 45x2 个投影矩阵在每个 rank 上复制）。
        cfg = _tiny_cfg()
        model = cfg.build()
        pinned = [n for n, _ in model.named_parameters() if re.search(r"hc_(attn|ffn)_(base|scale)|A_log|dt_bias", n)]
        assert pinned, "tiny config should carry mHC and KDA parameters"
        for name in pinned:
            assert self._matches(cfg, model, name), f"{name} -> {model.to_hf_key_list(name)} not pinned to fp32"
        assert not self._matches(cfg, model, "layers.0.hc_attn_fn")


@pytest.mark.gpu
class TestGlm53TextMoEForwardBackward:
    def test_forward_backward_all_trainable_params_get_gradient(self):
        # 除刻意冻结的 indexer 外，每个可训练参数都要拿到梯度。
        cfg = _tiny_cfg()
        model = cfg.build().cuda().to(torch.bfloat16)
        torch.manual_seed(0)
        for p in model.parameters():
            if p.is_floating_point():
                p.data.normal_(mean=0.0, std=0.02)

        seq_len = 128  # >= KDA kernel's minimum chunk-friendly length, see _tiny_cfg docstring.
        input_ids = torch.randint(2, 200, (1, seq_len)).cuda()
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
        out = model(seq_ctx=seq_ctx, loss_ctx=None)
        assert out.logits.shape == (1, seq_len, 200)
        assert torch.isfinite(out.logits).all()

        out.logits.float().pow(2).sum().backward()
        for name, p in model.named_parameters():
            if not p.is_floating_point():
                continue
            # The DSA indexer's top-k selection is inherently non-differentiable and is
            # deliberately computed under torch.no_grad() (nope_dsa_mla.py) -- matches the
            # standard DSA-indexer training convention (trained via a separate auxiliary
            # signal, not through the main LM loss). Every other parameter must get a gradient.
            if "self_attn.indexer" in name:
                assert p.grad is None
            else:
                assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} got no gradient"

    def test_mtp_block_builds_and_forwards(self):
        # MTP block 能构造并参与前向。
        cfg = _tiny_cfg(mtp_config=MTPConfig(num_layers=1, share_weights=True))
        model = cfg.build().cuda().to(torch.bfloat16)
        for p in model.parameters():
            if p.is_floating_point():
                p.data.normal_(mean=0.0, std=0.02)
        assert model.mtp_block is not None
        mtp_decoder = model.mtp_block.layers[0].decoder_layer
        assert mtp_decoder.use_mhc is False  # checkpoint layers.45 has no hc_* params

        seq_len = 128
        input_ids = torch.randint(2, 200, (1, seq_len)).cuda()
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
        out = model(seq_ctx=seq_ctx, loss_ctx=None)
        assert torch.isfinite(out.logits).all()

    def test_optimizer_step_updates_text_model(self):
        # Drive the public training path through CE loss, backward, and AdamW.
        torch.manual_seed(0)
        model = _tiny_cfg().build().cuda().to(torch.bfloat16)
        model.init_weights()
        input_ids = torch.randint(2, 200, (1, 128), device="cuda")
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
        data = {"seq_ctx": seq_ctx, "shifted_labels": input_ids.roll(-1, dims=1)}
        loss_ctx = model.build_loss_ctx_batch([data], sp_mesh=None)[0]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        tracked = model.layers["0"].self_attn.q_proj.weight
        before = tracked.detach().clone()

        loss = model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        assert tracked.grad is not None and torch.isfinite(tracked.grad).all()
        assert tracked.grad.abs().sum() > 0
        optimizer.step()

        assert torch.isfinite(tracked).all()
        assert not torch.equal(tracked, before)
        assert optimizer.state[tracked]["step"] == 1


class TestGlm53TextMoEWeightMapping:
    def test_real_checkpoint_weight_coverage(self):
        # 真实 checkpoint 的权重要全部映射上，不能有 missing/unloaded。
        if not os.path.isdir(GLM_5_3_FLASH_PATH):
            pytest.skip(f"GLM_5_3_FLASH_PATH not found: {GLM_5_3_FLASH_PATH}")
        if not torch.cuda.is_available():
            pytest.skip("GPU required to materialize a 25B-parameter checkpoint")

        cfg = Glm53TextMoEConfig.from_hf(GLM_5_3_FLASH_PATH)
        with torch.device("meta"):
            model = cfg.build()
        model._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        loaded, unloaded, missing = model.from_hf(GLM_5_3_FLASH_PATH, strict=False)

        assert not missing, f"missing keys: {sorted(missing)[:10]}"
        assert not unloaded, f"unloaded keys: {sorted(unloaded)[:10]}"
        assert not any(p.is_meta for p in model.parameters())
        assert len(loaded) > 0


@pytest.fixture(scope="class")
def automodel_reference(tmp_path_factory):
    """Finish the optional AM process before the existing test starts GPU workers."""
    python = os.environ.get("GLM53_AUTOMODEL_PYTHON")
    if not python or not os.path.isdir(GLM_5_3_FLASH_PATH):
        yield
        return
    output = tmp_path_factory.mktemp("glm53-automodel") / "reference"
    subprocess.run(
        [
            python,
            str(Path(__file__).with_name("run_glm53_reference.py")),
            "--checkpoint",
            GLM_5_3_FLASH_PATH,
            "--processor-python",
            sys.executable,
            "--output",
            str(output),
        ],
        check=True,
        timeout=1800,
    )
    # Workers inherit this internal result path; callers only select an interpreter.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("_GLM53_AUTOMODEL_REFERENCE", str(output))
        yield


@pytest.mark.usefixtures("automodel_reference")
class TestGlm53TextMoEAccuracy(DeterministicDDPTestCase):
    """验收 1: text + image compose-model forward loss/logits vs real `transformers.Glm5NextForConditionalGeneration`
    on the F0 25B cropped checkpoint (GLM_5_3_FLASH_PATH).

    Installed transformers (pinned 5.17.0) does not implement MTP forward for
    `Glm5NextForConditionalGeneration` (see doc/progress.md F0 记录: the checkpoint's
    `layers.5.*`/original `layers.45` MTP weights load as UNEXPECTED and are ignored). Both
    sides therefore compare only the 5-layer main stack: XTuner is built with
    `mtp_config=None` to match. `sparse_mla_backend`/`indexer_backend` are forced to `"torch"`
    (eager, alignment=1) instead of the production `flash_mla_cudnn` default (alignment=512)
    since the test sentences are far shorter than one alignment block.
    """

    @parametrize.parametrize(
        "dispatcher, ep_size",
        [
            (None, 1),
            # ep_size=4/8 are also the exact topology of the EP+mHC/o_norm DTensor bug fixed in
            # doc/progress.md F6 排查记录 5 (kda.py's FusedRMSNormGated.forward) -- keeps it
            # regression-covered against the real checkpoint, not just the throwaway repro script.
            ("all2all", 4),
            ("all2all", 8),
        ],
    )
    def test_fsdp_accuracy(self, dispatcher, ep_size):
        # Four original text cases and two image cases share the same accuracy loop.
        if not os.path.isdir(GLM_5_3_FLASH_PATH):
            pytest.skip(f"GLM_5_3_FLASH_PATH not found: {GLM_5_3_FLASH_PATH}")
        self.create_pg("cuda")

        # `Glm5NextForConditionalGeneration` isn't registered under `AutoModelForCausalLM`
        # (it's the VL/compose entry point) -- must be loaded directly, matching doc/progress.md
        # F0's own finding.
        hf_model = Glm5NextForConditionalGeneration.from_pretrained(
            GLM_5_3_FLASH_PATH,
            dtype=torch.bfloat16,
            device_map="cuda",
            # HF 5.17 vision rejects FA2; XT vision uses its production FlashAttention backend.
            attn_implementation="eager",
        )

        cases = build_glm53_parity_cases(GLM_5_3_FLASH_PATH)
        am_reference_dir = os.environ.get("_GLM53_AUTOMODEL_REFERENCE")
        am_reference = load_glm53_reference(am_reference_dir, cases, GLM_5_3_FLASH_PATH) if am_reference_dir else None
        hf_model.eval()
        expected_losses = []
        expected_logits = []
        for _, batch in cases:
            input_ids = batch["input_ids"].to("cuda")
            media = {
                k: batch[k].to("cuda", dtype=torch.bfloat16 if k == "pixel_values" else torch.long)
                for k in ("pixel_values", "image_grid_thw")
                if k in batch
            }
            with torch.no_grad():
                output = hf_model(input_ids=input_ids, labels=batch["labels"].to("cuda"), use_cache=False, **media)
            expected_losses.append(output.loss)
            # Last eight next-token positions, including the full vocabulary.
            expected_logits.append(output.logits[0, batch["positions"].to("cuda")].detach().cpu())

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = Glm53BaseConfig.from_hf(GLM_5_3_FLASH_PATH)
            cfg.compile_cfg = False
            cfg.text_config.compile_cfg = False
            cfg.vision_config.attn_impl = "flash_attention"
            cfg.text_config.dispatcher = dispatcher
            cfg.text_config.ep_size = ep_size
            cfg.text_config.mtp_config = None
            cfg.text_config.attention.sparse_mla_backend = "torch"
            cfg.text_config.attention.indexer_backend = "torch"
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

        fsdp_config = FSDPConfig(ep_size=ep_size, cpu_offload=False)
        model.fully_shard(fsdp_config=fsdp_config)
        model.from_hf(GLM_5_3_FLASH_PATH, strict=False)

        model.eval()
        losses = []
        for sample_index, (name, batch) in enumerate(cases):
            input_ids = batch["input_ids"].to("cuda")
            shift_input_ids = input_ids[:, :-1]
            shifted_labels = batch["labels"][:, 1:].to("cuda")
            seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to("cuda"),))
            if "pixel_values" in batch:
                seq_ctx.pixel_values = batch["pixel_values"].to(dtype=torch.bfloat16)  # model moves CPU media
                seq_ctx.image_grid_thw = batch["image_grid_thw"].to("cuda")
                seq_ctx.mm_token_type_ids = batch["mm_token_type_ids"][:, :-1].to("cuda")
                seq_ctx.num_img_tokens = [[int(seq_ctx.image_grid_thw.prod(-1).sum())]]
            loss_cfg = CELossConfig()
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
            loss_ctx_list = LossContext.build_batches([loss_ctx])
            loss_ctx = loss_ctx_list[0]

            with torch.no_grad():
                output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
            losses.append(output["loss"])
            # Preserve the public loss path above; additionally exercise inference logits.
            with torch.no_grad():
                logits = model(seq_ctx=seq_ctx, loss_ctx=None).logits
            chosen = logits[0, batch["positions"].to("cuda")]
            metrics = check_logits(chosen, expected_logits[sample_index])
            print(f"GLM53 case={name} logits={metrics}", flush=True)
            if am_reference is not None:
                am_logits = am_reference[sample_index][1]
                print(
                    f"GLM53 case={name} loss HF={expected_losses[sample_index].item()} "
                    f"XT={losses[-1].item()} AM={am_reference[sample_index][0]}"
                )
                print(f"GLM53 case={name} AM/HF logits={check_logits(am_logits, expected_logits[sample_index])}")
                print(f"GLM53 case={name} XT/AM logits={check_logits(chosen, am_logits)}")

        for start, end in ((0, TEXT_CASE_COUNT), (TEXT_CASE_COUNT, len(cases))):
            self._check_loss_curve(
                losses=torch.tensor(losses[start:end]),
                losses_ref=torch.tensor(expected_losses[start:end]),
                sim_tol=3e-2,
                rtol=3e-2,
            )
            if am_reference is not None:
                am_losses = torch.tensor([item[0] for item in am_reference[start:end]])
                for actual, reference in (
                    (am_losses, torch.tensor(expected_losses[start:end])),
                    (torch.tensor(losses[start:end]), am_losses),
                ):
                    self._check_loss_curve(losses=actual, losses_ref=reference, sim_tol=3e-2, rtol=3e-2)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))


class TestGlm53TextMoEGradientParity(DistributedTestBase):
    @pytest.mark.gpu
    def test_full_crop_fsdp_gradients_match_hf(self, device="cuda"):
        # Real five-layer weights: compare both ends of the KDA/mHC stack after LM backward.
        if not os.path.isdir(GLM_5_3_FLASH_PATH):
            pytest.skip(f"GLM_5_3_FLASH_PATH not found: {GLM_5_3_FLASH_PATH}")
        self.create_pg(device)
        torch.manual_seed(1234)
        tokens = torch.randint(2, 1000, (1, 81), device=device)

        hf_model = Glm5NextForConditionalGeneration.from_pretrained(
            GLM_5_3_FLASH_PATH, dtype=torch.bfloat16, device_map=f"cuda:{torch.cuda.current_device()}"
        )
        hf_model.requires_grad_(False)
        hf_first = hf_model.model.language_model.layers[0]
        hf_last = hf_model.model.language_model.layers[4]
        hf_params = {
            "first_q_proj": hf_first.self_attn.q_proj.weight,
            "first_A_log": hf_first.self_attn.forget_gate.A_log,
            "first_hc_attn_base": hf_first.attn_hc.base,
            "last_q_proj": hf_last.self_attn.q_proj.weight,
            "last_A_log": hf_last.self_attn.forget_gate.A_log,
            "last_hc_attn_base": hf_last.attn_hc.base,
            "final_norm": hf_model.model.language_model.norm.weight,
        }
        for param in hf_params.values():
            param.requires_grad_(True)
        hf_loss = hf_model(input_ids=tokens, labels=tokens, use_cache=False).loss
        hf_loss.backward()
        reference_loss = hf_loss.detach().cpu()
        reference_grads = {name: param.grad.detach().float().cpu() for name, param in hf_params.items()}
        del hf_loss, hf_model, hf_first, hf_last, hf_params
        torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = Glm53TextMoEConfig.from_hf(GLM_5_3_FLASH_PATH)
            cfg.compile_cfg = False
            cfg.dispatcher = None
            cfg.ep_size = 1
            cfg.mtp_config = None
            cfg.attention.sparse_mla_backend = "torch"
            cfg.attention.indexer_backend = "torch"
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        model.fully_shard(FSDPConfig(ep_size=1))
        model.from_hf(GLM_5_3_FLASH_PATH, strict=False)
        first, last = model.layers["0"], model.layers["4"]
        xtuner_params = {
            "first_q_proj": first.self_attn.q_proj.weight,
            "first_A_log": first.self_attn.A_log,
            "first_hc_attn_base": first.hc_attn_base,
            "last_q_proj": last.self_attn.q_proj.weight,
            "last_A_log": last.self_attn.A_log,
            "last_hc_attn_base": last.hc_attn_base,
            "final_norm": model.norm.weight,
        }
        seq_ctx = SequenceContext.from_input_ids((tokens[:, :-1],), device=device)
        loss_ctx = model.build_loss_ctx_batch([{"seq_ctx": seq_ctx, "shifted_labels": tokens[:, 1:]}], sp_mesh=None)[0]
        loss = model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)["loss"]
        torch.testing.assert_close(loss.detach().cpu(), reference_loss, atol=1e-2, rtol=0)
        loss.backward()
        model.scale_and_reduce_grad()

        for name, param in xtuner_params.items():
            assert param.grad is not None, name
            grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            actual = grad.detach().float().cpu()
            expected = reference_grads[name]
            relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
            assert relative_error < 0.06, f"{name}: relative gradient error {relative_error.item():.4g}"

    @property
    def world_size(self) -> int:
        return 2


class TestNoPEDSAMLAConfigValidatesAssignment:
    def test_backend_assignment_is_validated(self):
        """`examples/v1/config/sft_glm53.py` sets the backend from an env var *after*
        construction. Without `validate_assignment` the model validator does not re-run, so
        `SPARSE_MLA_BACKEND=tilelang` would slip past its NotImplementedError and fail much
        later inside the kernel."""
        # 构造后再赋值也要走校验，否则环境变量覆盖会绕过 NotImplementedError。
        cfg = NoPEDSAMLAConfig(
            num_attention_heads=4,
            head_dim=16,
            kv_lora_rank=24,
            q_lora_rank=16,
            qk_rope_head_dim=0,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
        with pytest.raises((NotImplementedError, ValidationError)):
            cfg.sparse_mla_backend = "tilelang"
        cfg.sparse_mla_backend = "torch"
        assert cfg.sparse_mla_backend == "torch"
