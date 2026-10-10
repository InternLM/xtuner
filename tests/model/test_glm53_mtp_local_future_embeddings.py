"""Real XTuner GLM53 supplied-embedding regression without optional HF generation imports."""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.glm53.glm53 import Glm53TextMoEConfig
from xtuner.v1.model.moe.glm53.nope_dsa_mla import NoPEDSAMLAConfig
from xtuner.v1.module.decoder_layer.mhc import MHCConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig


def run_model(depth, objective, legacy):
    torch.manual_seed(0)
    cache_config = {"use_local_future_embeddings": False} if legacy else {}
    mtp_config = MTPConfig(num_layers=depth, share_weights=True, loss_type=objective, **cache_config)
    assert mtp_config.use_local_future_embeddings is (not legacy)
    cfg = Glm53TextMoEConfig(
        compile_cfg=False,
        vocab_size=200,
        pad_token_id=0,
        eos_token_id=1,
        hf_eos_token_id=[1],
        num_hidden_layers=1,
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
            freeze_dsa_indexer=True,
        ),
        glm53_layer_types=["deepseek_sparse_attention"],
        mhc=MHCConfig(hc_mult=4, hc_sinkhorn_iters=4),
        router=NoAuxRouterConfig(
            n_group=1, topk_group=1, scoring_func="sigmoid", norm_topk_prob=True, router_scaling_factor=1.0
        ),
        mtp_config=mtp_config,
        lm_loss_cfg=CELossConfig(mode="chunk", chunk_size=64),
        dispatcher=None,
        ep_size=1,
    )
    model = cfg.build().cuda().to(torch.bfloat16)
    model.init_weights()
    assert len(model.mtp_block.layers) == 1
    assert model.mtp_block.layers[0].decoder_layer.use_mhc is False
    ids = torch.randint(2, 200, (1, 64), device="cuda")
    source = torch.randn(1, 64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ctx = SequenceContext.from_input_ids((ids[:, :5], ids[:, 5:]), device="cuda")
    ctx.inputs_embeds = source
    ctx.input_ids = None  # Match the composed model's supplied-embedding path for the main stack too.
    labels = ids.clone()
    labels[:, -1] = -100
    loss_ctx = model.build_loss_ctx_batch([{"seq_ctx": ctx, "shifted_labels": labels}])[0]
    if objective == "e2e_tv":
        assert loss_ctx["mtp"] is None and loss_ctx["mtp_e2e_tv"] is not None
        assert loss_ctx["mtp_e2e_tv"].loss_cfg.num_steps == depth
    else:
        assert len(loss_ctx["mtp"]) == depth
    output = model(seq_ctx=ctx, loss_ctx=loss_ctx)
    loss = output.loss + output.mtp_loss + output.balancing_loss
    assert torch.isfinite(loss) and output.mtp_loss.item() > 0
    loss.backward()
    gradients = {
        name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }
    for name, parameter in model.mtp_block.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.isfinite().all(), name
    assert source.grad is not None and source.grad.isfinite().all()
    return output.loss.detach().cpu(), output.mtp_loss.detach().cpu(), gradients, source.grad.detach().cpu().clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("depth", [1, 7])
@pytest.mark.parametrize("objective", ["ce", "e2e_tv"])
def test_real_glm53_default_cache_matches_legacy_loss_and_all_gradients(depth, objective):
    old_lm, old_mtp, old_gradients, old_source = run_model(depth, objective, True)
    lm, mtp, gradients, source = run_model(depth, objective, False)
    torch.testing.assert_close(lm, old_lm, rtol=0, atol=0)
    torch.testing.assert_close(mtp, old_mtp, rtol=0, atol=0)
    assert gradients.keys() == old_gradients.keys()
    for name, gradient in gradients.items():
        if gradient is None:
            assert old_gradients[name] is None, name
        else:
            assert gradient.isfinite().all(), name
            torch.testing.assert_close(gradient, old_gradients[name], msg=lambda msg: f"{name}: {msg}")
    # SP1's BF16 embedding gradient accumulation order may differ; use BF16 tolerances.
    torch.testing.assert_close(source, old_source)
