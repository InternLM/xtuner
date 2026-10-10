"""CPU coverage of one physical GLM5.3 MTP layer reused at seven logical depths."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm53.glm53 import Glm53TextMoEConfig
from xtuner.v1.module.mtp import MTPBlock, MTPConfig


@pytest.mark.parametrize("depth", [1, 7])
def test_glm53_builder_and_hf_mapping_keep_one_physical_layer(depth):
    cfg = Glm53TextMoEConfig(
        compile_cfg=False, ep_size=1, dispatcher=None, mtp_config=MTPConfig(num_layers=depth, share_weights=True)
    )
    # Construction creates a CUDA offload stream even on meta; only that runtime handle is mocked.
    with patch("torch.cuda.Stream"), torch.device("meta"):
        model = cfg.build()
    assert len(model.mtp_block.layers) == 1
    assert model.mtp_block.mtp_config.num_layers == depth
    keys = [key for name, _ in model.mtp_block.named_parameters() for key in model.to_hf_key_list(f"mtp_block.{name}")]
    assert all("model.language_model.layers.45." in key for key in keys)
    assert not any("hc_" in name for name, _ in model.mtp_block.named_parameters())
    assert model.to_hf_key_list("mtp_block.layers.0.eh_proj.weight") == ["model.language_model.layers.45.eh_proj.weight"]


class _RecurrentLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(self, hidden_states, *, future_embeddings, position_embeddings, seq_ctx):
        self.calls.append(seq_ctx.input_ids.clone())
        output = self.weight * hidden_states + future_embeddings
        return dict(hidden_states=output, router_logits=None, router_weights=None, router_topk_ids=None)


def test_seven_recurrent_predictions_accumulate_gradient_on_the_same_parameter():
    layer = _RecurrentLayer()
    block = MTPBlock(mtp_config=MTPConfig(num_layers=7, share_weights=True), mtp_layers=[layer])
    ctx = SequenceContext.from_input_ids((torch.arange(16).view(1, -1),), device="cpu")
    start = torch.ones(1, 16, 1, requires_grad=True)
    outputs = block(
        start,
        embed_tokens_fn=lambda ids: ids.float().unsqueeze(-1),
        position_embeddings=(torch.empty(0), torch.empty(0)),
        seq_ctx=ctx,
    )
    assert len(outputs) == len(layer.calls) == 7
    assert list(block.parameters()) == [layer.weight]
    loss = torch.stack([out["hidden_states"].square().mean() for out in outputs]).mean() * 0.1
    loss.backward()
    assert start.grad is not None and layer.weight.grad.isfinite() and layer.weight.grad.abs() > 0
    # A differentiable reference unroll uses this very same parameter at each depth.
    weight = layer.weight.detach().clone().requires_grad_()
    state = start.detach()
    terms = []
    for ids in layer.calls:
        state = weight * state + ids.float().unsqueeze(-1)
        terms.append(state.square().mean())
    (torch.stack(terms).mean() * 0.1).backward()
    torch.testing.assert_close(layer.weight.grad, weight.grad)
